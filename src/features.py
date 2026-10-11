"""Engenharia de features para previsao de churn.

Definicao do problema (sem vazamento de dados)
----------------------------------------------
Para uma *data de corte* T:

* **Populacao (base ativa)**: clientes com ao menos uma compra valida (nao cancelada) nos
  ``JANELA_ATIVOS_DIAS`` (180) dias anteriores a T. Quem ja abandonou ha muito tempo nao entra:
  prever "churn" de quem sumiu ha um ano seria trivial e nao serve a uma acao de retencao.
* **Features**: calculadas usando SOMENTE pedidos com data <= T.
* **Alvo** ``churn``: 1 se o cliente NAO fez nenhuma compra valida em (T, T + 90 dias]; 0 caso contrario.

Cada cliente gera uma linha por data de corte (``ml.features_churn`` tem chave ``cliente_id, data_corte``).
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config

log = logging.getLogger(__name__)

FEATURES_NUMERICAS = [
    "recencia_dias", "frequencia_total", "freq_90d", "freq_prev_90d", "tendencia_freq",
    "intervalo_medio_dias", "razao_recencia_intervalo", "valor_total", "ticket_medio",
    "itens_por_pedido", "categorias_distintas", "desconto_medio_pct", "pct_cancelados",
    "pct_devolvidos", "nota_media", "frete_medio", "pct_fim_de_semana", "tempo_de_casa_dias", "idade",
]
FEATURES_CATEGORICAS = ["segmento", "canal_aquisicao", "regiao"]
FEATURES_BOOLEANAS = ["aceita_marketing"]
ALVO = "churn"


def carregar_base_dw(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Le do DW: pedidos (com desconto medio), pares pedido-categoria e clientes (com regiao)."""
    pedidos = pd.read_sql(text("""
        SELECT p.pedido_id, p.cliente_id, p.data_id AS data, p.status, p.frete, p.nota_avaliacao,
               p.qtd_itens, p.valor_total, d.eh_fim_de_semana,
               AVG(i.desconto_pct)::float AS desconto_medio
        FROM dw.fato_pedido p
        JOIN dw.fato_item_pedido i USING (pedido_id)
        JOIN dw.dim_data d ON d.data_id = p.data_id
        GROUP BY p.pedido_id, d.eh_fim_de_semana
    """), engine, parse_dates=["data"])
    categorias = pd.read_sql(text("""
        SELECT DISTINCT i.pedido_id, pr.categoria
        FROM dw.fato_item_pedido i JOIN dw.dim_produto pr USING (produto_id)
    """), engine)
    clientes = pd.read_sql(text("""
        SELECT c.cliente_id, c.data_cadastro, c.data_nascimento, c.segmento, c.canal_aquisicao,
               c.aceita_marketing, COALESCE(l.regiao, 'nao_informada') AS regiao
        FROM dw.dim_cliente c LEFT JOIN dw.dim_localidade l USING (localidade_id)
    """), engine, parse_dates=["data_cadastro", "data_nascimento"])
    for col in ("valor_total", "frete"):
        pedidos[col] = pedidos[col].astype(float)
    pedidos["nota_avaliacao"] = pd.to_numeric(pedidos["nota_avaliacao"], errors="coerce")
    return pedidos, categorias, clientes


def construir_features(pedidos: pd.DataFrame, categorias: pd.DataFrame, clientes: pd.DataFrame,
                       data_corte: date, horizonte_dias: int = config.HORIZONTE_DIAS) -> pd.DataFrame:
    """Monta features + alvo para uma data de corte. Funcao pura (nao acessa o banco)."""
    corte = pd.Timestamp(data_corte)
    fim_alvo = corte + pd.Timedelta(days=horizonte_dias)
    if fim_alvo > pd.Timestamp(config.DATA_FIM):
        raise ValueError(f"Janela do alvo ({fim_alvo.date()}) ultrapassa o fim dos dados ({config.DATA_FIM})")

    hist = pedidos[pedidos["data"] <= corte]                       # tudo que se sabia em T
    validos = hist[hist["status"] != "cancelado"].sort_values(["cliente_id", "data"])
    futuro = pedidos[(pedidos["data"] > corte) & (pedidos["data"] <= fim_alvo) & (pedidos["status"] != "cancelado")]

    g = validos.groupby("cliente_id")
    f = pd.DataFrame({
        "recencia_dias": (corte - g["data"].max()).dt.days,
        "frequencia_total": g.size(),
        "freq_90d": validos[validos["data"] > corte - pd.Timedelta(days=90)].groupby("cliente_id").size(),
        "freq_prev_90d": validos[(validos["data"] <= corte - pd.Timedelta(days=90))
                                 & (validos["data"] > corte - pd.Timedelta(days=180))].groupby("cliente_id").size(),
        "valor_total": g["valor_total"].sum(),
        "ticket_medio": g["valor_total"].mean(),
        "itens_por_pedido": g["qtd_itens"].mean(),
        "desconto_medio_pct": g["desconto_medio"].mean(),
        "frete_medio": g["frete"].mean(),
        "pct_fim_de_semana": g["eh_fim_de_semana"].mean(),
    })
    f[["freq_90d", "freq_prev_90d"]] = f[["freq_90d", "freq_prev_90d"]].fillna(0).astype(int)
    f["tendencia_freq"] = (f["freq_90d"] - f["freq_prev_90d"]) / (f["freq_90d"] + f["freq_prev_90d"] + 1)

    # intervalo medio entre compras sucessivas (nulo para quem comprou so uma vez)
    intervalos = validos.groupby("cliente_id")["data"].diff().dt.days
    f["intervalo_medio_dias"] = intervalos.groupby(validos["cliente_id"]).mean()
    f["razao_recencia_intervalo"] = f["recencia_dias"] / f["intervalo_medio_dias"].clip(lower=1)

    # qualidade da experiencia: considera TODOS os pedidos ate T (inclusive cancelados/devolvidos)
    gh = hist.groupby("cliente_id")
    f["pct_cancelados"] = gh["status"].apply(lambda s: (s == "cancelado").mean())
    f["pct_devolvidos"] = gh["status"].apply(lambda s: (s == "devolvido").mean())
    f["nota_media"] = gh["nota_avaliacao"].mean()

    cat = categorias[categorias["pedido_id"].isin(validos["pedido_id"])].merge(
        validos[["pedido_id", "cliente_id"]], on="pedido_id")
    f["categorias_distintas"] = cat.groupby("cliente_id")["categoria"].nunique()

    f = f[f["recencia_dias"] <= config.JANELA_ATIVOS_DIAS]          # restringe a base ativa

    f = f.reset_index().merge(clientes, on="cliente_id", how="left")
    f["tempo_de_casa_dias"] = (corte - f["data_cadastro"]).dt.days
    idade = (corte - f["data_nascimento"]).dt.days / 365.25
    f["idade"] = idade.round()
    f["data_corte"] = corte.date()
    f[ALVO] = (~f["cliente_id"].isin(futuro["cliente_id"].unique())).astype(int)

    colunas = (["cliente_id", "data_corte"] + FEATURES_NUMERICAS + FEATURES_CATEGORICAS
               + FEATURES_BOOLEANAS + [ALVO])
    return f[colunas]


def gerar_e_gravar_features(engine: Engine, cortes: list[date]) -> pd.DataFrame:
    pedidos, categorias, clientes = carregar_base_dw(engine)
    partes = [construir_features(pedidos, categorias, clientes, c) for c in cortes]
    base = pd.concat(partes, ignore_index=True)

    with engine.begin() as con:
        con.execute(text("DELETE FROM ml.features_churn"))
    base_db = base.copy()
    base_db["idade"] = base_db["idade"].astype("Int64")
    base_db.to_sql("features_churn", engine, schema="ml", if_exists="append", index=False,
                   method="multi", chunksize=2000)
    for c, parte in base.groupby("data_corte"):
        log.info("features %s: %d clientes, churn=%.1f%%", c, len(parte), 100 * parte[ALVO].mean())
    return base


def carregar_features(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM ml.features_churn ORDER BY data_corte, cliente_id", engine, parse_dates=["data_corte"])
    for col in df.columns.difference(["data_corte", "segmento", "canal_aquisicao", "regiao", "aceita_marketing"]):
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["data_corte"] = df["data_corte"].dt.date
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gerar_e_gravar_features(config.get_engine(), config.SNAPSHOTS_TREINO + [config.SNAPSHOT_TESTE])
