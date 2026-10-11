"""Etapa T do ETL: limpeza, padronizacao, validacao, anonimizacao e enriquecimento.

Linhas que violam regras de qualidade nao sao descartadas em silencio: vao para a quarentena
(``etl.rejeitados``) com o motivo.
"""
from __future__ import annotations

import hashlib
import unicodedata

import numpy as np
import pandas as pd

from . import config

STATUS_VALIDOS = {"entregue", "cancelado", "devolvido"}
SEGMENTOS_VALIDOS = {"basico", "standard", "premium"}
CANAIS_VALIDOS = {"organico", "ads", "social", "email", "indicacao"}
DATA_MIN = pd.Timestamp(config.DATA_INICIO)
DATA_MAX = pd.Timestamp(config.DATA_FIM)


def normalizar_texto(s: pd.Series) -> pd.Series:
    """minusculas, sem acento, sem espacos nas pontas (chave de comparacao)."""
    def _n(v):
        if not isinstance(v, str):
            return v
        v = unicodedata.normalize("NFD", v.strip().lower())
        return "".join(ch for ch in v if unicodedata.category(ch) != "Mn")
    return s.map(_n).astype(object)


def _rejeitar(df: pd.DataFrame, mascara: pd.Series, motivo: str, fonte: str) -> pd.DataFrame:
    rej = df.loc[mascara].copy()
    rej["_fonte"], rej["_motivo"] = fonte, motivo
    return rej


# --------------------------------------------------------------------------- localidades
def transformar_ibge(ibge: pd.DataFrame) -> pd.DataFrame:
    out = ibge.dropna(subset=["codigo_ibge", "municipio", "uf"]).drop_duplicates("codigo_ibge").copy()
    out["codigo_ibge"] = out["codigo_ibge"].astype(int)
    out["_chave_municipio"] = normalizar_texto(out["municipio"])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- clientes
def _parse_data(s: pd.Series) -> pd.Series:
    """Aceita ISO (aaaa-mm-dd) e dd/mm/aaaa; o que nao converter vira NaT."""
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    br = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(br)


def _parse_bool(s: pd.Series) -> pd.Series:
    mapa = {"true": True, "1": True, "sim": True, "s": True,
            "false": False, "0": False, "nao": False, "n": False}
    return normalizar_texto(s.astype(str)).map(mapa)


def _hash_email(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


def transformar_clientes(crm: pd.DataFrame, localidades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = crm.copy()
    rejeitados = []

    # 1) deduplicacao por chave: fica a versao mais recente do registro
    df["_atualizado"] = pd.to_datetime(df["atualizado_em"], errors="coerce")
    df = df.sort_values(["id_cliente", "_atualizado"]).drop_duplicates("id_cliente", keep="last")

    # 2) obrigatorios: sem e-mail nao ha como identificar/anonimizar com seguranca
    sem_email = df["email"].isna() | (df["email"].astype(str).str.strip() == "")
    rejeitados.append(_rejeitar(df, sem_email, "email_ausente", "crm"))
    df = df[~sem_email].copy()

    # 3) tipagem e padronizacao
    df["data_cadastro"] = _parse_data(df["data_cadastro"])
    sem_cadastro = df["data_cadastro"].isna()
    rejeitados.append(_rejeitar(df, sem_cadastro, "data_cadastro_invalida", "crm"))
    df = df[~sem_cadastro].copy()

    df["segmento"] = normalizar_texto(df["segmento"])
    df["canal_aquisicao"] = normalizar_texto(df["canal_aquisicao"])
    invalido = ~df["segmento"].isin(SEGMENTOS_VALIDOS) | ~df["canal_aquisicao"].isin(CANAIS_VALIDOS)
    rejeitados.append(_rejeitar(df, invalido, "dominio_invalido", "crm"))
    df = df[~invalido].copy()

    df["genero"] = normalizar_texto(df["genero"]).map(
        {"f": "feminino", "feminino": "feminino", "m": "masculino", "masculino": "masculino", "outro": "outro"})
    df["aceita_marketing"] = _parse_bool(df["aceita_marketing"]).fillna(False).astype(bool)

    nasc = _parse_data(df["data_nascimento"])
    idade = (df["data_cadastro"] - nasc).dt.days / 365.25
    df["data_nascimento"] = nasc.where((idade >= 16) & (idade <= 100))   # implausivel => nulo (nao rejeita)

    # 4) anonimizacao (LGPD): so o hash do e-mail segue para o DW
    df["email_hash"] = df["email"].map(_hash_email)

    # 5) enriquecimento com IBGE: casa por (municipio normalizado, UF)
    df["_chave_municipio"] = normalizar_texto(df["cidade"])
    df["uf"] = df["uf"].astype(str).str.strip().str.upper()
    chaves = localidades[["_chave_municipio", "uf", "codigo_ibge"]]
    df = df.merge(chaves, on=["_chave_municipio", "uf"], how="left").drop_duplicates("id_cliente")

    final = df.rename(columns={"id_cliente": "cliente_id"})[
        ["cliente_id", "email_hash", "genero", "data_nascimento", "data_cadastro", "segmento",
         "canal_aquisicao", "aceita_marketing", "codigo_ibge"]].reset_index(drop=True)
    final["cliente_id"] = final["cliente_id"].astype(int)
    return final, pd.concat(rejeitados, ignore_index=True)


# --------------------------------------------------------------------------- pedidos
def transformar_pedidos(raw: pd.DataFrame, clientes_validos: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    rejeitados = []

    # 1) duplicatas exatas (mesma linha exportada duas vezes)
    df = df.drop_duplicates()

    # 2) tipagem
    df["data_pedido"] = pd.to_datetime(df["data_pedido"], format="%Y-%m-%d", errors="coerce")
    for col in ["pedido_id", "item_seq", "cliente_id", "produto_id", "quantidade",
                "preco_unitario", "desconto_pct", "frete", "nota_avaliacao"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["status", "forma_pagamento", "canal_pedido", "categoria"]:
        df[col] = normalizar_texto(df[col])

    # 3) regras de qualidade -> quarentena
    regras = [
        ("data_invalida_ou_fora_da_janela",
         lambda d: d["data_pedido"].isna() | (d["data_pedido"] < DATA_MIN) | (d["data_pedido"] > DATA_MAX)),
        ("quantidade_invalida", lambda d: d["quantidade"].isna() | (d["quantidade"] <= 0)),
        ("desconto_invalido", lambda d: d["desconto_pct"].isna() | ~d["desconto_pct"].between(0, 90)),
        ("status_invalido", lambda d: ~d["status"].isin(STATUS_VALIDOS)),
        ("cliente_inexistente_no_crm", lambda d: ~d["cliente_id"].isin(clientes_validos)),
    ]
    pedidos_afetados: set = set()
    for motivo, regra in regras:
        mascara = regra(df)
        rejeitados.append(_rejeitar(df, mascara, motivo, "pedidos"))
        pedidos_afetados |= set(df.loc[mascara, "pedido_id"].dropna())
        df = df[~mascara].copy()

    # 4) preco ausente: recupera pelo preco de catalogo (mediana do proprio produto)
    mediana = df.groupby("produto_id")["preco_unitario"].transform("median")
    df["preco_unitario"] = df["preco_unitario"].fillna(mediana)
    sem_preco = df["preco_unitario"].isna() | (df["preco_unitario"] <= 0)
    rejeitados.append(_rejeitar(df, sem_preco, "preco_ausente", "pedidos"))
    pedidos_afetados |= set(df.loc[sem_preco, "pedido_id"].dropna())
    df = df[~sem_preco].copy()

    # 5) consistencia: se um item do pedido foi rejeitado, o pedido inteiro vai para a quarentena
    #    (manter so parte dos itens subestimaria o valor do pedido e distorceria as features)
    incompleto = df["pedido_id"].isin(pedidos_afetados)
    rejeitados.append(_rejeitar(df, incompleto, "pedido_com_item_invalido", "pedidos"))
    df = df[~incompleto].copy()

    # 6) metricas derivadas e tipos finais
    df["valor_item"] = (df["quantidade"] * df["preco_unitario"] * (1 - df["desconto_pct"] / 100)).round(2)
    for col in ["pedido_id", "item_seq", "cliente_id", "produto_id", "quantidade"]:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True), pd.concat(rejeitados, ignore_index=True)


def montar_tabelas_fato_dim(itens: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A partir dos itens validos, deriva dim_produto, fato_pedido e fato_item_pedido."""
    produto = (itens.groupby("produto_id")
               .agg(nome=("produto_nome", "first"), categoria=("categoria", "first"),
                    preco_lista=("preco_unitario", "median")).reset_index())
    produto["preco_lista"] = produto["preco_lista"].round(2)

    pedido = (itens.groupby("pedido_id")
              .agg(cliente_id=("cliente_id", "first"), data_id=("data_pedido", "first"),
                   status=("status", "first"), canal_pedido=("canal_pedido", "first"),
                   forma_pagamento=("forma_pagamento", "first"), frete=("frete", "first"),
                   nota_avaliacao=("nota_avaliacao", "first"), qtd_itens=("quantidade", "sum"),
                   valor_total=("valor_item", "sum")).reset_index())
    pedido["frete"] = pedido["frete"].fillna(0.0)
    pedido["valor_total"] = (pedido["valor_total"] + pedido["frete"]).round(2)
    pedido["data_id"] = pedido["data_id"].dt.date
    pedido["nota_avaliacao"] = pedido["nota_avaliacao"].astype("Int64")

    item = itens[["pedido_id", "item_seq", "produto_id", "quantidade",
                  "preco_unitario", "desconto_pct", "valor_item"]]
    return {"dim_produto": produto, "fato_pedido": pedido, "fato_item_pedido": item}


def montar_dim_data() -> pd.DataFrame:
    datas = pd.date_range(config.DATA_INICIO, config.DATA_FIM, freq="D")
    nomes = np.array(["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"])
    return pd.DataFrame({
        "data_id": datas.date, "ano": datas.year, "trimestre": datas.quarter, "mes": datas.month,
        "dia": datas.day, "dia_semana": datas.dayofweek, "nome_dia_semana": nomes[datas.dayofweek],
        "eh_fim_de_semana": datas.dayofweek >= 5,
    })
