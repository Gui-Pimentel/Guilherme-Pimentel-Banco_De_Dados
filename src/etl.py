"""Orquestra o ETL completo: 3 fontes -> transformacao -> DW em PostgreSQL."""
from __future__ import annotations

import logging

import pandas as pd

from . import extract, load, transform
from . import config

log = logging.getLogger(__name__)


def executar_etl(baixar_ibge: bool = False) -> dict:
    # ---------------- Extract
    pedidos_raw = extract.extrair_pedidos()
    crm_raw = extract.extrair_clientes()
    ibge_raw = extract.extrair_ibge(forcar_download=baixar_ibge)

    # ---------------- Transform
    localidades = transform.transformar_ibge(ibge_raw)
    clientes, rej_clientes = transform.transformar_clientes(crm_raw, localidades)
    itens, rej_pedidos = transform.transformar_pedidos(pedidos_raw, set(clientes["cliente_id"]))
    tabelas = transform.montar_tabelas_fato_dim(itens)
    dim_data = transform.montar_dim_data()

    # ---------------- Load
    load.garantir_banco()
    engine = config.get_engine()
    load.criar_esquema(engine)
    execucao_id = load.iniciar_execucao(engine)
    resumo: dict = {"entrada": {"pedidos_csv": len(pedidos_raw), "clientes_crm": len(crm_raw),
                                "municipios_ibge": len(ibge_raw)}}
    try:
        cols_loc = ["codigo_ibge", "municipio", "uf", "estado", "regiao", "mesorregiao", "microrregiao"]
        load.carregar(localidades[cols_loc], "dim_localidade", engine)
        mapa_loc = pd.read_sql("SELECT localidade_id, codigo_ibge FROM dw.dim_localidade", engine)
        clientes_dw = clientes.merge(mapa_loc, on="codigo_ibge", how="left").drop(columns="codigo_ibge")
        clientes_dw["localidade_id"] = clientes_dw["localidade_id"].astype("Int64")

        carga = {
            "dim_localidade": len(localidades),
            "dim_data": load.carregar(dim_data, "dim_data", engine),
            "dim_cliente": load.carregar(clientes_dw, "dim_cliente", engine),
            "dim_produto": load.carregar(tabelas["dim_produto"], "dim_produto", engine),
            "fato_pedido": load.carregar(tabelas["fato_pedido"], "fato_pedido", engine),
            "fato_item_pedido": load.carregar(tabelas["fato_item_pedido"], "fato_item_pedido", engine),
        }
        rejeitados = pd.concat([rej_clientes, rej_pedidos], ignore_index=True)
        load.carregar_rejeitados(rejeitados, execucao_id, engine)

        resumo["carga"] = carga
        resumo["rejeitados_por_motivo"] = (
            rejeitados.groupby(["_fonte", "_motivo"]).size().rename("n").reset_index()
            .assign(chave=lambda d: d["_fonte"] + "/" + d["_motivo"]).set_index("chave")["n"].astype(int).to_dict())
        resumo["clientes_sem_localidade"] = int(clientes_dw["localidade_id"].isna().sum())
        load.finalizar_execucao(engine, execucao_id, resumo)
    except Exception:
        load.finalizar_execucao(engine, execucao_id, resumo, status="falha")
        raise
    log.info("ETL concluido: %s", resumo["carga"])
    return resumo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    executar_etl()
