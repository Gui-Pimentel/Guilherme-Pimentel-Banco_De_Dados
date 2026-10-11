"""Etapa L do ETL: cria o banco/esquema e carrega as tabelas do DW no PostgreSQL."""
from __future__ import annotations

import json
import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import config

log = logging.getLogger(__name__)

COLUNAS_REJEITADOS_INTERNAS = {"_fonte", "_motivo", "_atualizado", "_chave_municipio"}


def garantir_banco() -> None:
    """Cria o banco do projeto se ele ainda nao existir (conecta no banco administrativo 'postgres')."""
    nome = config.db_url().database
    admin = config.get_engine("postgres", isolation_level="AUTOCOMMIT")
    with admin.connect() as con:
        existe = con.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": nome}).scalar()
        if not existe:
            con.execute(text(f'CREATE DATABASE "{nome}"'))
            log.info("Banco '%s' criado", nome)
    admin.dispose()


def criar_esquema(engine: Engine) -> None:
    """Executa sql/01_schema.sql (recria os schemas dw, ml e etl do zero)."""
    ddl = (config.SQL_DIR / "01_schema.sql").read_text(encoding="utf-8")
    with engine.begin() as con:
        con.exec_driver_sql(ddl)
    log.info("Esquema criado a partir de sql/01_schema.sql")


def iniciar_execucao(engine: Engine) -> int:
    with engine.begin() as con:
        return con.execute(text("INSERT INTO etl.execucoes DEFAULT VALUES RETURNING execucao_id")).scalar()


def finalizar_execucao(engine: Engine, execucao_id: int, resumo: dict, status: str = "sucesso") -> None:
    with engine.begin() as con:
        con.execute(
            text("UPDATE etl.execucoes SET finalizado_em = now(), status = :s, resumo = CAST(:r AS JSONB) "
                 "WHERE execucao_id = :id"),
            {"s": status, "r": json.dumps(resumo, default=str), "id": execucao_id},
        )


def carregar(df: pd.DataFrame, tabela: str, engine: Engine, schema: str = "dw") -> int:
    df.to_sql(tabela, engine, schema=schema, if_exists="append", index=False, method="multi", chunksize=5000)
    log.info("%s.%s: %d linhas carregadas", schema, tabela, len(df))
    return len(df)


def carregar_rejeitados(rejeitados: pd.DataFrame, execucao_id: int, engine: Engine) -> int:
    if rejeitados.empty:
        return 0
    registros = rejeitados.drop(columns=[c for c in COLUNAS_REJEITADOS_INTERNAS if c in rejeitados.columns])
    out = pd.DataFrame({
        "execucao_id": execucao_id,
        "fonte": rejeitados["_fonte"].to_numpy(),
        "motivo": rejeitados["_motivo"].to_numpy(),
        "registro": [json.dumps(r, default=str, ensure_ascii=False)
                     for r in registros.astype(object).where(registros.notna(), None).to_dict("records")],
    })
    with engine.begin() as con:
        con.execute(
            text("INSERT INTO etl.rejeitados (execucao_id, fonte, motivo, registro) "
                 "VALUES (:execucao_id, :fonte, :motivo, CAST(:registro AS JSONB))"),
            out.to_dict("records"),
        )
    return len(out)
