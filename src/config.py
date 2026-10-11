"""Configuracao central: caminhos, conexao com o PostgreSQL e parametros do experimento."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figuras"
SQL_DIR = ROOT / "sql"

PEDIDOS_CSV = RAW_DIR / "pedidos.csv"
CLIENTES_JSON = RAW_DIR / "clientes_crm.json"
IBGE_JSON = RAW_DIR / "ibge_municipios.json"

IBGE_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

# --- parametros do experimento --------------------------------------------
SEED = 42
DATA_INICIO = date(2024, 1, 1)   # inicio da janela de pedidos exportada pelo sistema
DATA_FIM = date(2025, 12, 31)    # fim da janela
HORIZONTE_DIAS = 90              # churn = nenhuma compra valida nos proximos 90 dias
JANELA_ATIVOS_DIAS = 180         # base ativa = comprou nos ultimos 180 dias antes da data de corte
SNAPSHOTS_TREINO = [date(2025, 3, 31), date(2025, 6, 30)]
SNAPSHOT_TESTE = date(2025, 9, 30)

load_dotenv(ROOT / ".env")


def db_url(database: str | None = None) -> URL:
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=database or os.getenv("POSTGRES_DB", "projeto_integrador"),
    )


def get_engine(database: str | None = None, **kwargs) -> Engine:
    return create_engine(db_url(database), future=True, **kwargs)
