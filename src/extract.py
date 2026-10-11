"""Etapa E do ETL: coleta das 3 fontes, sem aplicar nenhuma regra de negocio."""
from __future__ import annotations

import json
import logging

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)


def extrair_pedidos() -> pd.DataFrame:
    """Fonte 1 - CSV do sistema transacional. Tudo entra como texto; a tipagem e feita na transformacao."""
    df = pd.read_csv(config.PEDIDOS_CSV, dtype=str, keep_default_na=True)
    log.info("pedidos.csv: %d linhas", len(df))
    return df


def extrair_clientes() -> pd.DataFrame:
    """Fonte 2 - JSON exportado do CRM."""
    with open(config.CLIENTES_JSON, encoding="utf-8") as f:
        registros = json.load(f)
    df = pd.DataFrame(registros)
    log.info("clientes_crm.json: %d registros", len(df))
    return df


def extrair_ibge(forcar_download: bool = False) -> pd.DataFrame:
    """Fonte 3 - API de localidades do IBGE (5.570 municipios).

    O JSON baixado e guardado em ``data/raw`` (cache). Se a API estiver fora do ar, usa o cache,
    para o pipeline continuar reproduzivel offline.
    """
    dados = None
    if forcar_download or not config.IBGE_JSON.exists():
        try:
            resp = requests.get(config.IBGE_URL, timeout=60)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            dados = resp.json()
            config.IBGE_JSON.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
            log.info("IBGE: %d municipios baixados da API", len(dados))
        except requests.RequestException as exc:
            if not config.IBGE_JSON.exists():
                raise RuntimeError("API do IBGE indisponivel e sem cache local") from exc
            log.warning("API do IBGE indisponivel (%s); usando cache local", exc)
    if dados is None:
        dados = json.loads(config.IBGE_JSON.read_text(encoding="utf-8"))
        log.info("IBGE: %d municipios lidos do cache", len(dados))

    linhas = []
    for m in dados:
        micro = m.get("microrregiao") or {}
        meso = micro.get("mesorregiao") or {}
        uf = meso.get("UF") or m.get("regiao-imediata", {}).get("regiao-intermediaria", {}).get("UF") or {}
        linhas.append({
            "codigo_ibge": m["id"],
            "municipio": m["nome"],
            "uf": uf.get("sigla"),
            "estado": uf.get("nome"),
            "regiao": (uf.get("regiao") or {}).get("nome"),
            "mesorregiao": meso.get("nome"),
            "microrregiao": micro.get("nome"),
        })
    return pd.DataFrame(linhas)
