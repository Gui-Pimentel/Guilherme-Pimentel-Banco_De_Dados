"""Executa o pipeline completo: fontes -> ETL -> PostgreSQL -> features -> modelo -> metricas.

Uso (a partir da raiz do repositorio):
    python -m src.run_pipeline                 # tudo, do zero
    python -m src.run_pipeline --skip-generate # reaproveita data/raw ja existente
    python -m src.run_pipeline --baixar-ibge   # forca novo download da API do IBGE
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")                          # graficos salvos em arquivo, sem abrir janelas

from . import config, etl, features, generate_data, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-generate", action="store_true", help="nao regera pedidos.csv / clientes_crm.json")
    ap.add_argument("--baixar-ibge", action="store_true", help="forca novo download da API do IBGE")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pipeline")

    if not args.skip_generate or not config.PEDIDOS_CSV.exists():
        log.info("[1/4] Gerando fontes brutas (CSV + JSON)")
        generate_data.gerar()
    else:
        log.info("[1/4] Reaproveitando fontes em data/raw")

    log.info("[2/4] ETL: extracao, transformacao e carga no PostgreSQL")
    etl.executar_etl(baixar_ibge=args.baixar_ibge)

    engine = config.get_engine()
    log.info("[3/4] Engenharia de features (ml.features_churn)")
    base = features.gerar_e_gravar_features(engine, config.SNAPSHOTS_TREINO + [config.SNAPSHOT_TESTE])

    log.info("[4/4] Treino, selecao e avaliacao do modelo")
    resultado = train.treinar_e_avaliar(base, engine)
    teste = resultado["melhor_modelo_teste_limiar_otimizado"]
    log.info("Modelo: %s | teste out-of-time: F1=%.3f  ROC-AUC=%.3f  Accuracy=%.3f",
             resultado["modelo_selecionado"], teste["f1"], teste["roc_auc"], teste["accuracy"])


if __name__ == "__main__":
    main()
