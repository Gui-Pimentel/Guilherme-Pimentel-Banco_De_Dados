"""Treino, comparacao e avaliacao de modelos de classificacao de churn.

Protocolo de validacao
----------------------
* **Treino**: snapshots de 2025-03-31 e 2025-06-30.
* **Teste (out-of-time)**: snapshot de 2025-09-30 - o futuro em relacao ao treino. O alvo do
  ultimo snapshot de treino (jul-set/2025) ja era conhecido em 30/09, entao nao ha vazamento.
* **Validacao cruzada**: ``GroupKFold`` por cliente, para o mesmo cliente nunca estar ao mesmo tempo
  no treino e na validacao (ele aparece em mais de um snapshot).
* **Selecao**: melhor F1 medio na validacao cruzada (o conjunto de teste so e usado no final).
"""
from __future__ import annotations

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from . import config
from .features import ALVO, FEATURES_BOOLEANAS, FEATURES_CATEGORICAS, FEATURES_NUMERICAS

log = logging.getLogger(__name__)

COLUNAS_X = FEATURES_NUMERICAS + FEATURES_CATEGORICAS + FEATURES_BOOLEANAS
SCORING_CV = {"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}


def dividir_treino_teste(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separacao temporal: treino = snapshots antigos; teste = snapshot mais recente."""
    # ordem canonica: o resultado nao pode depender da ordem em que o banco devolve as linhas
    ordenado = features.sort_values(["data_corte", "cliente_id"], kind="stable")
    teste = ordenado[ordenado["data_corte"] == config.SNAPSHOT_TESTE].reset_index(drop=True)
    treino = ordenado[ordenado["data_corte"].isin(config.SNAPSHOTS_TREINO)].reset_index(drop=True)
    return treino, teste


def montar_preprocessador() -> ColumnTransformer:
    numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),   # indicador de "faltava" vira feature
        ("scaler", StandardScaler()),
    ])
    categorico = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numerico, FEATURES_NUMERICAS),
        ("cat", categorico, FEATURES_CATEGORICAS),
        ("bool", "passthrough", FEATURES_BOOLEANAS),
    ])


def modelos_candidatos() -> dict[str, Pipeline]:
    estimadores = {
        "baseline_dummy": DummyClassifier(strategy="most_frequent"),
        "regressao_logistica": LogisticRegression(max_iter=2000, random_state=config.SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1,
                                                random_state=config.SEED),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                                           random_state=config.SEED),
    }
    return {nome: Pipeline([("prep", montar_preprocessador()), ("modelo", est)]) for nome, est in estimadores.items()}


def validacao_cruzada(modelos: dict[str, Pipeline], treino: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    cv = GroupKFold(n_splits=n_splits)
    linhas = []
    for nome, pipe in modelos.items():
        res = cross_validate(pipe, treino[COLUNAS_X], treino[ALVO], groups=treino["cliente_id"], cv=cv,
                             scoring=SCORING_CV, n_jobs=1)
        linhas.append({"modelo": nome,
                       **{f"{m}_media": res[f"test_{m}"].mean() for m in SCORING_CV},
                       **{f"{m}_desvio": res[f"test_{m}"].std() for m in SCORING_CV}})
        log.info("CV %-20s F1=%.3f  AUC=%.3f", nome, linhas[-1]["f1_media"], linhas[-1]["roc_auc_media"])
    return pd.DataFrame(linhas).sort_values("f1_media", ascending=False).reset_index(drop=True)


def escolher_limiar(pipe: Pipeline, treino: pd.DataFrame, n_splits: int = 5) -> tuple[float, pd.DataFrame]:
    """Limiar que maximiza o F1 usando apenas predicoes out-of-fold do TREINO (o teste nao e tocado)."""
    prob = cross_val_predict(pipe, treino[COLUNAS_X], treino[ALVO], groups=treino["cliente_id"],
                             cv=GroupKFold(n_splits=n_splits), method="predict_proba")[:, 1]
    grade = np.round(np.arange(0.10, 0.91, 0.02), 2)
    curva = pd.DataFrame({"limiar": grade,
                          "f1": [f1_score(treino[ALVO], prob >= t) for t in grade],
                          "precision": [precision_score(treino[ALVO], prob >= t, zero_division=0) for t in grade],
                          "recall": [recall_score(treino[ALVO], prob >= t) for t in grade]})
    return float(curva.loc[curva["f1"].idxmax(), "limiar"]), curva


def metricas(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def avaliar_no_teste(modelos: dict[str, Pipeline], treino: pd.DataFrame, teste: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Treina cada modelo no treino completo e mede no teste out-of-time."""
    linhas, ajustados = [], {}
    for nome, pipe in modelos.items():
        pipe.fit(treino[COLUNAS_X], treino[ALVO])
        prob = pipe.predict_proba(teste[COLUNAS_X])[:, 1]
        pred = (prob >= 0.5).astype(int)
        linhas.append({"modelo": nome, **metricas(teste[ALVO], pred, prob)})
        ajustados[nome] = pipe
    return pd.DataFrame(linhas).set_index("modelo"), ajustados


# --------------------------------------------------------------------------- graficos
def _salvar(fig, nome: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = config.FIGURES_DIR / nome
    fig.savefig(caminho, dpi=130, bbox_inches="tight")
    return str(caminho)


def grafico_matriz_confusao(y_true, y_pred, titulo: str):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["ficou", "churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(titulo)
    return fig


def grafico_roc(modelos_ajustados: dict[str, Pipeline], teste: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for nome, pipe in modelos_ajustados.items():
        RocCurveDisplay.from_estimator(pipe, teste[COLUNAS_X], teste[ALVO], ax=ax, name=nome)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("Curva ROC no teste (out-of-time)")
    return fig


def importancia_permutacao(pipe: Pipeline, teste: pd.DataFrame, n_repeats: int = 10) -> pd.DataFrame:
    """Queda de ROC-AUC ao embaralhar cada feature original (agnostico ao tipo de modelo)."""
    r = permutation_importance(pipe, teste[COLUNAS_X], teste[ALVO], scoring="roc_auc",
                               n_repeats=n_repeats, random_state=config.SEED, n_jobs=1)
    return (pd.DataFrame({"feature": COLUNAS_X, "queda_auc": r.importances_mean, "desvio": r.importances_std})
            .sort_values("queda_auc", ascending=False).reset_index(drop=True))


def grafico_importancia(imp: pd.DataFrame, top: int = 12):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    d = imp.head(top).iloc[::-1]
    ax.barh(d["feature"], d["queda_auc"], xerr=d["desvio"], color="#2a6f97")
    ax.set_xlabel("Queda do ROC-AUC ao embaralhar a feature")
    ax.set_title("Importancia por permutacao (teste)")
    return fig


# --------------------------------------------------------------------------- fluxo completo
def treinar_e_avaliar(features: pd.DataFrame, engine=None) -> dict:
    treino, teste = dividir_treino_teste(features)
    log.info("treino=%d linhas (churn %.1f%%) | teste=%d linhas (churn %.1f%%)",
             len(treino), 100 * treino[ALVO].mean(), len(teste), 100 * teste[ALVO].mean())

    cv = validacao_cruzada(modelos_candidatos(), treino)
    melhor_nome = cv.loc[~cv["modelo"].eq("baseline_dummy")].iloc[0]["modelo"]

    tabela_teste, ajustados = avaliar_no_teste(modelos_candidatos(), treino, teste)
    melhor = ajustados[melhor_nome]
    prob = melhor.predict_proba(teste[COLUNAS_X])[:, 1]
    limiar, curva_limiar = escolher_limiar(modelos_candidatos()[melhor_nome], treino)
    pred = (prob >= limiar).astype(int)
    metricas_limiar_05 = metricas(teste[ALVO], (prob >= 0.5).astype(int), prob)
    metricas_limiar_otimo = metricas(teste[ALVO], pred, prob)
    log.info("limiar escolhido no treino (OOF) = %.2f | teste F1: %.3f (limiar 0.5) -> %.3f", limiar,
             metricas_limiar_05["f1"], metricas_limiar_otimo["f1"])
    imp = importancia_permutacao(melhor, teste)

    _salvar(grafico_matriz_confusao(teste[ALVO], pred, f"Matriz de confusao - {melhor_nome} (limiar {limiar:.2f})"), "matriz_confusao.png")
    _salvar(grafico_roc(ajustados, teste), "curva_roc.png")
    _salvar(grafico_importancia(imp), "importancia_features.png")
    plt.close("all")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": melhor, "colunas": COLUNAS_X, "modelo": melhor_nome, "limiar": limiar,
                 "snapshots_treino": [str(d) for d in config.SNAPSHOTS_TREINO]},
                config.MODELS_DIR / "modelo_churn.joblib")

    resultado = {
        "modelo_selecionado": melhor_nome,
        "criterio_selecao": "maior F1 medio na validacao cruzada (GroupKFold por cliente, k=5), excluindo o baseline",
        "treino": {"linhas": len(treino), "clientes": int(treino["cliente_id"].nunique()),
                   "snapshots": [str(d) for d in config.SNAPSHOTS_TREINO], "taxa_churn": float(treino[ALVO].mean())},
        "teste": {"linhas": len(teste), "snapshot": str(config.SNAPSHOT_TESTE), "taxa_churn": float(teste[ALVO].mean())},
        "validacao_cruzada": cv.round(4).to_dict("records"),
        "teste_por_modelo": tabela_teste.round(4).reset_index().to_dict("records"),
        "limiar_escolhido_no_treino": limiar,
        "melhor_modelo_teste_limiar_0.5": {k: round(v, 4) for k, v in metricas_limiar_05.items()},
        "melhor_modelo_teste_limiar_otimizado": {k: round(v, 4) for k, v in metricas_limiar_otimo.items()},
        "matriz_confusao_teste": confusion_matrix(teste[ALVO], pred).tolist(),
        "importancia_permutacao_top10": imp.head(10).round(4).to_dict("records"),
    }
    (config.DOCS_DIR / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    if engine is not None:
        _gravar_no_banco(engine, cv, tabela_teste, melhor_nome, teste, prob, pred)
    return resultado


def _gravar_no_banco(engine, cv, tabela_teste, melhor_nome, teste, prob, pred) -> None:
    from sqlalchemy import text
    linhas = []
    for r in cv.itertuples():
        for m in SCORING_CV:
            linhas.append({"modelo": r.modelo, "conjunto": "cv", "metrica": m, "valor": getattr(r, f"{m}_media"),
                           "selecionado": r.modelo == melhor_nome})
    for nome, r in tabela_teste.iterrows():
        for m, v in r.items():
            linhas.append({"modelo": nome, "conjunto": "teste", "metrica": m, "valor": v,
                           "selecionado": nome == melhor_nome})
    pred_df = pd.DataFrame({"cliente_id": teste["cliente_id"], "data_corte": teste["data_corte"],
                            "prob_churn": prob.round(5), "churn_previsto": pred, "churn_real": teste[ALVO]})
    with engine.begin() as con:
        con.execute(text("DELETE FROM ml.metricas_modelo"))
        con.execute(text("DELETE FROM ml.predicoes_churn"))
    pd.DataFrame(linhas).to_sql("metricas_modelo", engine, schema="ml", if_exists="append", index=False)
    pred_df.to_sql("predicoes_churn", engine, schema="ml", if_exists="append", index=False)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")                      # execucao em linha de comando: sem janelas
    from .features import carregar_features
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eng = config.get_engine()
    res = treinar_e_avaliar(carregar_features(eng), eng)
    print(json.dumps(res["teste_por_modelo"], indent=1))
