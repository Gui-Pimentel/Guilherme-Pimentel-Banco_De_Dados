"""Gera os notebooks do projeto a partir de celulas definidas em codigo.

    python build_notebooks.py              # apenas cria os .ipynb (sem outputs)
    python build_notebooks.py --execute    # cria e executa (outputs embutidos; requer PostgreSQL com o ETL rodado)

Notebooks gerados em notebooks/:
  01_analise_exploratoria.ipynb       EDA sobre o DW em PostgreSQL
  02_features_modelagem.ipynb         features, treino, avaliacao e interpretacao
  Projeto_Integrador_Final.ipynb      notebook final = introducao + 01 + 02 (arquivo anexado no Moodle)
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell as code, new_markdown_cell as md, new_notebook

NB_DIR = Path(__file__).parent / "notebooks"

SETUP = '''import sys, warnings
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config, features, train

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 150); pd.set_option("display.max_columns", 40)
FIG = config.FIGURES_DIR; FIG.mkdir(parents=True, exist_ok=True)
engine = config.get_engine()

def q(sql):
    """Executa SQL no PostgreSQL e devolve um DataFrame."""
    return pd.read_sql(sql, engine)

print("Conectado a:", config.db_url().render_as_string(hide_password=True))'''

# =============================================================================== INTRODUCAO
INTRO = [
    md("""# Projeto Integrador — Pipeline de Engenharia de Dados e Machine Learning
**Previsão de churn de clientes de e-commerce** · MBA em IA — Banco de Dados para Engenharia de Dados (BDED-2026.1)

## Problema de negócio
Uma loja virtual quer **identificar, antes que aconteça, quais clientes ativos deixarão de comprar nos próximos 90 dias**,
para direcionar ações de retenção (cupom, contato, benefício) a quem realmente está em risco.

## Pipeline end-to-end
```
 Fonte 1: pedidos.csv  (sistema transacional) ─┐
 Fonte 2: clientes_crm.json (CRM)              ├─►  ETL Python  ─►  PostgreSQL (schemas dw / etl / ml)
 Fonte 3: API IBGE (localidades)               ─┘   extract → transform → load      │
                                                                                       ▼
                                      Features (corte temporal) ─► Modelo de classificação ─► Métricas / predições
```

| Etapa do enunciado | Onde está |
|---|---|
| 1. Coleta de ≥ 2 fontes | `src/extract.py` (CSV + JSON + API REST do IBGE) |
| 2. PostgreSQL com modelagem adequada | `sql/01_schema.sql` (estrela em `dw`, controle em `etl`, resultados em `ml`) |
| 3. ETL em Python | `src/transform.py`, `src/load.py`, `src/etl.py` |
| 4. Preparação e features | `src/features.py` + seção *Features* deste notebook |
| 5. Modelo de classificação | `src/train.py` + seção *Modelagem* deste notebook |
| 6. Avaliação com métricas | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| 7. Documentação | `README.md`, este notebook, `docs/` |

> **Pré-requisito:** o pipeline já deve ter sido executado (`python -m src.run_pipeline`), pois este notebook
> lê do PostgreSQL. A célula abaixo permite reexecutá-lo daqui.""",),
    code('''RODAR_PIPELINE = False   # True => regenera dados, ETL, features e modelo antes de analisar (~1 min)
if RODAR_PIPELINE:
    import sys, subprocess
    from pathlib import Path
    root = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
    subprocess.run([sys.executable, "-m", "src.run_pipeline"], cwd=root, check=True)'''),
]

# =============================================================================== EDA
EDA = [
    md("""# Parte 1 — Análise exploratória (EDA)
Toda a análise parte do **Data Warehouse no PostgreSQL**, não dos arquivos brutos: o que se analisa é o que o ETL entregou."""),
    code(SETUP),
    md("## 1.1 Visão geral do Data Warehouse"),
    code('''q("""
SELECT 'dw.dim_localidade (IBGE)' AS tabela, COUNT(*) AS linhas FROM dw.dim_localidade UNION ALL
SELECT 'dw.dim_data',              COUNT(*) FROM dw.dim_data              UNION ALL
SELECT 'dw.dim_cliente (CRM)',     COUNT(*) FROM dw.dim_cliente           UNION ALL
SELECT 'dw.dim_produto',           COUNT(*) FROM dw.dim_produto           UNION ALL
SELECT 'dw.fato_pedido',           COUNT(*) FROM dw.fato_pedido           UNION ALL
SELECT 'dw.fato_item_pedido',      COUNT(*) FROM dw.fato_item_pedido      UNION ALL
SELECT 'etl.rejeitados (quarentena)', COUNT(*) FROM etl.rejeitados        UNION ALL
SELECT 'ml.features_churn',        COUNT(*) FROM ml.features_churn
""")'''),
    md("""## 1.2 Qualidade dos dados: o que o ETL rejeitou e por quê
As fontes foram geradas com problemas propositais. Nada é descartado em silêncio: cada linha rejeitada fica em `etl.rejeitados` com o motivo."""),
    code('''rej = q("SELECT fonte, motivo, COUNT(*) AS linhas FROM etl.rejeitados GROUP BY 1, 2 ORDER BY linhas DESC")
entrada = q("SELECT resumo->'entrada' AS entrada FROM etl.execucoes ORDER BY execucao_id DESC LIMIT 1").iloc[0, 0]
display(rej)
print("Entrada da última execução:", entrada)
fig, ax = plt.subplots(figsize=(8, 3.2))
sns.barplot(data=rej, y="motivo", x="linhas", hue="fonte", dodge=False, ax=ax)
ax.set_title("Linhas em quarentena por motivo"); plt.tight_layout(); plt.savefig(FIG / "eda_rejeitados.png", dpi=130); plt.show()'''),
    code('''sem_loc = q("""SELECT COUNT(*) FILTER (WHERE localidade_id IS NULL) AS sem_localidade, COUNT(*) AS total FROM dw.dim_cliente""")
display(sem_loc)
print("Clientes cujo município/UF não casou com o IBGE ficam com localidade nula (região = 'nao_informada'), em vez de serem descartados.")'''),
    md("## 1.3 Comportamento de compras"),
    code('''mensal = q("""
SELECT make_date(d.ano::int, d.mes::int, 1) AS mes, COUNT(*) AS pedidos, SUM(p.valor_total) AS receita
FROM dw.fato_pedido p JOIN dw.dim_data d ON d.data_id = p.data_id
WHERE p.status <> 'cancelado' GROUP BY 1 ORDER BY 1""")
mensal["receita"] = mensal["receita"].astype(float)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].plot(mensal["mes"], mensal["pedidos"], marker="o"); ax[0].set_title("Pedidos por mês (sem cancelados)")
ax[1].plot(mensal["mes"], mensal["receita"] / 1e3, marker="o", color="#2a6f97"); ax[1].set_title("Receita por mês (R$ mil)")
for a in ax: a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(FIG / "eda_serie_mensal.png", dpi=130); plt.show()'''),
    code('''fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
status = q("SELECT status, COUNT(*) AS n FROM dw.fato_pedido GROUP BY 1 ORDER BY n DESC")
sns.barplot(data=status, x="status", y="n", ax=ax[0], color="#2a6f97"); ax[0].set_title("Pedidos por status")
cat = q("""SELECT pr.categoria, SUM(i.valor_item)::float AS receita FROM dw.fato_item_pedido i
           JOIN dw.dim_produto pr USING (produto_id) GROUP BY 1 ORDER BY 2 DESC""")
sns.barplot(data=cat, y="categoria", x="receita", ax=ax[1], color="#2a6f97"); ax[1].set_title("Receita por categoria")
reg = q("""SELECT COALESCE(l.regiao, 'nao_informada') AS regiao, COUNT(DISTINCT c.cliente_id) AS clientes
           FROM dw.dim_cliente c LEFT JOIN dw.dim_localidade l USING (localidade_id) GROUP BY 1 ORDER BY 2 DESC""")
sns.barplot(data=reg, y="regiao", x="clientes", ax=ax[2], color="#2a6f97"); ax[2].set_title("Clientes por região (via IBGE)")
plt.tight_layout(); plt.savefig(FIG / "eda_status_categoria_regiao.png", dpi=130); plt.show()'''),
    code('''ticket = q("SELECT valor_total::float AS valor_total FROM dw.fato_pedido WHERE status <> 'cancelado'")["valor_total"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
sns.histplot(ticket, bins=60, ax=ax[0]); ax[0].set_title("Valor do pedido (escala linear): cauda longa")
sns.histplot(np.log10(ticket.clip(lower=1)), bins=60, ax=ax[1], color="#e07a5f"); ax[1].set_title("log10(valor do pedido)")
plt.tight_layout(); plt.show()
print(ticket.describe().round(2).to_string())'''),
    md("""## 1.4 Base de modelagem (`ml.features_churn`)
Cada linha é um **cliente ativo numa data de corte**; `churn = 1` se não comprou nos 90 dias seguintes. Detalhes na Parte 2."""),
    code('''df = features.carregar_features(engine)
print("shape:", df.shape)
display(df.groupby("data_corte")["churn"].agg(clientes="size", taxa_churn="mean").round(3))
display(df.describe().T.round(2))'''),
    code('''nulos = df.isna().mean().sort_values(ascending=False)
nulos = nulos[nulos > 0]
display(nulos.round(3).to_frame("proporção de nulos"))
print("Nulos têm explicação de negócio: sem intervalo (1 só compra), sem nota (cliente não avaliou), sem idade (data de nascimento ausente/implausível).")'''),
    md("### Churn por perfil do cliente"),
    code('''fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for a, col in zip(ax.ravel(), ["segmento", "canal_aquisicao", "regiao", "aceita_marketing"]):
    t = df.groupby(col)["churn"].mean().sort_values(ascending=False).mul(100)
    sns.barplot(x=t.index.astype(str), y=t.values, ax=a, color="#2a6f97")
    a.axhline(df["churn"].mean() * 100, color="red", ls="--", lw=1); a.set_title(f"% churn por {col} (linha = média)")
    a.set_ylabel("% churn"); a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(FIG / "eda_churn_por_perfil.png", dpi=130); plt.show()'''),
    md("### Comportamento de compra: quem sofre churn vs. quem fica"),
    code('''cols = ["recencia_dias", "freq_90d", "tendencia_freq", "razao_recencia_intervalo", "nota_media", "pct_devolvidos"]
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, c in zip(ax.ravel(), cols):
    sns.boxplot(data=df.assign(churn=df["churn"].map({0: "ficou", 1: "churn"})), x="churn", y=c, ax=a,
                order=["ficou", "churn"], palette=["#2a9d8f", "#e76f51"], showfliers=False)
    a.set_title(c)
plt.tight_layout(); plt.savefig(FIG / "eda_features_vs_churn.png", dpi=130); plt.show()
display(df.groupby("churn")[cols].mean().round(2))'''),
    code('''corr = df.select_dtypes("number").drop(columns=["cliente_id"]).corr()
ordem = corr["churn"].drop("churn").sort_values()
fig, ax = plt.subplots(figsize=(6, 6.5))
sns.barplot(x=ordem.values, y=ordem.index, color="#2a6f97", ax=ax)
ax.set_title("Correlação (Pearson) de cada feature com churn"); plt.tight_layout()
plt.savefig(FIG / "eda_correlacao_churn.png", dpi=130); plt.show()'''),
    md("""### Conclusões da EDA
* **Recência e ritmo recente de compra** são os sinais mais fortes: quem some há mais tempo, compra menos nos últimos 90 dias e tem frequência em queda tende ao churn.
* **Segmento** importa: o churn cai de `basico` para `standard` e `premium`.
* **Nota de avaliação** menor (3,5 vs 3,8) e um pouco mais de devoluções (7% vs 5%) acompanham o churn: sinais fracos, mas coerentes, de insatisfação.
* A taxa de churn é **estável entre as datas de corte** (~30%), então o problema é moderadamente desbalanceado e a validação temporal é justa.
* Há **cauda longa** no valor dos pedidos, e vários nulos com explicação de negócio (tratados por imputação + indicador de ausência na modelagem).""",),
]

# =============================================================================== MODELAGEM
MODELAGEM = [
    md("""# Parte 2 — Engenharia de features e modelagem

## 2.1 Definição do problema (e como evitamos vazamento de dados)
Para uma **data de corte T**:

| Elemento | Definição |
|---|---|
| **População** | clientes com ao menos 1 compra válida (não cancelada) nos **180 dias anteriores a T** (base ativa) |
| **Features** | calculadas usando **somente pedidos com data ≤ T** |
| **Alvo `churn`** | 1 se o cliente **não** fez nenhuma compra válida em **(T, T + 90 dias]** |

```
treino:  cortes 31/03/2025 e 30/06/2025   →  alvo observado em abr–set/2025
teste:   corte  30/09/2025 (out-of-time)  →  alvo observado em out–dez/2025
```
O teste é o **futuro** em relação ao treino. Restringir à base ativa evita um problema trivial (prever "churn" de quem sumiu há um ano).

> No projeto M5/P14 o alvo era derivado de `dias_sem_comprar` e essa mesma coluna entrava como feature — vazamento clássico. Aqui o alvo
> vem de um período **posterior** ao das features, e isso é verificado por teste automatizado e por uma checagem neste notebook."""),
    code(SETUP),
    md("""## 2.2 Dicionário de features
| Grupo | Feature | Descrição |
|---|---|---|
| RFM / ritmo | `recencia_dias` | dias desde a última compra válida até T |
| | `frequencia_total` | nº de compras válidas até T |
| | `freq_90d`, `freq_prev_90d` | compras nos últimos 90 dias e nos 90 anteriores |
| | `tendencia_freq` | (freq_90d − freq_prev_90d) / (freq_90d + freq_prev_90d + 1): ritmo subindo ou caindo |
| | `intervalo_medio_dias` | média de dias entre compras sucessivas (nulo se só 1 compra) |
| | `razao_recencia_intervalo` | recência ÷ intervalo médio: "quão atrasado" o cliente está em relação ao próprio ritmo |
| | `valor_total`, `ticket_medio`, `itens_por_pedido` | volume e tamanho de compra |
| | `categorias_distintas` | diversidade de categorias compradas |
| Experiência | `desconto_medio_pct`, `frete_medio` | sensibilidade a preço/frete |
| | `pct_cancelados`, `pct_devolvidos` | fricção (considera todos os pedidos até T) |
| | `nota_media` | satisfação declarada (nulo se nunca avaliou) |
| | `pct_fim_de_semana` | hábito de compra |
| Perfil | `tempo_de_casa_dias`, `idade` | antiguidade e idade em T |
| | `segmento`, `canal_aquisicao`, `regiao`, `aceita_marketing` | cadastro (CRM) e região (IBGE) |

**Tratamentos:** nulos numéricos → mediana + coluna indicadora de ausência; numéricas → `StandardScaler`; categóricas → `OneHotEncoder`.
Tudo dentro de um `Pipeline`, ajustado **só no treino** em cada dobra (sem vazamento pelo pré-processamento)."""),
    code('''df = features.carregar_features(engine)
treino, teste = train.dividir_treino_teste(df)
print(f"treino: {len(treino):>5} linhas, {treino['cliente_id'].nunique()} clientes, churn = {treino['churn'].mean():.1%}")
print(f"teste : {len(teste):>5} linhas, {teste['cliente_id'].nunique()} clientes, churn = {teste['churn'].mean():.1%}")
display(df.head(5).T.head(26))'''),
    md("""### Checagem de vazamento
Se as features usam só o passado, então **apagar todo o futuro do banco não pode alterá-las**. Recalculamos as features do corte de teste
com os pedidos truncados em T e comparamos com o que está gravado em `ml.features_churn`."""),
    code('''pedidos, categorias, clientes = features.carregar_base_dw(engine)
T = config.SNAPSHOT_TESTE
so_passado = pedidos[pedidos["data"] <= pd.Timestamp(T)]
a = features.construir_features(pedidos, categorias, clientes, T).set_index("cliente_id")
b = features.construir_features(so_passado, categorias, clientes, T).set_index("cliente_id")
pd.testing.assert_frame_equal(a[features.FEATURES_NUMERICAS], b[features.FEATURES_NUMERICAS])
print(f"OK: features de {T} idênticas com e sem o futuro ({len(a)} clientes). Nenhuma feature enxerga pedidos posteriores ao corte.")'''),
    md("## 2.3 Comparação de modelos — validação cruzada no treino\n`GroupKFold(k=5)` agrupando por cliente (o mesmo cliente aparece em mais de um corte e não pode estar nos dois lados da validação)."),
    code('''cv = train.validacao_cruzada(train.modelos_candidatos(), treino)
display(cv.round(4))'''),
    code('''fig, ax = plt.subplots(figsize=(8, 3.4))
d = cv[cv["modelo"] != "baseline_dummy"]
ax.barh(d["modelo"], d["f1_media"], xerr=d["f1_desvio"], color="#2a6f97"); ax.set_xlabel("F1 médio na validação cruzada (± desvio)")
ax.set_title("Os modelos reais ficam estatisticamente empatados"); plt.tight_layout(); plt.show()
melhor_nome = cv.loc[cv["modelo"] != "baseline_dummy"].iloc[0]["modelo"]
print("Selecionado pela regra pré-definida (maior F1 médio na CV, sem olhar o teste):", melhor_nome)'''),
    md("""## 2.4 Avaliação no teste *out-of-time*
Cada modelo é treinado no treino completo e medido no corte de 30/09/2025. O baseline (`most_frequent`) mostra o piso: acurácia alta por prever sempre "ficou", mas F1 = 0."""),
    code('''tabela, ajustados = train.avaliar_no_teste(train.modelos_candidatos(), treino, teste)
display(tabela.round(4))
fig = train.grafico_roc(ajustados, teste); plt.show()'''),
    md("""### Limiar de decisão
O limiar 0,5 não é sagrado. Escolhemos o que maximiza o F1 usando **apenas predições out-of-fold do treino** — o teste continua intocado."""),
    code('''melhor = ajustados[melhor_nome]
limiar, curva = train.escolher_limiar(train.modelos_candidatos()[melhor_nome], treino)
fig, ax = plt.subplots(figsize=(7, 3.6))
for c in ["f1", "precision", "recall"]:
    ax.plot(curva["limiar"], curva[c], label=c)
ax.axvline(limiar, color="k", ls="--", lw=1, label=f"escolhido = {limiar:.2f}"); ax.legend(); ax.set_xlabel("limiar")
ax.set_title("Trade-off precision × recall (predições out-of-fold do treino)"); plt.tight_layout(); plt.show()
print("Limiar escolhido:", limiar)'''),
    md("## 2.5 Resultado final do modelo selecionado"),
    code('''from sklearn.metrics import classification_report
prob = melhor.predict_proba(teste[train.COLUNAS_X])[:, 1]
pred = (prob >= limiar).astype(int)
y = teste["churn"]
res = pd.DataFrame({"limiar 0.50": train.metricas(y, (prob >= 0.5).astype(int), prob),
                    f"limiar {limiar:.2f} (escolhido)": train.metricas(y, pred, prob)}).round(4)
display(res)
print(classification_report(y, pred, target_names=["ficou", "churn"], digits=3))
fig = train.grafico_matriz_confusao(y, pred, f"Matriz de confusão — {melhor_nome} (limiar {limiar:.2f})"); plt.show()'''),
    md("### Interpretação: o que o modelo usa para decidir?\nImportância por **permutação** no teste (queda do ROC-AUC ao embaralhar cada feature) — vale para qualquer tipo de modelo."),
    code('''imp = train.importancia_permutacao(melhor, teste)
display(imp.head(10).round(4))
fig = train.grafico_importancia(imp); plt.show()'''),
    md("""### Utilidade de negócio: o ranking de risco funciona?
Se a equipe só consegue abordar uma fração dos clientes, vale ordenar por probabilidade. A tabela mostra a taxa real de churn por decil de risco e o *lift* sobre a média."""),
    code('''t = teste.assign(prob=prob)
t["decil_risco"] = pd.qcut(t["prob"].rank(method="first"), 10, labels=range(10, 0, -1))   # 1 = maior risco
dec = (t.groupby("decil_risco", observed=True)
         .agg(clientes=("churn", "size"), churn_real=("churn", "mean"), prob_media=("prob", "mean")).sort_index())
dec["lift"] = dec["churn_real"] / t["churn"].mean()
display(dec.round(3))
top = t.nlargest(int(len(t) * 0.2), "prob")
print(f"Abordando só os 20% de maior risco: {top['churn'].mean():.1%} deles são churners reais "
      f"(média da base: {t['churn'].mean():.1%}) e capturamos {top['churn'].sum() / t['churn'].sum():.1%} de todos os churners.")'''),
    md("## 2.6 Modelo salvo e inferência"),
    code('''import joblib
art = joblib.load(config.MODELS_DIR / "modelo_churn.joblib")
print("modelo:", art["modelo"], "| limiar:", art["limiar"], "| treinado nos cortes:", art["snapshots_treino"])
exemplo = teste.sample(3, random_state=1)
p = art["pipeline"].predict_proba(exemplo[art["colunas"]])[:, 1]
display(exemplo[["cliente_id", "recencia_dias", "freq_90d", "nota_media", "segmento"]].assign(
    prob_churn=p.round(3), previsto=(p >= art["limiar"]).astype(int), real=exemplo["churn"]))'''),
    md("""## 2.7 Conclusões, limitações e próximos passos
**Resultado.** O pipeline entrega um modelo claramente melhor que o baseline (ROC-AUC ≈ 0,82 vs 0,50; F1 ≈ 0,63 vs 0). Os números são **modestos de propósito**:
sem vazamento e com validação no futuro, prever comportamento humano é difícil. A **recência** domina; nota média e frequência recente vêm depois.

**Honestidade sobre a seleção.** Regressão logística, Random Forest e Gradient Boosting ficaram **empatados dentro do desvio da CV**. O Gradient Boosting foi
escolhido pela regra pré-definida (maior F1 médio na CV); no teste, a regressão logística foi marginalmente melhor. Na prática, o modelo linear (mais simples
e interpretável) seria uma escolha igualmente defensável.

**Limitações.**
* Dados **sintéticos**: a dificuldade e as importâncias refletem o processo gerador, não um negócio real.
* `segmento` e demais atributos do CRM são o estado **atual** do cadastro, não o histórico em T; num sistema real seria preciso versionar o cadastro (SCD tipo 2) para evitar vazamento sutil.
* Um único corte de teste: a incerteza da métrica não foi estimada por múltiplos cortes futuros.
* Sem ajuste de hiperparâmetros: foi mantido "simples" conforme o enunciado.

**Próximos passos.** Calibrar probabilidades, otimizar o limiar pelo custo real de retenção, testar vários cortes (*rolling origin*) e agendar o pipeline (Airflow/cron)."""),
]


def _salvar(cells, nome: str, executar: bool) -> Path:
    nb = new_notebook(cells=[copy.deepcopy(c) for c in cells])
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NB_DIR.mkdir(exist_ok=True)
    caminho = NB_DIR / nome
    if executar:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbf.write(nb, caminho)
    print("gerado:", caminho)
    return caminho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="executa os notebooks e embute os outputs")
    args = ap.parse_args()
    _salvar(EDA, "01_analise_exploratoria.ipynb", args.execute)
    _salvar(MODELAGEM, "02_features_modelagem.ipynb", args.execute)
    # notebook final: introducao + EDA + modelagem (setup repetido na parte 2 e inofensivo)
    _salvar(INTRO + EDA + MODELAGEM, "Projeto_Integrador_Final.ipynb", args.execute)


if __name__ == "__main__":
    main()
