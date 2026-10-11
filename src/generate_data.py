"""Gera as duas fontes locais do projeto (a terceira, a API do IBGE, e coletada em extract.py).

Fonte 1 - ``data/raw/pedidos.csv``      : exportacao do sistema transacional (nivel item de pedido).
Fonte 2 - ``data/raw/clientes_crm.json``: cadastro de clientes exportado do CRM.

Os dados sao sinteticos (nao ha dados reais de clientes no repositorio), mas seguem um processo
gerador com sinal verdadeiro: cada cliente tem uma satisfacao latente, uma taxa de compra e um
instante de abandono. Antes de abandonar, o ritmo de compras decai - e esse comportamento que o
modelo tenta capturar. O churn NAO e escrito em nenhum arquivo: ele so e derivado depois, a
partir do historico de compras (ver features.py).

Os arquivos saem propositalmente "sujos" (duplicatas, orfaos, formatos mistos, nulos, valores
invalidos) para que o ETL tenha o que tratar.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config

N_CLIENTES = 3200

CIDADES = [  # (municipio, uf, peso) - grafia oficial IBGE
    ("São Paulo", "SP", 14), ("Rio de Janeiro", "RJ", 9), ("Belo Horizonte", "MG", 6),
    ("Salvador", "BA", 5), ("Fortaleza", "CE", 5), ("Brasília", "DF", 5), ("Curitiba", "PR", 5),
    ("Recife", "PE", 4), ("Porto Alegre", "RS", 4), ("Manaus", "AM", 3), ("Belém", "PA", 3),
    ("Goiânia", "GO", 3), ("Macapá", "AP", 3), ("Santana", "AP", 2), ("Campinas", "SP", 3),
    ("São Luís", "MA", 2), ("Maceió", "AL", 2), ("Natal", "RN", 2), ("Teresina", "PI", 2),
    ("Campo Grande", "MS", 2), ("João Pessoa", "PB", 2), ("Florianópolis", "SC", 2),
    ("Vitória", "ES", 2), ("Cuiabá", "MT", 2), ("Aracaju", "SE", 1), ("Porto Velho", "RO", 1),
    ("Rio Branco", "AC", 1), ("Boa Vista", "RR", 1), ("Palmas", "TO", 1), ("Uberlândia", "MG", 2),
    ("Londrina", "PR", 2), ("Joinville", "SC", 2), ("Ribeirão Preto", "SP", 2), ("Santos", "SP", 2),
    ("Niterói", "RJ", 2), ("Juiz de Fora", "MG", 1), ("Caxias do Sul", "RS", 1),
    ("Feira de Santana", "BA", 1), ("Sorocaba", "SP", 2), ("Guarulhos", "SP", 2),
]

CATEGORIAS = {  # categoria: (preco medio, peso de popularidade)
    "Eletronicos": (900, 3), "Informatica": (1500, 2), "Casa": (180, 4), "Moda": (120, 5),
    "Esporte": (200, 2), "Beleza": (70, 4), "Livros": (45, 3), "Brinquedos": (90, 2),
}
SEGMENTOS = ["basico", "standard", "premium"]
CANAIS_AQUISICAO = ["organico", "ads", "social", "email", "indicacao"]


def _fabricar_produtos(rng: np.random.Generator) -> pd.DataFrame:
    linhas, pid = [], 100
    for cat, (preco, peso) in CATEGORIAS.items():
        for k in range(1, 4 + peso + 1):
            pid += 1
            linhas.append({
                "produto_id": pid,
                "produto_nome": f"{cat} Item {k:02d}",
                "categoria": cat,
                "preco_lista": round(float(preco * rng.lognormal(0, 0.35)), 2),
                "peso": peso,
            })
    return pd.DataFrame(linhas)


def _simular_clientes(rng: np.random.Generator) -> pd.DataFrame:
    n = N_CLIENTES
    pesos_cid = np.array([c[2] for c in CIDADES], dtype=float)
    cid_idx = rng.choice(len(CIDADES), n, p=pesos_cid / pesos_cid.sum())
    cad_inicio = date(2023, 10, 1)
    span = (date(2025, 8, 31) - cad_inicio).days

    cli = pd.DataFrame({
        "cliente_id": np.arange(1, n + 1),
        "segmento": rng.choice(SEGMENTOS, n, p=[0.55, 0.30, 0.15]),
        "canal_aquisicao": rng.choice(CANAIS_AQUISICAO, n, p=[0.30, 0.25, 0.20, 0.10, 0.15]),
        "idade": np.clip(rng.normal(36, 11, n), 18, 75).round().astype(int),
        "genero": rng.choice(["F", "M", "outro"], n, p=[0.52, 0.45, 0.03]),
        "aceita_marketing": rng.random(n) < 0.6,
        "cidade": [CIDADES[i][0] for i in cid_idx],
        "uf": [CIDADES[i][1] for i in cid_idx],
        "cadastro": [cad_inicio + timedelta(days=int(d)) for d in rng.integers(0, span, n)],
        "satisfacao": rng.normal(0, 1, n),
    })
    return cli


def _simular_pedidos(rng: np.random.Generator, cli: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    seg_taxa = {"basico": 1.0, "standard": 1.3, "premium": 1.9}
    seg_risco = {"basico": 0.10, "standard": 0.0, "premium": -0.35}
    can_risco = {"organico": 0.0, "ads": 0.25, "social": 0.10, "email": -0.05, "indicacao": -0.30}
    d0 = config.DATA_INICIO
    dias_janela = (config.DATA_FIM - d0).days

    prod_p = (prod["peso"] / prod["peso"].sum()).to_numpy()
    registros = []

    for c in cli.itertuples(index=False):
        s = c.satisfacao
        taxa = 1.0 * seg_taxa[c.segmento] * rng.lognormal(0, 0.5) * (0.9 if c.canal_aquisicao == "ads" else 1.0)
        # risco mensal de abandono: menor satisfacao e certos perfis => abandona mais cedo
        hazard = 0.045 * np.exp(-0.55 * s + seg_risco[c.segmento] + can_risco[c.canal_aquisicao]
                               + (0.0 if c.aceita_marketing else 0.15))
        cad_off = (c.cadastro - d0).days
        abandono = cad_off + rng.exponential(30.0 / hazard)          # dia (desde d0) em que para de comprar

        t0, t1 = max(cad_off, 0), min(abandono, dias_janela)
        if t1 <= t0:
            continue
        # pausa temporaria (falso alarme): ~20% dos clientes ficam 60-120 dias quase sem comprar
        pausa = None
        if rng.random() < 0.20:
            ini = rng.uniform(t0, max(t0 + 1, t1 - 30))
            pausa = (ini, ini + rng.uniform(60, 120))

        n_cand = rng.poisson(taxa / 30.0 * (t1 - t0) * 1.6)
        tempos = rng.uniform(t0, t1, n_cand)
        aceita = np.minimum(1.0, (abandono - tempos) / 60.0) * 0.7 + 0.3  # ritmo cai antes de abandonar
        aceita = np.where(abandono > dias_janela + 400, 1.0, aceita)
        if pausa:
            aceita = np.where((tempos > pausa[0]) & (tempos < pausa[1]), aceita * 0.08, aceita)
        tempos = tempos[rng.random(n_cand) < aceita / 1.6]
        tempos = np.append(tempos, t0 + rng.exponential(4.0))          # compra de boas-vindas
        tempos = np.sort(tempos[tempos <= t1])

        for t in tempos:
            dt = d0 + timedelta(days=int(t))
            p_canc = min(0.25, 0.04 * np.exp(-0.4 * s))
            p_dev = min(0.30, 0.05 * np.exp(-0.5 * s))
            u = rng.random()
            status = "cancelado" if u < p_canc else ("devolvido" if u < p_canc + p_dev else "entregue")
            nota = np.nan
            if status != "cancelado" and rng.random() < 0.55:
                nota = float(np.clip(round(3.8 + 0.8 * s - (1.5 if status == "devolvido" else 0) + rng.normal(0, 0.8)), 1, 5))
            n_itens = 1 + rng.poisson(0.6 + (0.3 if c.segmento == "premium" else 0))
            frete = 0.0 if c.segmento == "premium" else round(float(rng.lognormal(2.8, 0.4)), 2)
            registros.append({
                "o_cliente_id": c.cliente_id, "o_data": dt, "o_status": status, "o_nota": nota,
                "o_frete": frete, "o_n_itens": n_itens,
                "o_pagto": rng.choice(["cartao_credito", "pix", "boleto", "cartao_debito"], p=[0.55, 0.30, 0.10, 0.05]),
                "o_canal": rng.choice(["app", "site", "marketplace"], p=[0.40, 0.45, 0.15]),
            })

    reg = pd.DataFrame(registros).sort_values(["o_data", "o_cliente_id"], kind="stable").reset_index(drop=True)
    reg["pedido_id"] = np.arange(50001, 50001 + len(reg))

    linhas = []
    for r in reg.itertuples(index=False):
        escolhidos = rng.choice(len(prod), r.o_n_itens, replace=False, p=prod_p) if r.o_n_itens <= len(prod) else []
        for seq, idx in enumerate(escolhidos, start=1):
            p = prod.iloc[idx]
            linhas.append({
                "pedido_id": r.pedido_id, "item_seq": seq, "data_pedido": r.o_data.isoformat(),
                "cliente_id": r.o_cliente_id, "produto_id": int(p.produto_id), "produto_nome": p.produto_nome,
                "categoria": p.categoria, "quantidade": 1 + int(rng.poisson(0.25)),
                "preco_unitario": float(p.preco_lista), "desconto_pct": float(rng.choice([0, 5, 10, 15, 20], p=[0.45, 0.2, 0.15, 0.12, 0.08])),
                "frete": r.o_frete, "status": r.o_status, "forma_pagamento": r.o_pagto,
                "canal_pedido": r.o_canal, "nota_avaliacao": r.o_nota,
            })
    return pd.DataFrame(linhas)


def _sujar_pedidos(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    pick = lambda frac: rng.choice(n, int(n * frac), replace=False)  # noqa: E731

    df["status"] = df["status"].astype(object)
    idx = pick(0.10)
    df.loc[idx, "status"] = [s.upper() if rng.random() < 0.5 else f" {s} " for s in df.loc[idx, "status"]]
    df.loc[pick(0.006), "quantidade"] = rng.choice([0, -1], size=int(n * 0.006))
    df.loc[pick(0.005), "preco_unitario"] = np.nan
    df.loc[pick(0.003), "desconto_pct"] = 150.0
    df["data_pedido"] = df["data_pedido"].astype(object)
    df.loc[pick(0.003), "data_pedido"] = "31/02/2025"
    df.loc[pick(0.003), "data_pedido"] = "2027-03-15"
    df.loc[pick(0.008), "cliente_id"] = rng.integers(90000, 99999, int(n * 0.008))   # cliente inexistente no CRM
    duplicadas = df.sample(frac=0.015, random_state=config.SEED)
    return pd.concat([df, duplicadas], ignore_index=True).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)


def _montar_crm(rng: np.random.Generator, cli: pd.DataFrame) -> list[dict]:
    def variar_texto(s: str) -> str:
        r = rng.random()
        if r < 0.12:
            return s.lower()
        if r < 0.20:
            return s.upper()
        if r < 0.26:
            return f"  {s} "
        if r < 0.32:
            import unicodedata
            return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
        return s

    registros = []
    for c in cli.itertuples(index=False):
        cad = c.cadastro
        nasc = date(cad.year, cad.month, min(cad.day, 28)) - timedelta(days=int(c.idade * 365.25) + int(rng.integers(0, 300)))
        reg = {
            "id_cliente": int(c.cliente_id),
            "email": f"cliente{c.cliente_id:05d}@exemplo.com.br" if rng.random() > 0.003 else None,
            "genero": {"F": rng.choice(["F", "feminino", "Feminino"]), "M": rng.choice(["M", "masculino", "Masculino"]),
                       "outro": "outro"}[c.genero] if rng.random() > 0.04 else None,
            "data_nascimento": nasc.isoformat() if rng.random() > 0.02 else None,
            "data_cadastro": cad.isoformat() if rng.random() > 0.15 else cad.strftime("%d/%m/%Y"),
            "segmento": variar_texto(c.segmento),
            "canal_aquisicao": variar_texto(c.canal_aquisicao),
            "aceita_marketing": (lambda v: (v if rng.random() < 0.5 else ("sim" if v else "não") if rng.random() < 0.5 else int(v)))(bool(c.aceita_marketing)),
            "cidade": variar_texto(c.cidade),
            "uf": c.uf.lower() if rng.random() < 0.05 else c.uf,
            "atualizado_em": "2025-12-31",
        }
        if rng.random() < 0.01:                       # data de nascimento implausivel
            reg["data_nascimento"] = "1900-01-01"
        if rng.random() < 0.01:                       # municipio que nao existe + UF invalida
            reg["cidade"], reg["uf"] = "Atlantida", "XX"
        registros.append(reg)

    # ~1% de registros duplicados no CRM (mesma chave, versao antiga com segmento diferente)
    for i in rng.choice(len(registros), int(len(registros) * 0.01), replace=False):
        antigo = dict(registros[i])
        antigo["segmento"] = "basico"
        antigo["atualizado_em"] = "2025-06-30"
        registros.append(antigo)
    rng.shuffle(registros)
    return registros


def gerar(seed: int = config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    prod = _fabricar_produtos(rng)
    cli = _simular_clientes(rng)
    pedidos = _simular_pedidos(rng, cli, prod)
    pedidos_sujos = _sujar_pedidos(rng, pedidos)
    crm = _montar_crm(rng, cli)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pedidos_sujos.to_csv(config.PEDIDOS_CSV, index=False)
    with open(config.CLIENTES_JSON, "w", encoding="utf-8") as f:
        json.dump(crm, f, ensure_ascii=False, indent=1)

    resumo = {
        "clientes_crm": len(crm), "linhas_pedidos_csv": len(pedidos_sujos),
        "pedidos_distintos": int(pedidos["pedido_id"].nunique()),
    }
    print("Fontes geradas:", resumo)
    return resumo


if __name__ == "__main__":
    gerar()
