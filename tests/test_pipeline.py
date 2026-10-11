"""Testes unitarios das regras de transformacao e, principalmente, da ausencia de vazamento nas features.

Rodam sem banco de dados: usam DataFrames pequenos montados a mao.
    pytest -q
"""
from datetime import date

import pandas as pd
import pytest

from src import config, features, transform


# --------------------------------------------------------------------------- transform
def test_normalizar_texto_remove_acento_caixa_e_espacos():
    s = pd.Series(["  São Paulo ", "BRASÍLIA", "belém"])
    assert transform.normalizar_texto(s).tolist() == ["sao paulo", "brasilia", "belem"]


def test_parse_data_aceita_iso_e_formato_brasileiro():
    s = pd.Series(["2025-03-01", "01/03/2025", "31/02/2025", None])
    out = transform._parse_data(s)
    assert out.iloc[0] == out.iloc[1] == pd.Timestamp("2025-03-01")
    assert out.iloc[2:].isna().all()


def test_email_vira_hash_estavel_e_nao_reversivel_a_olho():
    h1, h2 = transform._hash_email("A@x.com "), transform._hash_email("a@x.com")
    assert h1 == h2 and len(h1) == 64 and "x.com" not in h1


def _pedidos_raw(**sobrescrever):
    base = {"pedido_id": "1", "item_seq": "1", "data_pedido": "2025-01-10", "cliente_id": "7", "produto_id": "101",
            "produto_nome": "p", "categoria": "Casa", "quantidade": "2", "preco_unitario": "100",
            "desconto_pct": "10", "frete": "5", "status": " ENTREGUE ", "forma_pagamento": "pix",
            "canal_pedido": "app", "nota_avaliacao": "5"}
    base.update(sobrescrever)
    return base


def test_pedido_valido_e_padronizado_e_valorado():
    itens, rej = transform.transformar_pedidos(pd.DataFrame([_pedidos_raw()]), {7})
    assert rej.empty and itens.loc[0, "status"] == "entregue"
    assert itens.loc[0, "valor_item"] == pytest.approx(2 * 100 * 0.9)


@pytest.mark.parametrize("campo,valor,motivo", [
    ("quantidade", "0", "quantidade_invalida"),
    ("data_pedido", "31/02/2025", "data_invalida_ou_fora_da_janela"),
    ("data_pedido", "2027-03-15", "data_invalida_ou_fora_da_janela"),
    ("desconto_pct", "150", "desconto_invalido"),
    ("status", "perdido", "status_invalido"),
    ("cliente_id", "999", "cliente_inexistente_no_crm"),
])
def test_regras_de_qualidade_enviam_para_quarentena(campo, valor, motivo):
    itens, rej = transform.transformar_pedidos(pd.DataFrame([_pedidos_raw(**{campo: valor})]), {7})
    assert itens.empty
    assert rej["_motivo"].iloc[0] == motivo


def test_item_invalido_derruba_o_pedido_inteiro():
    raw = pd.DataFrame([_pedidos_raw(), _pedidos_raw(item_seq="2", quantidade="-1")])
    itens, rej = transform.transformar_pedidos(raw, {7})
    assert itens.empty
    assert sorted(rej["_motivo"]) == ["pedido_com_item_invalido", "quantidade_invalida"]


def test_duplicata_exata_e_removida():
    raw = pd.DataFrame([_pedidos_raw(), _pedidos_raw()])
    itens, _ = transform.transformar_pedidos(raw, {7})
    assert len(itens) == 1


# --------------------------------------------------------------------------- features / vazamento
def _base_features():
    def ped(i, cli, dia, status="entregue"):
        return {"pedido_id": i, "cliente_id": cli, "data": pd.Timestamp(dia), "status": status, "frete": 10.0,
                "nota_avaliacao": 4.0, "qtd_itens": 1, "valor_total": 100.0, "eh_fim_de_semana": False,
                "desconto_medio": 0.0}
    pedidos = pd.DataFrame([
        ped(1, 1, "2025-08-01"), ped(2, 1, "2025-09-20"),          # cliente 1: ativo e volta em outubro
        ped(3, 1, "2025-10-15"),
        ped(4, 2, "2025-08-10"),                                    # cliente 2: some depois -> churn
        ped(5, 2, "2025-10-05", status="cancelado"),                # cancelado NAO conta como retorno
        ped(6, 3, "2025-02-01"),                                    # cliente 3: fora da base ativa
    ])
    categorias = pd.DataFrame({"pedido_id": [1, 2, 3, 4, 5, 6], "categoria": ["Casa"] * 6})
    clientes = pd.DataFrame({
        "cliente_id": [1, 2, 3], "data_cadastro": pd.Timestamp("2025-01-01"),
        "data_nascimento": pd.Timestamp("1990-01-01"), "segmento": "basico", "canal_aquisicao": "ads",
        "aceita_marketing": True, "regiao": "Norte"})
    return pedidos, categorias, clientes


def test_alvo_e_populacao_na_data_de_corte():
    f = features.construir_features(*_base_features(), data_corte=date(2025, 9, 30))
    assert set(f["cliente_id"]) == {1, 2}                         # 3 esta fora da base ativa (>180 dias)
    alvo = f.set_index("cliente_id")["churn"]
    assert alvo[1] == 0 and alvo[2] == 1


def test_features_nao_enxergam_o_futuro():
    pedidos, categorias, clientes = _base_features()
    f1 = features.construir_features(pedidos, categorias, clientes, date(2025, 9, 30)).set_index("cliente_id")
    # se removermos TODO o futuro, as features em 30/09 devem ser identicas
    so_passado = pedidos[pedidos["data"] <= pd.Timestamp("2025-09-30")]
    f2 = features.construir_features(so_passado, categorias, clientes, date(2025, 9, 30)).set_index("cliente_id")
    cols = features.FEATURES_NUMERICAS
    pd.testing.assert_frame_equal(f1[cols], f2[cols])


def test_recencia_e_frequencia_do_cliente_1():
    f = features.construir_features(*_base_features(), data_corte=date(2025, 9, 30)).set_index("cliente_id")
    assert f.loc[1, "recencia_dias"] == 10 and f.loc[1, "frequencia_total"] == 2
    assert f.loc[1, "intervalo_medio_dias"] == 50


def test_janela_do_alvo_nao_pode_passar_do_fim_dos_dados():
    with pytest.raises(ValueError):
        features.construir_features(*_base_features(), data_corte=config.DATA_FIM)
