-- =============================================================================
-- Consultas analiticas sobre o DW (rodar depois do ETL):
--   psql -d projeto_integrador -f sql/02_consultas_analiticas.sql
-- =============================================================================

-- 1) Receita mensal e ticket medio (pedidos nao cancelados)
SELECT d.ano, d.mes,
       COUNT(*)                          AS pedidos,
       ROUND(SUM(p.valor_total), 2)      AS receita,
       ROUND(AVG(p.valor_total), 2)      AS ticket_medio
FROM dw.fato_pedido p
JOIN dw.dim_data d ON d.data_id = p.data_id
WHERE p.status <> 'cancelado'
GROUP BY d.ano, d.mes
ORDER BY d.ano, d.mes;

-- 2) Receita e clientes por regiao (enriquecimento com a API do IBGE)
SELECT COALESCE(l.regiao, 'nao_informada') AS regiao,
       COUNT(DISTINCT c.cliente_id)        AS clientes,
       ROUND(SUM(p.valor_total), 2)        AS receita
FROM dw.fato_pedido p
JOIN dw.dim_cliente c       ON c.cliente_id = p.cliente_id
LEFT JOIN dw.dim_localidade l ON l.localidade_id = c.localidade_id
WHERE p.status <> 'cancelado'
GROUP BY 1
ORDER BY receita DESC;

-- 3) Categorias mais vendidas (join fato_item x dim_produto)
SELECT pr.categoria,
       SUM(i.quantidade)        AS unidades,
       ROUND(SUM(i.valor_item), 2) AS receita
FROM dw.fato_item_pedido i
JOIN dw.dim_produto pr ON pr.produto_id = i.produto_id
GROUP BY pr.categoria
ORDER BY receita DESC;

-- 4) Taxa de churn por segmento na data de corte de teste (ml.features_churn)
SELECT segmento,
       COUNT(*)                          AS clientes,
       ROUND(100.0 * AVG(churn), 1)      AS pct_churn
FROM ml.features_churn
WHERE data_corte = DATE '2025-09-30'
GROUP BY segmento
ORDER BY pct_churn DESC;

-- 5) Clientes de maior risco para acao de retencao (maior probabilidade prevista)
SELECT p.cliente_id, c.segmento, p.prob_churn, f.recencia_dias, f.freq_90d, f.valor_total
FROM ml.predicoes_churn p
JOIN dw.dim_cliente c       ON c.cliente_id = p.cliente_id
JOIN ml.features_churn f    ON f.cliente_id = p.cliente_id AND f.data_corte = p.data_corte
ORDER BY p.prob_churn DESC
LIMIT 20;

-- 6) Auditoria de qualidade: o que foi para a quarentena e por que
SELECT fonte, motivo, COUNT(*) AS linhas
FROM etl.rejeitados
GROUP BY fonte, motivo
ORDER BY fonte, linhas DESC;

-- 7) Integridade: pedidos sem itens ou com valor divergente da soma dos itens (esperado: 0 linhas)
SELECT p.pedido_id, p.valor_total, SUM(i.valor_item) + p.frete AS soma_itens_mais_frete
FROM dw.fato_pedido p
JOIN dw.fato_item_pedido i USING (pedido_id)
GROUP BY p.pedido_id, p.valor_total, p.frete
HAVING ABS(p.valor_total - (SUM(i.valor_item) + p.frete)) > 0.01;
