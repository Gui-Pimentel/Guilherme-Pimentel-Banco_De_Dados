-- =============================================================================
-- Projeto Integrador - Esquema do PostgreSQL
--
--   dw  : Data Warehouse em estrela (dimensoes + fatos)  - saida do ETL
--   ml  : features, metricas e predicoes do modelo        - saida da etapa de ML
--   etl : controle operacional (execucoes e linhas rejeitadas)
--
-- O script e idempotente: pode ser executado varias vezes (recria os schemas).
-- =============================================================================

DROP SCHEMA IF EXISTS ml  CASCADE;
DROP SCHEMA IF EXISTS dw  CASCADE;
DROP SCHEMA IF EXISTS etl CASCADE;

CREATE SCHEMA dw;
CREATE SCHEMA ml;
CREATE SCHEMA etl;

-- ---------------------------------------------------------------------------
-- DW: dimensoes
-- ---------------------------------------------------------------------------

-- Fonte 3 (API IBGE): hierarquia geografica oficial.
CREATE TABLE dw.dim_localidade (
    localidade_id  SERIAL       PRIMARY KEY,
    codigo_ibge    INTEGER      NOT NULL UNIQUE,
    municipio      VARCHAR(120) NOT NULL,
    uf             CHAR(2)      NOT NULL,
    estado         VARCHAR(60)  NOT NULL,
    regiao         VARCHAR(20)  NOT NULL,
    mesorregiao    VARCHAR(80),
    microrregiao   VARCHAR(80)
);
CREATE INDEX ix_dim_localidade_nome_uf ON dw.dim_localidade (municipio, uf);

-- Calendario: permite agregar por periodo sem recalcular datas na consulta.
CREATE TABLE dw.dim_data (
    data_id          DATE        PRIMARY KEY,
    ano              SMALLINT    NOT NULL,
    trimestre        SMALLINT    NOT NULL CHECK (trimestre BETWEEN 1 AND 4),
    mes              SMALLINT    NOT NULL CHECK (mes BETWEEN 1 AND 12),
    dia              SMALLINT    NOT NULL CHECK (dia BETWEEN 1 AND 31),
    dia_semana       SMALLINT    NOT NULL CHECK (dia_semana BETWEEN 0 AND 6),  -- 0 = segunda
    nome_dia_semana  VARCHAR(15) NOT NULL,
    eh_fim_de_semana BOOLEAN     NOT NULL
);

-- Fonte 2 (CRM em JSON). Dado pessoal anonimizado: guarda apenas o hash do e-mail.
CREATE TABLE dw.dim_cliente (
    cliente_id       INTEGER      PRIMARY KEY,
    email_hash       CHAR(64)     NOT NULL UNIQUE,
    genero           VARCHAR(15),
    data_nascimento  DATE,
    data_cadastro    DATE         NOT NULL,
    segmento         VARCHAR(20)  NOT NULL,
    canal_aquisicao  VARCHAR(20)  NOT NULL,
    aceita_marketing BOOLEAN      NOT NULL,
    localidade_id    INTEGER      REFERENCES dw.dim_localidade (localidade_id)
);
CREATE INDEX ix_dim_cliente_localidade ON dw.dim_cliente (localidade_id);

-- Catalogo derivado da Fonte 1 (pedidos em CSV).
CREATE TABLE dw.dim_produto (
    produto_id   INTEGER       PRIMARY KEY,
    nome         VARCHAR(120)  NOT NULL,
    categoria    VARCHAR(40)   NOT NULL,
    preco_lista  NUMERIC(10,2) NOT NULL CHECK (preco_lista > 0)
);

-- ---------------------------------------------------------------------------
-- DW: fatos (granularidade pedido e item de pedido)
-- ---------------------------------------------------------------------------

CREATE TABLE dw.fato_pedido (
    pedido_id        INTEGER       PRIMARY KEY,
    cliente_id       INTEGER       NOT NULL REFERENCES dw.dim_cliente (cliente_id),
    data_id          DATE          NOT NULL REFERENCES dw.dim_data (data_id),
    status           VARCHAR(15)   NOT NULL CHECK (status IN ('entregue', 'cancelado', 'devolvido')),
    canal_pedido     VARCHAR(20)   NOT NULL,
    forma_pagamento  VARCHAR(20)   NOT NULL,
    frete            NUMERIC(10,2) NOT NULL CHECK (frete >= 0),
    nota_avaliacao   SMALLINT      CHECK (nota_avaliacao BETWEEN 1 AND 5),
    qtd_itens        INTEGER       NOT NULL CHECK (qtd_itens > 0),
    valor_total      NUMERIC(12,2) NOT NULL CHECK (valor_total >= 0)
);
CREATE INDEX ix_fato_pedido_cliente_data ON dw.fato_pedido (cliente_id, data_id);
CREATE INDEX ix_fato_pedido_data         ON dw.fato_pedido (data_id);

CREATE TABLE dw.fato_item_pedido (
    pedido_id       INTEGER       NOT NULL REFERENCES dw.fato_pedido (pedido_id) ON DELETE CASCADE,
    item_seq        SMALLINT      NOT NULL,
    produto_id      INTEGER       NOT NULL REFERENCES dw.dim_produto (produto_id),
    quantidade      INTEGER       NOT NULL CHECK (quantidade > 0),
    preco_unitario  NUMERIC(10,2) NOT NULL CHECK (preco_unitario > 0),
    desconto_pct    NUMERIC(5,2)  NOT NULL CHECK (desconto_pct BETWEEN 0 AND 90),
    valor_item      NUMERIC(12,2) NOT NULL CHECK (valor_item >= 0),
    PRIMARY KEY (pedido_id, item_seq)
);
CREATE INDEX ix_fato_item_produto ON dw.fato_item_pedido (produto_id);

-- ---------------------------------------------------------------------------
-- ETL: controle operacional
-- ---------------------------------------------------------------------------

CREATE TABLE etl.execucoes (
    execucao_id     SERIAL       PRIMARY KEY,
    iniciado_em     TIMESTAMP    NOT NULL DEFAULT now(),
    finalizado_em   TIMESTAMP,
    status          VARCHAR(15)  NOT NULL DEFAULT 'em_andamento',
    resumo          JSONB
);

-- Linhas que falharam nas regras de qualidade ficam em quarentena (nao se perde dado).
CREATE TABLE etl.rejeitados (
    rejeitado_id  BIGSERIAL    PRIMARY KEY,
    execucao_id   INTEGER      REFERENCES etl.execucoes (execucao_id),
    fonte         VARCHAR(20)  NOT NULL,
    motivo        VARCHAR(60)  NOT NULL,
    registro      JSONB        NOT NULL,
    rejeitado_em  TIMESTAMP    NOT NULL DEFAULT now()
);
CREATE INDEX ix_rejeitados_fonte_motivo ON etl.rejeitados (fonte, motivo);

-- ---------------------------------------------------------------------------
-- ML: features (uma linha por cliente x data de corte), metricas e predicoes
-- ---------------------------------------------------------------------------

CREATE TABLE ml.features_churn (
    cliente_id              INTEGER      NOT NULL REFERENCES dw.dim_cliente (cliente_id),
    data_corte              DATE         NOT NULL,
    -- RFM e ritmo de compra
    recencia_dias           INTEGER      NOT NULL,
    frequencia_total        INTEGER      NOT NULL,
    freq_90d                INTEGER      NOT NULL,
    freq_prev_90d           INTEGER      NOT NULL,
    tendencia_freq          NUMERIC(8,4) NOT NULL,
    intervalo_medio_dias    NUMERIC(10,2),
    razao_recencia_intervalo NUMERIC(10,4),
    valor_total             NUMERIC(14,2) NOT NULL,
    ticket_medio            NUMERIC(12,2) NOT NULL,
    itens_por_pedido        NUMERIC(8,3)  NOT NULL,
    categorias_distintas    INTEGER      NOT NULL,
    -- qualidade da experiencia
    desconto_medio_pct      NUMERIC(6,2) NOT NULL,
    pct_cancelados          NUMERIC(6,4) NOT NULL,
    pct_devolvidos          NUMERIC(6,4) NOT NULL,
    nota_media              NUMERIC(4,2),
    frete_medio             NUMERIC(10,2) NOT NULL,
    pct_fim_de_semana       NUMERIC(6,4) NOT NULL,
    -- perfil
    tempo_de_casa_dias      INTEGER      NOT NULL,
    idade                   INTEGER,
    segmento                VARCHAR(20)  NOT NULL,
    canal_aquisicao         VARCHAR(20)  NOT NULL,
    aceita_marketing        BOOLEAN      NOT NULL,
    regiao                  VARCHAR(20)  NOT NULL,
    -- alvo: 1 = nao comprou nos HORIZONTE_DIAS seguintes a data_corte
    churn                   SMALLINT     NOT NULL CHECK (churn IN (0, 1)),
    PRIMARY KEY (cliente_id, data_corte)
);
CREATE INDEX ix_features_data_corte ON ml.features_churn (data_corte);

CREATE TABLE ml.metricas_modelo (
    metrica_id   SERIAL       PRIMARY KEY,
    treinado_em  TIMESTAMP    NOT NULL DEFAULT now(),
    modelo       VARCHAR(40)  NOT NULL,
    conjunto     VARCHAR(10)  NOT NULL CHECK (conjunto IN ('cv', 'teste')),
    metrica      VARCHAR(30)  NOT NULL,
    valor        NUMERIC(8,5) NOT NULL,
    selecionado  BOOLEAN      NOT NULL DEFAULT FALSE
);

CREATE TABLE ml.predicoes_churn (
    cliente_id       INTEGER      NOT NULL REFERENCES dw.dim_cliente (cliente_id),
    data_corte       DATE         NOT NULL,
    prob_churn       NUMERIC(6,5) NOT NULL,
    churn_previsto   SMALLINT     NOT NULL,
    churn_real       SMALLINT     NOT NULL,
    PRIMARY KEY (cliente_id, data_corte)
);

-- ---------------------------------------------------------------------------
-- Visao de apoio: pedidos validos (cancelados nao contam como compra)
-- ---------------------------------------------------------------------------
CREATE VIEW dw.vw_pedidos_validos AS
SELECT *
FROM dw.fato_pedido
WHERE status <> 'cancelado';
