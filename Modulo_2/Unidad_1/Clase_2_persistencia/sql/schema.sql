-- =============================================================================
-- Esquema de la base de datos "ventas_ia" en SQL puro.
--
-- Este archivo es el EQUIVALENTE de src/schema.py.
-- SQLAlchemy genera exactamente este SQL a partir de las definiciones de Python.
--
-- Sirve para dos cosas en clase:
--   1) Comparar el codigo Python con el SQL real que se ejecuta.
--   2) Poder crear el esquema a mano desde pgAdmin o psql si hiciera falta:
--        psql -U postgres -d ventas_ia -f sql/schema.sql
--
-- Normalmente NO hace falta ejecutarlo: lo crea scripts/create_schema.py.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. CLIENTES  (tabla maestra / dimension)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clientes (
    cliente_id     INTEGER      PRIMARY KEY,          -- identificador unico del cliente
    nombre         VARCHAR(120) NOT NULL,             -- NOT NULL: obligatorio
    email          VARCHAR(150) NOT NULL UNIQUE,      -- UNIQUE: no se puede repetir
    ciudad         VARCHAR(80),
    pais           VARCHAR(80),
    segmento       VARCHAR(20),
    fecha_registro DATE         NOT NULL,
    -- CHECK: restringe los valores validos de la columna
    CONSTRAINT ck_clientes_segmento
        CHECK (segmento IN ('bronce', 'plata', 'oro', 'platino'))
);


-- -----------------------------------------------------------------------------
-- 2. PRODUCTOS  (tabla maestra / dimension)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS productos (
    producto_id INTEGER        PRIMARY KEY,
    nombre      VARCHAR(150)   NOT NULL,
    categoria   VARCHAR(60)    NOT NULL,
    -- NUMERIC(10,2) para dinero: precision exacta. Nunca usar FLOAT para importes.
    precio      NUMERIC(10, 2) NOT NULL,
    coste       NUMERIC(10, 2) NOT NULL,
    stock       INTEGER        NOT NULL DEFAULT 0,
    activo      BOOLEAN        NOT NULL DEFAULT TRUE,
    CONSTRAINT ck_productos_precio_positivo CHECK (precio >= 0),
    CONSTRAINT ck_productos_stock_positivo  CHECK (stock  >= 0)
);


-- -----------------------------------------------------------------------------
-- 3. VENTAS  (tabla de hechos: lo que ocurre en el negocio)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ventas (
    venta_id        INTEGER        PRIMARY KEY,
    -- REFERENCES = clave ajena (FOREIGN KEY).
    -- PostgreSQL impedira insertar una venta de un cliente que no existe.
    cliente_id      INTEGER        NOT NULL REFERENCES clientes(cliente_id),
    producto_id     INTEGER        NOT NULL REFERENCES productos(producto_id),
    fecha_venta     DATE           NOT NULL,
    cantidad        INTEGER        NOT NULL,
    precio_unitario NUMERIC(10, 2) NOT NULL,
    descuento       NUMERIC(4, 2)  NOT NULL DEFAULT 0,
    total           NUMERIC(12, 2) NOT NULL,
    canal           VARCHAR(20),
    CONSTRAINT ck_ventas_cantidad_positiva CHECK (cantidad > 0),
    CONSTRAINT ck_ventas_descuento         CHECK (descuento >= 0 AND descuento <= 1)
);


-- -----------------------------------------------------------------------------
-- 4. PREDICCIONES  (salida del modelo de Machine Learning)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predicciones (
    -- SERIAL: PostgreSQL genera el id automaticamente en cada INSERT.
    prediccion_id       SERIAL         PRIMARY KEY,
    modelo              VARCHAR(80)    NOT NULL,
    version             VARCHAR(20)    NOT NULL,
    -- DEFAULT now(): si no enviamos la fecha, la pone el servidor.
    fecha_prediccion    TIMESTAMP      NOT NULL DEFAULT now(),
    cliente_id          INTEGER        NOT NULL REFERENCES clientes(cliente_id),
    valor_predicho      NUMERIC(12, 4) NOT NULL,
    confianza           NUMERIC(5, 4),
    -- Se rellena mas tarde, cuando conocemos lo que paso de verdad.
    resultado_real      NUMERIC(12, 4),
    estado              VARCHAR(20)    NOT NULL DEFAULT 'ok',
    tiempo_ejecucion_ms INTEGER,
    CONSTRAINT ck_predicciones_confianza CHECK (confianza >= 0 AND confianza <= 1),
    CONSTRAINT ck_predicciones_estado    CHECK (estado IN ('ok', 'error', 'pendiente'))
);


-- -----------------------------------------------------------------------------
-- 5. AUDITORIA  (trazabilidad de los procesos: quien hizo que y cuando)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS auditoria (
    auditoria_id         SERIAL         PRIMARY KEY,
    fecha_hora           TIMESTAMP      NOT NULL DEFAULT now(),
    proceso              VARCHAR(80)    NOT NULL,   -- ej: 'carga_csv'
    operacion            VARCHAR(30)    NOT NULL,   -- ej: 'INSERT'
    tabla_afectada       VARCHAR(60),
    registros_procesados INTEGER        NOT NULL DEFAULT 0,
    estado               VARCHAR(20)    NOT NULL,   -- 'exito' o 'error'
    mensaje              TEXT,
    duracion_segundos    NUMERIC(10, 3),
    CONSTRAINT ck_auditoria_estado CHECK (estado IN ('exito', 'error'))
);


-- -----------------------------------------------------------------------------
-- 6. HISTORIAL_ENTRENAMIENTO  (Clase 4: metadatos de cada dataset congelado)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS historial_entrenamiento (
    snapshot_id     SERIAL         PRIMARY KEY,
    nombre_dataset  VARCHAR(80)    NOT NULL,
    version         VARCHAR(20)    NOT NULL,
    fecha_creacion  TIMESTAMP      NOT NULL DEFAULT now(),
    fecha_corte     DATE           NOT NULL,          -- hasta que dia mira el modelo
    consulta_sql    TEXT           NOT NULL,          -- linaje: el SQL exacto
    n_filas         INTEGER        NOT NULL DEFAULT 0,
    n_columnas      INTEGER,
    descripcion     TEXT,
    parametros      TEXT,                             -- JSON o texto con filtros
    checksum        VARCHAR(64),                      -- SHA-256 de las filas
    estado          VARCHAR(20)    NOT NULL DEFAULT 'activo',
    CONSTRAINT ck_historial_estado
        CHECK (estado IN ('activo', 'archivado', 'error'))
);


-- -----------------------------------------------------------------------------
-- 7. DATASET_ENTRENAMIENTO  (Clase 4: filas congeladas, features + target)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_entrenamiento (
    fila_id                    SERIAL         PRIMARY KEY,
    snapshot_id                INTEGER        NOT NULL
        REFERENCES historial_entrenamiento(snapshot_id) ON DELETE CASCADE,
    cliente_id                 INTEGER        NOT NULL REFERENCES clientes(cliente_id),
    segmento                   VARCHAR(20),
    pais                       VARCHAR(80),
    num_compras                INTEGER        NOT NULL,
    gasto_historico            NUMERIC(12, 2) NOT NULL,
    ticket_medio               NUMERIC(12, 2) NOT NULL,
    cantidad_total             INTEGER        NOT NULL,
    dias_desde_ultima_compra   INTEGER,
    canal_preferido            VARCHAR(20),
    categoria_favorita         VARCHAR(60),
    -- Target: gasto en los 90 dias posteriores a fecha_corte.
    gasto_siguiente_periodo    NUMERIC(12, 2) NOT NULL DEFAULT 0,
    fecha_corte                DATE           NOT NULL
);
