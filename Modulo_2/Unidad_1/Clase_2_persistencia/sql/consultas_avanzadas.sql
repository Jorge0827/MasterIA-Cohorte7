-- =============================================================================
-- Consultas avanzadas de la Clase 4.
--
-- Pensadas para pgAdmin, psql o el notebook (via src.db.consultar).
-- Cada bloque ensena UNA idea. Lee el comentario ANTES de ejecutar.
--
--   psql -U postgres -d ventas_ia -f sql/consultas_avanzadas.sql
-- =============================================================================


-- -----------------------------------------------------------------------------
-- A. Agregaciones complejas
-- -----------------------------------------------------------------------------

-- A1. CASE WHEN: convertimos un numero continuo (total) en un cubo de negocio.
--     Pregunta: ¿cómo se reparte la facturación entre tickets pequeños,
--     medios y grandes? (umbrales de negocio: 30 € y 120 €)
SELECT
    CASE
        WHEN total < 30  THEN 'pequeno'
        WHEN total < 120 THEN 'medio'
        ELSE 'grande'
    END                    AS tipo_ticket,
    COUNT(*)               AS num_ventas,
    ROUND(AVG(total), 2)   AS ticket_medio,
    SUM(total)             AS facturacion
FROM ventas
GROUP BY 1
ORDER BY facturacion DESC;


-- A2. CTE (WITH): damos nombre a un resultado intermedio.
--     Pregunta: ¿cuánto facturó cada mes y qué % del total aporta?
WITH mensual AS (
    SELECT
        DATE_TRUNC('month', fecha_venta)::date AS mes,
        SUM(total)                             AS facturacion
    FROM ventas
    GROUP BY 1
)
SELECT
    mes,
    facturacion,
    -- Subconsulta correlacionada al total: porcentaje de cada mes.
    ROUND(
        100.0 * facturacion / (SELECT SUM(facturacion) FROM mensual),
        2
    ) AS pct_del_total
FROM mensual
ORDER BY mes;


-- A3. FILTER: agregacion condicional sin un CASE extra.
--     Pregunta: ¿cuántos clientes hay por país y cómo se reparte el segmento?
SELECT
    c.pais,
    COUNT(*)                              AS clientes,
    COUNT(*) FILTER (WHERE segmento = 'platino') AS platino,
    COUNT(*) FILTER (WHERE segmento = 'oro')     AS oro,
    COUNT(*) FILTER (WHERE segmento = 'plata')   AS plata,
    COUNT(*) FILTER (WHERE segmento = 'bronce')  AS bronce
FROM clientes c
GROUP BY c.pais
ORDER BY clientes DESC;


-- A4. HAVING vs WHERE (repaso con trampa habitual).
--     Pregunta: en el canal web, ¿qué categorías superan 500.000 €?
--     WHERE filtra FILAS (antes de agrupar). HAVING filtra GRUPOS.
SELECT
    p.categoria,
    SUM(v.total) AS facturacion
FROM ventas v
JOIN productos p ON v.producto_id = p.producto_id
WHERE v.canal = 'web'                 -- filtra filas
GROUP BY p.categoria
HAVING SUM(v.total) > 500000          -- filtra categorias ya agregadas
ORDER BY facturacion DESC;


-- A5. EXTRACT / DATE_TRUNC: el tiempo como dimension de analisis.
--     Pregunta: ¿cómo evoluciona la facturación por año y trimestre?
SELECT
    EXTRACT(YEAR  FROM fecha_venta)::int AS anio,
    EXTRACT(QUARTER FROM fecha_venta)::int AS trimestre,
    COUNT(*) AS num_ventas,
    SUM(total) AS facturacion
FROM ventas
GROUP BY 1, 2
ORDER BY 1, 2;


-- -----------------------------------------------------------------------------
-- B. Window functions  (la idea: NO colapsar filas)
-- -----------------------------------------------------------------------------

-- B1. RANK vs DENSE_RANK vs ROW_NUMBER sobre el top de productos.
--     Pregunta: ¿cuáles son los productos que más facturan y cómo se numera
--     el puesto si hay empate?
SELECT
    p.nombre,
    p.categoria,
    SUM(v.total) AS facturacion,
    ROW_NUMBER() OVER (ORDER BY SUM(v.total) DESC) AS row_number,
    RANK()       OVER (ORDER BY SUM(v.total) DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY SUM(v.total) DESC) AS dense_rank
FROM ventas v
JOIN productos p ON v.producto_id = p.producto_id
GROUP BY p.nombre, p.categoria
ORDER BY facturacion DESC
LIMIT 15;


-- B2. PARTITION BY: ranking DENTRO de cada categoria (top 3).
--     Pregunta: ¿cuáles son los 3 productos que más facturan en cada categoría?
SELECT *
FROM (
    SELECT
        p.categoria,
        p.nombre,
        SUM(v.total) AS facturacion,
        ROW_NUMBER() OVER (
            PARTITION BY p.categoria
            ORDER BY SUM(v.total) DESC
        ) AS puesto
    FROM ventas v
    JOIN productos p ON v.producto_id = p.producto_id
    GROUP BY p.categoria, p.nombre
) ranking
WHERE puesto <= 3
ORDER BY categoria, puesto;


-- B3. LAG: comparar este mes con el anterior (variacion %).
--     Pregunta: ¿este mes facturamos más o menos que el anterior, y en qué %?
WITH mensual AS (
    SELECT
        DATE_TRUNC('month', fecha_venta)::date AS mes,
        SUM(total)                             AS facturacion
    FROM ventas
    GROUP BY 1
)
SELECT
    mes,
    facturacion,
    LAG(facturacion) OVER (ORDER BY mes) AS mes_anterior,
    ROUND(
        100.0 * (
            facturacion - LAG(facturacion) OVER (ORDER BY mes)
        ) / NULLIF(LAG(facturacion) OVER (ORDER BY mes), 0),
        2
    ) AS variacion_pct
FROM mensual
ORDER BY mes;


-- B4. Suma acumulada (running total).
--     Pregunta: ¿cuánto llevamos facturado desde el origen, mes a mes?
SELECT
    DATE_TRUNC('month', fecha_venta)::date AS mes,
    SUM(total) AS facturacion_mes,
    SUM(SUM(total)) OVER (ORDER BY DATE_TRUNC('month', fecha_venta)) AS acumulado
FROM ventas
GROUP BY 1
ORDER BY 1;


-- B5. Media movil de 3 meses (el mes actual y los 2 anteriores).
--     Pregunta: ¿cuál es la tendencia suavizada de facturación a 3 meses?
WITH mensual AS (
    SELECT
        DATE_TRUNC('month', fecha_venta)::date AS mes,
        SUM(total)                             AS facturacion
    FROM ventas
    GROUP BY 1
)
SELECT
    mes,
    facturacion,
    ROUND(
        AVG(facturacion) OVER (
            ORDER BY mes
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS media_movil_3m
FROM mensual
ORDER BY mes;


-- B6. NTILE(4): parte a los clientes en cuartiles de gasto.
--     Pregunta: si partimos a los clientes en 4 grupos de gasto, ¿cómo se
--     diferencia el que menos gasta (Q1) del VIP (Q4)?
SELECT
    cuartil,
    COUNT(*)             AS clientes,
    ROUND(MIN(gasto), 2) AS gasto_min,
    ROUND(MAX(gasto), 2) AS gasto_max,
    ROUND(AVG(gasto), 2) AS gasto_medio
FROM (
    SELECT
        c.cliente_id,
        SUM(v.total) AS gasto,
        NTILE(4) OVER (ORDER BY SUM(v.total)) AS cuartil
    FROM ventas v
    JOIN clientes c ON v.cliente_id = c.cliente_id
    GROUP BY c.cliente_id
) t
GROUP BY cuartil
ORDER BY cuartil;


-- -----------------------------------------------------------------------------
-- C. Vistas (una vez creadas con src.vistas.crear_vistas o sql/vistas.sql)
-- -----------------------------------------------------------------------------

-- C1. Pregunta: ¿cuál es el panorama global (ventas, clientes, ticket, periodo)?
SELECT * FROM v_resumen_ventas;

-- C2. Pregunta: ¿cómo evoluciona el negocio mes a mes?
SELECT * FROM v_facturacion_mensual ORDER BY mes;

-- C3. Pregunta: ¿qué categorías venden más y cuáles dejan más margen?
SELECT * FROM v_margen_categoria ORDER BY facturacion DESC;

-- C4. Pregunta: ¿qué canal factura más y con qué ticket medio?
SELECT * FROM v_rendimiento_canal ORDER BY facturacion DESC;

-- C5. Pregunta: ¿quiénes son los 10 clientes que más gastan en la empresa?
SELECT * FROM v_ranking_clientes ORDER BY ranking_global LIMIT 10;


-- -----------------------------------------------------------------------------
-- D. Historial de entrenamiento
-- -----------------------------------------------------------------------------

-- D1. Catalogo de snapshots (el "registro" de datasets).
--     Pregunta: ¿qué datasets de entrenamiento tenemos versionados?
SELECT
    snapshot_id,
    nombre_dataset,
    version,
    fecha_corte,
    n_filas,
    n_columnas,
    LEFT(checksum, 12) AS checksum_corto,
    fecha_creacion
FROM historial_entrenamiento
ORDER BY snapshot_id DESC;


-- D2. Muestra de un snapshot (cambia el id si hace falta).
--     Pregunta: ¿qué aspecto tiene el dataset congelado del snapshot 1?
SELECT *
FROM dataset_entrenamiento
WHERE snapshot_id = 1
ORDER BY cliente_id
LIMIT 20;


-- D3. El target no es uniforme: muchos ceros (clientes que no volvieron).
--     Pregunta: ¿cuántos clientes no volvieron a comprar tras el corte?
SELECT
    snapshot_id,
    COUNT(*) AS clientes,
    COUNT(*) FILTER (WHERE gasto_siguiente_periodo = 0) AS sin_compra_futura,
    ROUND(AVG(gasto_siguiente_periodo), 2) AS target_medio,
    ROUND(AVG(gasto_historico), 2) AS gasto_historico_medio
FROM dataset_entrenamiento
GROUP BY snapshot_id;
