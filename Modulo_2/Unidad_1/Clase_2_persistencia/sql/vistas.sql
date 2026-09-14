-- =============================================================================
-- Vistas de analisis de negocio (Clase 4).
--
-- Equivalente SQL de src/vistas.py.
--
-- Una VISTA no guarda filas: guarda un SELECT. Cada vez que haces
--     SELECT * FROM v_facturacion_mensual;
-- PostgreSQL vuelve a ejecutar la consulta sobre `ventas`.
--
--   psql -U postgres -d ventas_ia -f sql/vistas.sql
--
-- Normalmente NO hace falta: lo crea src.vistas.crear_vistas() desde el notebook.
-- =============================================================================


-- 1. KPI globales: 200.000 ventas -> 1 fila
CREATE OR REPLACE VIEW v_resumen_ventas AS
SELECT
    COUNT(*)                   AS num_ventas,
    COUNT(DISTINCT cliente_id) AS clientes_con_compra,
    COUNT(DISTINCT producto_id) AS productos_vendidos,
    SUM(total)                 AS facturacion_total,
    ROUND(AVG(total), 2)       AS ticket_medio,
    MIN(fecha_venta)           AS primera_venta,
    MAX(fecha_venta)           AS ultima_venta
FROM ventas;


-- 2. Serie mensual. DATE_TRUNC recorta al dia 1 del mes.
CREATE OR REPLACE VIEW v_facturacion_mensual AS
SELECT
    DATE_TRUNC('month', fecha_venta)::date AS mes,
    COUNT(*)                               AS num_ventas,
    COUNT(DISTINCT cliente_id)             AS clientes_activos,
    SUM(total)                             AS facturacion,
    ROUND(AVG(total), 2)                   AS ticket_medio
FROM ventas
GROUP BY DATE_TRUNC('month', fecha_venta);


-- 3. Rentabilidad por categoria (hecho + dimension)
CREATE OR REPLACE VIEW v_margen_categoria AS
SELECT
    p.categoria,
    COUNT(*)                            AS num_ventas,
    SUM(v.cantidad)                     AS unidades,
    SUM(v.total)                        AS facturacion,
    SUM(v.cantidad * p.coste)           AS coste_total,
    SUM(v.total - v.cantidad * p.coste) AS margen_total,
    ROUND(AVG(v.total - v.cantidad * p.coste), 2) AS margen_medio
FROM ventas v
JOIN productos p ON v.producto_id = p.producto_id
GROUP BY p.categoria;


-- 4. Rendimiento por canal
CREATE OR REPLACE VIEW v_rendimiento_canal AS
SELECT
    canal,
    COUNT(*)                   AS num_ventas,
    SUM(total)                 AS facturacion,
    ROUND(AVG(total), 2)       AS ticket_medio,
    COUNT(DISTINCT cliente_id) AS clientes_distintos
FROM ventas
GROUP BY canal;


-- 5. Ranking de clientes (window function RANK)
--    ranking_global: puesto en toda la empresa
--    ranking_en_pais: puesto solo dentro de su pais (PARTITION BY)
CREATE OR REPLACE VIEW v_ranking_clientes AS
SELECT
    c.cliente_id,
    c.nombre,
    c.pais,
    c.segmento,
    COUNT(*)     AS num_compras,
    SUM(v.total) AS gasto_total,
    RANK() OVER (ORDER BY SUM(v.total) DESC) AS ranking_global,
    RANK() OVER (
        PARTITION BY c.pais
        ORDER BY SUM(v.total) DESC
    ) AS ranking_en_pais
FROM ventas v
JOIN clientes c ON v.cliente_id = c.cliente_id
GROUP BY c.cliente_id, c.nombre, c.pais, c.segmento;
