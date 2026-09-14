-- =============================================================================
-- Consultas de ejemplo sobre la base de datos "ventas_ia".
--
-- Pensadas para ejecutarlas en pgAdmin o psql durante la clase, y para que los
-- estudiantes tengan un punto de partida en los ejercicios.
--
--   psql -U postgres -d ventas_ia -f sql/consultas_ejemplo.sql
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Comprobar que los datos estan cargados
-- -----------------------------------------------------------------------------
SELECT 'clientes'  AS tabla, COUNT(*) AS filas FROM clientes
UNION ALL
SELECT 'productos' AS tabla, COUNT(*) AS filas FROM productos
UNION ALL
SELECT 'ventas'    AS tabla, COUNT(*) AS filas FROM ventas;


-- -----------------------------------------------------------------------------
-- 2. Resumen general de la facturacion
--    Una sola fila calculada sobre 200.000: el servidor hace el trabajo.
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)              AS num_ventas,
    SUM(total)            AS facturacion_total,
    ROUND(AVG(total), 2)  AS ticket_medio,
    MIN(fecha_venta)      AS primera_venta,
    MAX(fecha_venta)      AS ultima_venta
FROM ventas;


-- -----------------------------------------------------------------------------
-- 3. Clientes por pais y segmento (GROUP BY con dos columnas)
-- -----------------------------------------------------------------------------
SELECT
    pais,
    segmento,
    COUNT(*) AS num_clientes
FROM clientes
GROUP BY pais, segmento
ORDER BY pais, num_clientes DESC;


-- -----------------------------------------------------------------------------
-- 4. JOIN de tres tablas: el detalle de las ventas mas grandes
-- -----------------------------------------------------------------------------
SELECT
    v.venta_id,
    v.fecha_venta,
    c.nombre    AS cliente,
    c.pais,
    p.nombre    AS producto,
    p.categoria,
    v.cantidad,
    v.total
FROM ventas v
JOIN clientes  c ON v.cliente_id  = c.cliente_id
JOIN productos p ON v.producto_id = p.producto_id
ORDER BY v.total DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 5. Top 10 de clientes por facturacion
--    WHERE filtra filas y HAVING filtra grupos. Es la confusion mas habitual.
-- -----------------------------------------------------------------------------
SELECT
    c.cliente_id,
    c.nombre,
    c.ciudad,
    c.segmento,
    COUNT(*)     AS num_compras,
    SUM(v.total) AS gasto_total
FROM ventas v
JOIN clientes c ON v.cliente_id = c.cliente_id
GROUP BY c.cliente_id, c.nombre, c.ciudad, c.segmento
HAVING COUNT(*) > 5
ORDER BY gasto_total DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 6. Rentabilidad por categoria de producto
--    Margen = precio de venta - coste. Con NUMERIC el calculo es exacto.
-- -----------------------------------------------------------------------------
SELECT
    p.categoria,
    COUNT(*)                                  AS num_ventas,
    SUM(v.total)                              AS facturacion,
    ROUND(AVG(p.precio - p.coste), 2)         AS margen_medio_unitario,
    ROUND(AVG(v.total), 2)                    AS ticket_medio
FROM ventas v
JOIN productos p ON v.producto_id = p.producto_id
GROUP BY p.categoria
ORDER BY facturacion DESC;


-- -----------------------------------------------------------------------------
-- 7. Evolucion mensual de las ventas
--    DATE_TRUNC recorta la fecha al inicio del mes: sirve para agrupar por mes.
-- -----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('month', fecha_venta)::date AS mes,
    COUNT(*)                               AS num_ventas,
    SUM(total)                             AS facturacion
FROM ventas
GROUP BY mes
ORDER BY mes;


-- -----------------------------------------------------------------------------
-- 8. Calidad del modelo: predicho contra real
--    IS NOT NULL descarta las predicciones aun sin verificar.
-- -----------------------------------------------------------------------------
SELECT
    modelo,
    version,
    COUNT(*)                                            AS evaluadas,
    ROUND(AVG(ABS(valor_predicho - resultado_real)), 2) AS error_medio_absoluto,
    ROUND(AVG(confianza), 3)                            AS confianza_media,
    ROUND(AVG(tiempo_ejecucion_ms), 1)                  AS latencia_media_ms
FROM predicciones
WHERE resultado_real IS NOT NULL
GROUP BY modelo, version;


-- -----------------------------------------------------------------------------
-- 9. Auditoria: ultimos procesos ejecutados
-- -----------------------------------------------------------------------------
SELECT
    fecha_hora,
    proceso,
    operacion,
    tabla_afectada,
    registros_procesados,
    estado,
    duracion_segundos
FROM auditoria
ORDER BY auditoria_id DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 10. Auditoria: procesos que han fallado
--     Esta consulta es el germen del monitoreo de la siguiente clase.
-- -----------------------------------------------------------------------------
SELECT
    proceso,
    tabla_afectada,
    COUNT(*)                       AS ejecuciones_fallidas,
    MAX(fecha_hora)                AS ultimo_fallo,
    MAX(mensaje)                   AS ultimo_mensaje
FROM auditoria
WHERE estado = 'error'
GROUP BY proceso, tabla_afectada
ORDER BY ejecuciones_fallidas DESC;
