"""
Vistas de PostgreSQL para el analisis de negocio (Clase 4).

Una VISTA es un SELECT guardado con nombre. No copia los datos: cada vez que
consultas la vista, PostgreSQL vuelve a ejecutar la consulta sobre las tablas
reales. Por eso siempre ves el dato actualizado.

    tabla  = datos persistidos en disco
    vista  = receta (la consulta), se recalcula al usarla
    vista materializada = receta + resultado cacheado (hay que REFRESH)

Por que existen:

1. Reutilizar un JOIN / agregacion compleja sin reescribirla.
2. Dar a un analista o a un notebook una "tabla" ya lista.
3. Ocultar columnas sensibles (la vista no incluye email, por ejemplo).
4. Separar el modelo fisico (tablas) del modelo que usa el negocio.

Estas definiciones son el equivalente Python de sql/vistas.sql.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.db import get_engine


# -----------------------------------------------------------------------------
# Definiciones: el SQL que hay detras de cada vista.
# CREATE OR REPLACE VIEW no admite un ";" al final de la consulta interna.
# -----------------------------------------------------------------------------

DEFINICIONES_VISTAS: dict[str, str] = {
    # KPI globales: 200.000 ventas resumidas en UNA fila.
    "v_resumen_ventas": """
        SELECT
            COUNT(*)                         AS num_ventas,
            COUNT(DISTINCT cliente_id)       AS clientes_con_compra,
            COUNT(DISTINCT producto_id)      AS productos_vendidos,
            SUM(total)                       AS facturacion_total,
            ROUND(AVG(total), 2)             AS ticket_medio,
            MIN(fecha_venta)                 AS primera_venta,
            MAX(fecha_venta)                 AS ultima_venta
        FROM ventas
    """,
    # Serie temporal mensual. DATE_TRUNC recorta la fecha al dia 1 del mes
    # para poder hacer GROUP BY mes sin inventarnos una columna extra.
    "v_facturacion_mensual": """
        SELECT
            DATE_TRUNC('month', fecha_venta)::date AS mes,
            COUNT(*)                               AS num_ventas,
            COUNT(DISTINCT cliente_id)             AS clientes_activos,
            SUM(total)                             AS facturacion,
            ROUND(AVG(total), 2)                   AS ticket_medio
        FROM ventas
        GROUP BY DATE_TRUNC('month', fecha_venta)
    """,
    # Margen por categoria: cruzamos hechos (ventas) con dimension (productos).
    # El margen usa el coste del catalogo, no el precio_unitario de la venta,
    # para que el estudiante vea que "precio de tarifa" y "precio cobrado"
    # no son siempre lo mismo (hay descuento).
    "v_margen_categoria": """
        SELECT
            p.categoria,
            COUNT(*)                          AS num_ventas,
            SUM(v.cantidad)                   AS unidades,
            SUM(v.total)                      AS facturacion,
            SUM(v.cantidad * p.coste)         AS coste_total,
            SUM(v.total - v.cantidad * p.coste) AS margen_total,
            ROUND(
                AVG(v.total - v.cantidad * p.coste),
                2
            )                                 AS margen_medio
        FROM ventas v
        JOIN productos p ON v.producto_id = p.producto_id
        GROUP BY p.categoria
    """,
    # Rendimiento por canal (web, app, tienda, telefono).
    "v_rendimiento_canal": """
        SELECT
            canal,
            COUNT(*)               AS num_ventas,
            SUM(total)             AS facturacion,
            ROUND(AVG(total), 2)   AS ticket_medio,
            COUNT(DISTINCT cliente_id) AS clientes_distintos
        FROM ventas
        GROUP BY canal
    """,
    # Ranking de clientes por gasto. Ya usa una WINDOW FUNCTION:
    # RANK() OVER (...) no colapsa filas, anade el puesto a cada cliente.
    "v_ranking_clientes": """
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
        GROUP BY c.cliente_id, c.nombre, c.pais, c.segmento
    """,
}


def crear_vistas() -> list[str]:
    """
    Crea (o reemplaza) todas las vistas de la Clase 4.

    CREATE OR REPLACE VIEW es idempotente en el sentido de que puedes
    ejecutarlo muchas veces. OJO: si cambias el numero o el orden de
    columnas, PostgreSQL a veces exige DROP VIEW primero.
    """
    engine = get_engine()
    creadas: list[str] = []

    # begin() abre transaccion: o se crean todas o no se crea ninguna.
    with engine.begin() as conexion:
        for nombre, consulta in DEFINICIONES_VISTAS.items():
            # El nombre de la vista lo controlamos nosotros (no viene de un
            # usuario), por eso es seguro interpolarlo en el SQL.
            sql = f"CREATE OR REPLACE VIEW {nombre} AS {consulta}"
            conexion.execute(text(sql))
            creadas.append(nombre)

    return creadas


def listar_vistas() -> pd.DataFrame:
    """
    Lista las vistas del esquema public.

    information_schema.views es el catalogo de PostgreSQL: la base de datos
    se describe a si misma. Es el equivalente a listar tablas, pero para vistas.
    """
    engine = get_engine()
    sql = """
        SELECT
            table_name   AS vista,
            view_definition AS definicion
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY table_name
    """
    with engine.connect() as conexion:
        return pd.read_sql_query(text(sql), conexion)


def consultar_vista(nombre: str, limite: int | None = 10) -> pd.DataFrame:
    """
    Consulta una vista como si fuera una tabla: SELECT * FROM nombre.

    limite=None devuelve todas las filas. En clase usamos un LIMIT para
    no inundar el notebook (v_ranking_clientes tiene ~10.000 filas).
    """
    if nombre not in DEFINICIONES_VISTAS:
        raise ValueError(
            f"No conozco la vista '{nombre}'. Opciones: {list(DEFINICIONES_VISTAS)}"
        )

    # LIMIT no acepta parametro con nombre en todos los drivers de forma
    # comoda; el limite es un entero nuestro, no texto de un usuario.
    sql = f"SELECT * FROM {nombre}"
    if limite is not None:
        sql += f" LIMIT {int(limite)}"

    engine = get_engine()
    with engine.connect() as conexion:
        return pd.read_sql_query(text(sql), conexion)


def borrar_vistas() -> None:
    """
    Borra las vistas de la Clase 4.

    CASCADE no es necesario aqui: ninguna otra vista depende de estas.
    IF EXISTS evita un error si todavia no se habian creado.
    """
    engine = get_engine()
    with engine.begin() as conexion:
        for nombre in DEFINICIONES_VISTAS:
            conexion.execute(text(f"DROP VIEW IF EXISTS {nombre}"))