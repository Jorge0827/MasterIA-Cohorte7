"""
Historial de entrenamiento: congelar un dataset a partir del ETL (Clase 4).

Problema real que resuelve:

    Entrenar un modelo "sobre la tabla ventas" es una mala idea.
    Manana habra mas filas, alguien corregira un precio, o recargaras el CSV.
    El experimento de hoy ya no se puede repetir.

Solucion: un SNAPSHOT versionado.

    1. Se elige una fecha_corte (el "hoy" ficticio del modelo).
    2. Features = agregados de ventas ANTERIORES o iguales a esa fecha.
    3. Target   = gasto del cliente en los 90 dias POSTERIORES.
    4. Se guarda el SQL exacto, el numero de filas y un checksum.
    5. Se audita el proceso (reutilizamos src.auditoria de la Clase 2).

Flujo:

    PostgreSQL (ventas limpias)
        -> consulta de features (CTE + window functions)
        -> historial_entrenamiento  (metadatos del recorte)
        -> dataset_entrenamiento    (filas congeladas)
        -> auditoria                (quien, cuando, cuantas filas)
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import delete, insert, select, text

from src.auditoria import auditar
from src.db import get_engine
from src.schema import dataset_entrenamiento as tabla_dataset
from src.schema import historial_entrenamiento as tabla_historial

# Fecha por defecto: deja 6 meses de historico y 90 dias de futuro
# (las ventas del dataset llegan hasta 2025-12-31).
FECHA_CORTE_POR_DEFECTO = date(2025, 6, 30)
HORIZONTE_DIAS = 90

# Nombre y version por defecto del dataset de la clase.
NOMBRE_DATASET = "features_gasto_90d"
VERSION_POR_DEFECTO = "1.0.0"

# Esta consulta es el CORAZON del ETL de features.
# Vive aqui (y en sql/consultas_avanzadas.sql) para que el notebook y el
# script usen exactamente el mismo SQL.
#
# :fecha_corte y :horizonte_dias son PARAMETROS, nunca texto concatenado.
SQL_FEATURES = """
WITH
-- 1) Ventas que el modelo PUEDE ver: solo el pasado (y el dia de corte).
ventas_pasado AS (
    SELECT *
    FROM ventas
    WHERE fecha_venta <= :fecha_corte
),

-- 2) Target: gasto en la ventana futura (lo que queremos predecir).
--    Si el cliente no compra en esos 90 dias, mas abajo pondremos 0.
ventas_futuro AS (
    SELECT
        cliente_id,
        SUM(total) AS gasto_siguiente_periodo
    FROM ventas
    WHERE fecha_venta > :fecha_corte
      AND fecha_venta <= CAST(:fecha_corte AS date)
                        + (:horizonte_dias * INTERVAL '1 day')
    GROUP BY cliente_id
),

-- 3) Agregados por cliente sobre el pasado.
agg_cliente AS (
    SELECT
        cliente_id,
        COUNT(*)                              AS num_compras,
        SUM(total)                            AS gasto_historico,
        ROUND(AVG(total), 2)                  AS ticket_medio,
        SUM(cantidad)                         AS cantidad_total,
        CAST(:fecha_corte AS date) - MAX(fecha_venta)
                                              AS dias_desde_ultima_compra
    FROM ventas_pasado
    GROUP BY cliente_id
),

-- 4) Canal mas frecuente. ROW_NUMBER parte por cliente y ordena por
--    numero de compras: rn = 1 es el canal preferido.
canal_pref AS (
    SELECT cliente_id, canal
    FROM (
        SELECT
            cliente_id,
            canal,
            COUNT(*) AS n,
            ROW_NUMBER() OVER (
                PARTITION BY cliente_id
                ORDER BY COUNT(*) DESC
            ) AS rn
        FROM ventas_pasado
        GROUP BY cliente_id, canal
    ) t
    WHERE rn = 1
),

-- 5) Categoria donde mas ha gastado. Misma idea: window + filtro rn = 1.
cat_fav AS (
    SELECT cliente_id, categoria
    FROM (
        SELECT
            v.cliente_id,
            p.categoria,
            SUM(v.total) AS gasto,
            ROW_NUMBER() OVER (
                PARTITION BY v.cliente_id
                ORDER BY SUM(v.total) DESC
            ) AS rn
        FROM ventas_pasado v
        JOIN productos p ON p.producto_id = v.producto_id
        GROUP BY v.cliente_id, p.categoria
    ) t
    WHERE rn = 1
)

-- 6) Montaje final: una fila por cliente que haya comprado ANTES del corte.
SELECT
    a.cliente_id,
    c.segmento,
    c.pais,
    a.num_compras,
    a.gasto_historico,
    a.ticket_medio,
    a.cantidad_total,
    a.dias_desde_ultima_compra,
    canal_pref.canal     AS canal_preferido,
    cat_fav.categoria    AS categoria_favorita,
    COALESCE(f.gasto_siguiente_periodo, 0) AS gasto_siguiente_periodo,
    CAST(:fecha_corte AS date)             AS fecha_corte
FROM agg_cliente a
JOIN clientes c          ON a.cliente_id = c.cliente_id
LEFT JOIN canal_pref     ON a.cliente_id = canal_pref.cliente_id
LEFT JOIN cat_fav        ON a.cliente_id = cat_fav.cliente_id
LEFT JOIN ventas_futuro f ON a.cliente_id = f.cliente_id
"""


def _checksum_dataframe(datos: pd.DataFrame) -> str:
    """
    Calcula un SHA-256 estable del DataFrame.

    Ordenamos por cliente_id y usamos las mismas columnas siempre, para que
    dos ejecuciones identicas produzcan el mismo hash. Si el hash cambia,
    el dataset cambio (aunque el nombre y la version sean iguales).
    """
    # copy() evita tocar el DataFrame original del llamador.
    ordenado = datos.sort_values("cliente_id").reset_index(drop=True)
    # to_csv es texto determinista: mismas filas -> mismo string -> mismo hash.
    payload = ordenado.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def construir_features(
    fecha_corte: date | str = FECHA_CORTE_POR_DEFECTO,
    horizonte_dias: int = HORIZONTE_DIAS,
) -> pd.DataFrame:
    """
    Ejecuta el SQL de features y devuelve el DataFrame EN MEMORIA.

    Todavia NO guarda nada. Separar "calcular" de "persistir" es lo mismo
    que hicimos con simular_prediccion / registrar_prediccion en la Clase 2.
    """
    # Aceptamos str ('2025-06-30') o date para que el notebook sea comodo.
    if isinstance(fecha_corte, str):
        fecha_corte = date.fromisoformat(fecha_corte)

    engine = get_engine()
    with engine.connect() as conexion:
        return pd.read_sql_query(
            text(SQL_FEATURES),
            conexion,
            params={
                "fecha_corte": fecha_corte,
                "horizonte_dias": horizonte_dias,
            },
        )


def listar_snapshots() -> pd.DataFrame:
    """Devuelve el historial de snapshots, el mas reciente primero."""
    engine = get_engine()
    consulta = (
        select(tabla_historial)
        .order_by(tabla_historial.c.snapshot_id.desc())
    )
    with engine.connect() as conexion:
        return pd.read_sql_query(consulta, conexion)


def filas_del_snapshot(snapshot_id: int, limite: int | None = 10) -> pd.DataFrame:
    """Lee las filas congeladas de un snapshot concreto."""
    engine = get_engine()
    consulta = (
        select(tabla_dataset)
        .where(tabla_dataset.c.snapshot_id == snapshot_id)
        .order_by(tabla_dataset.c.cliente_id)
    )
    if limite is not None:
        consulta = consulta.limit(int(limite))
    with engine.connect() as conexion:
        return pd.read_sql_query(consulta, conexion)


def borrar_snapshot(snapshot_id: int) -> int:
    """
    Borra un snapshot y, por ON DELETE CASCADE, sus filas en
    dataset_entrenamiento. Devuelve cuantos metadatos se eliminaron (0 o 1).
    """
    engine = get_engine()
    with engine.begin() as conexion:
        resultado = conexion.execute(
            delete(tabla_historial).where(
                tabla_historial.c.snapshot_id == snapshot_id
            )
        )
        return resultado.rowcount


def crear_snapshot(
    fecha_corte: date | str = FECHA_CORTE_POR_DEFECTO,
    horizonte_dias: int = HORIZONTE_DIAS,
    nombre_dataset: str = NOMBRE_DATASET,
    version: str = VERSION_POR_DEFECTO,
    descripcion: str | None = None,
    reemplazar_si_existe: bool = True,
) -> int:
    """
    Congela un dataset de entrenamiento y lo deja auditado.

    Pasos:
        1. Calcular features (SELECT en PostgreSQL).
        2. Escribir metadatos en historial_entrenamiento.
        3. Escribir las filas en dataset_entrenamiento.
        4. Auditar el proceso.

    Devuelve el snapshot_id generado por PostgreSQL.
    """
    if isinstance(fecha_corte, str):
        fecha_corte = date.fromisoformat(fecha_corte)

    if descripcion is None:
        descripcion = (
            f"Features hasta {fecha_corte.isoformat()} "
            f"y target a {horizonte_dias} dias."
        )

    parametros = json.dumps(
        {
            "fecha_corte": fecha_corte.isoformat(),
            "horizonte_dias": horizonte_dias,
        },
        ensure_ascii=False,
    )

    # auditar() mide tiempo y registra exito/error en la tabla auditoria.
    with auditar(
        proceso="snapshot_entrenamiento",
        operacion="INSERT",
        tabla_afectada="dataset_entrenamiento",
    ) as info:
        features = construir_features(fecha_corte, horizonte_dias)
        if features.empty:
            raise RuntimeError(
                "El SQL de features no devolvio filas. "
                "Revisa fecha_corte y que existan ventas anteriores a esa fecha."
            )

        checksum = _checksum_dataframe(features)

        engine = get_engine()
        with engine.begin() as conexion:
            # Si ya existe ese nombre+version, lo sustituimos para que la
            # clase se pueda repetir sin ensuciar el historial.
            if reemplazar_si_existe:
                existente = conexion.execute(
                    select(tabla_historial.c.snapshot_id).where(
                        tabla_historial.c.nombre_dataset == nombre_dataset,
                        tabla_historial.c.version == version,
                    )
                ).scalar()
                if existente is not None:
                    conexion.execute(
                        delete(tabla_historial).where(
                            tabla_historial.c.snapshot_id == existente
                        )
                    )

            resultado = conexion.execute(
                insert(tabla_historial).values(
                    nombre_dataset=nombre_dataset,
                    version=version,
                    fecha_corte=fecha_corte,
                    consulta_sql=SQL_FEATURES.strip(),
                    n_filas=int(len(features)),
                    n_columnas=int(features.shape[1]),
                    descripcion=descripcion,
                    parametros=parametros,
                    checksum=checksum,
                    estado="activo",
                ).returning(tabla_historial.c.snapshot_id)
            )
            snapshot_id = int(resultado.scalar_one())

            # Anadimos snapshot_id a cada fila: es la FK hacia el historial.
            filas = features.copy()
            filas.insert(0, "snapshot_id", snapshot_id)

            # to_sql con method=None usa INSERT estandar. Son ~8-10k filas,
            # cabe de sobra. No usamos COPY aqui para mantener el codigo corto.
            filas.to_sql(
                "dataset_entrenamiento",
                conexion,
                if_exists="append",
                index=False,
            )

        info["registros"] = int(len(features))
        info["mensaje"] = (
            f"snapshot_id={snapshot_id} version={version} "
            f"checksum={checksum[:12]}..."
        )

    return snapshot_id


def resumen_snapshot(snapshot_id: int) -> str:
    """Texto corto para imprimir en el notebook despues de crear el snapshot."""
    engine = get_engine()
    with engine.connect() as conexion:
        fila = conexion.execute(
            select(tabla_historial).where(
                tabla_historial.c.snapshot_id == snapshot_id
            )
        ).mappings().first()

    if fila is None:
        return f"No existe el snapshot {snapshot_id}."

    return (
        f"Snapshot {fila['snapshot_id']}: {fila['nombre_dataset']} "
        f"v{fila['version']}\n"
        f"  fecha_corte : {fila['fecha_corte']}\n"
        f"  filas       : {fila['n_filas']}\n"
        f"  columnas    : {fila['n_columnas']}\n"
        f"  checksum    : {fila['checksum']}\n"
        f"  creado      : {fila['fecha_creacion']}"
    )


def horizonte_por_defecto() -> timedelta:
    """Expone el horizonte como timedelta por si el notebook lo quiere mostrar."""
    return timedelta(days=HORIZONTE_DIAS)