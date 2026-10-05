"""
Features para el modelo de gasto (Unidad 2, Clase 3).

No hace falta PostgreSQL. Leemos los CSV de la Unidad 1 y aplicamos la
misma idea del snapshot de la Clase 4 de SQL:

    pasado  (fecha <= fecha_corte)  ->  lo que el modelo PUEDE ver
    futuro  (90 dias despues)       ->  el target (lo que queremos acertar)

Asi el notebook de FastAPI funciona aunque no hayas creado el snapshot SQL.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import CARPETA_DATOS

FECHA_CORTE = date(2025, 6, 30)
HORIZONTE_DIAS = 90

# Columnas que ENTRAN al modelo. El orden importa: joblib espera el mismo.
COLUMNAS_FEATURES = [
    "num_compras",
    "gasto_historico",
    "ticket_medio",
    "cantidad_total",
]

COLUMNA_TARGET = "gasto_siguiente_periodo"


def construir_tabla_features(
    fecha_corte: date = FECHA_CORTE,
    horizonte_dias: int = HORIZONTE_DIAS,
) -> pd.DataFrame:
    """
    Devuelve una fila por cliente que haya comprado ANTES del corte.

    Columnas: cliente_id + features + target.
    """
    ventas = pd.read_csv(CARPETA_DATOS / "ventas.csv", parse_dates=["fecha_venta"])
    corte = pd.Timestamp(fecha_corte)
    fin_futuro = corte + pd.Timedelta(days=horizonte_dias)

    # Pasado = historico que el modelo tiene derecho a ver.
    pasado = ventas.loc[ventas["fecha_venta"] <= corte]
    # Futuro = lo que ocurrio despues (el target). Si no compro, sera 0.
    futuro = ventas.loc[
        (ventas["fecha_venta"] > corte) & (ventas["fecha_venta"] <= fin_futuro)
    ]

    features = (
        pasado.groupby("cliente_id", as_index=False)
        .agg(
            num_compras=("venta_id", "count"),
            gasto_historico=("total", "sum"),
            ticket_medio=("total", "mean"),
            cantidad_total=("cantidad", "sum"),
        )
    )
    features["ticket_medio"] = features["ticket_medio"].round(2)
    features["gasto_historico"] = features["gasto_historico"].round(2)

    target = (
        futuro.groupby("cliente_id", as_index=False)["total"]
        .sum()
        .rename(columns={"total": COLUMNA_TARGET})
    )
    tabla = features.merge(target, on="cliente_id", how="left")
    tabla[COLUMNA_TARGET] = tabla[COLUMNA_TARGET].fillna(0).round(2)
    return tabla


def matriz_xy(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (lo que ve el modelo) e y (lo que intenta acertar)."""
    return tabla[COLUMNAS_FEATURES], tabla[COLUMNA_TARGET]
