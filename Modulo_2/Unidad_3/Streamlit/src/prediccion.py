"""
Un solo sitio para armar la fila y llamar a predict().

Lo usan el notebook (sin interfaz) y app.py (con interfaz).
Así no copiamos el DataFrame de 1 fila en tres sitios.
"""

from __future__ import annotations

import pandas as pd

from src.features_modelo import COLUMNAS_FEATURES


def fila_a_dataframe(
    num_compras: int,
    gasto_historico: float,
    ticket_medio: float,
    cantidad_total: int,
) -> pd.DataFrame:
    """
    predict() exige una matriz (1 fila × 4 columnas), no un dict suelto.
    columns=COLUMNAS_FEATURES fija el ORDEN que vio el modelo al entrenar.
    """
    return pd.DataFrame(
        [
            {
                "num_compras": num_compras,
                "gasto_historico": gasto_historico,
                "ticket_medio": ticket_medio,
                "cantidad_total": cantidad_total,
            }
        ],
        columns=COLUMNAS_FEATURES,
    )


def predecir(paquete: dict, **cifras) -> float:
    """
    paquete["modelo"] es el RandomForest o el Pipeline(MLP).
    .predict(x) devuelve un array numpy; [0] es el único número.
    """
    x = fila_a_dataframe(**cifras)
    valor = float(paquete["modelo"].predict(x)[0])
    return round(valor, 2)
