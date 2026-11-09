"""
Esto es el cerebro del tablero
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import iv
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.features_modelo import COLUMNA_TARGET, COLUMNAS_FEATURES

SEMILLA_OFICIAL = 42
TEST_SIZE_OFICAL = 0.2


def particion_oficial(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    tren, holdout = train_test_split(
        tabla,
        test_size=TEST_SIZE_OFICAL,
        random_state=SEMILLA_OFICIAL,
    )
    return tren, holdout


def predecir_lote(paquete: dict, tabla: pd.DataFrame) -> pd.Series:
    
    x = tabla.loc[:, COLUMNAS_FEATURES]
    valores = paquete["modelo"].predict(x)
    
    return pd.Series(valores, index=tabla.index, name="y_pred").round(2)


def metricas(y_real: pd.Series, y_pred: pd.Series) -> dict:
    
    real = y_real.to_numpy(dtype=float)
    pred = y_pred.to_numpy(dtype=float)
    residuo = pred - real
    return{
        "n": int(len(real)),
        "mae": round(float(mean_absolute_error(real, pred)), 2),
        "rmse": round(float(root_mean_squared_error(real, pred)), 2),
        "r2": round(float(r2_score(real, pred)), 3),
        "sesgo": round(float(np.mean(residuo)), 2)
    }
    

def construir_monitoreo(
    tabla: pd.DataFrame, 
    paquete_sklearn: dict, 
    paquete_mlp: dict,
    ) -> pd.DataFrame:
    
    pred_sk = predecir_lote(paquete_sklearn, tabla)
    pred_mlp = predecir_lote(paquete_mlp, tabla)
    y_real = tabla[COLUMNA_TARGET]
    
    salida = tabla.loc[:, ["cliente_id", *COLUMNAS_FEATURES, COLUMNA_TARGET]].copy()
    salida = salida.rename(columns={COLUMNA_TARGET: "y_real"})
    salida["pred_sklearn"] = pred_sk.to_numpy()
    salida["pred_mlp"] = pred_mlp.to_numpy()
    salida["error_sklearn"] = (salida["pred_sklearn"] - salida["y_real"]).round(2)
    salida["error_mlp"] = (salida["pred_mlp"] - salida["y_real"]).round(2)
    
    salida["abs_sklearn"] = salida["error_sklearn"].abs()
    salida["abs_mlp"] = salida["error_mlp"].abs()
    
    return salida.reset_index(drop=True) #1,2,3,4,5


def resumir_metricas(monitoreo: pd.DataFrame) -> dict:
    
    return {
        "sklearn": metricas(monitoreo["y_real"], monitoreo["pred_sklearn"]),
        "mlp": metricas(monitoreo["y_real"], monitoreo["pred_mlp"])
    } 
    

def peores_clientes(monitoreo: pd.DataFrame, motor: str = "sklearn", n: int = 10) -> pd.DataFrame:
    col = "abs_sklearn" if motor == "sklearn" else "abs_mlp"
    cols = [
        "cliente_id",
        "gasto_historico",
        "y_real",
        f"pred_{motor}",
        f"error_{motor}",
        col,
    ]
    
    return monitoreo.loc[:, cols].sort_values(col, ascending=False).head(n)


def error_por_tramo(
    monitoreo: pd.DataFrame,
    motor: str = "sklearn",
    col_corte: str = "gasto_historico"
) -> pd.DataFrame:
    
    col_abs = "abs_sklearn" if motor == "sklearn" else "abs_mlp"
    tramo = pd.qcut(
        monitoreo[col_corte],
        q=4,
        labels=["Q1 bajo", "Q2", "Q3", "Q4 alto"],
    )
    
    resumen = (
        monitoreo.assign(tramo=tramo)
        .groupby("tramo", observed=False)[col_abs]
        .mean()
        .round(2)
        .rename("mae")
        .to_frame()
    )
    
    resumen["n"] = monitoreo.groupby(tramo, observed=False).size()
    return resumen


def histograma_residual(
    monitoreo: pd.DataFrame,
    motor: str = "sklearn",
    n_bins: int = 20
) -> pd.DataFrame:
    
    col = "error_sklearn" if motor == "sklearn" else "abs_mlp"
    
    conteo = monitoreo[col].value_counts(bins=n_bins).sort_index()
    
    etiquetas = [f"{iv.left:.0f} a {iv.right:.0f}" for iv in conteo.index]
    
    return pd.DataFrame({"clientes": conteo.to_numpy()}, index=etiquetas)


    
    
    
    
    
    