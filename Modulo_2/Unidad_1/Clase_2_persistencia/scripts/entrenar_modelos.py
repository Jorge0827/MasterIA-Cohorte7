"""
Entrena y GUARDA los dos modelos de la Clase 3 (FastAPI).

No es la clase de Machine Learning: los modelos son pequenos a proposito.
Lo que importa es el artefacto en disco:

    modelos/gasto_sklearn.joblib   -> Random Forest (scikit-learn clasico)
    modelos/gasto_mlp.joblib       -> red neuronal (MLP de scikit-learn)
    modelos/features_clientes.csv  -> lookup por cliente_id para la API
    modelos/meta.json              -> nombres de columnas y metricas

Uso:
    python scripts/entrenar_modelos.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features_modelo import (
    COLUMNA_TARGET,
    COLUMNAS_FEATURES,
    FECHA_CORTE,
    HORIZONTE_DIAS,
    construir_tabla_features,
    matriz_xy,
)

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_MODELOS = RAIZ / "modelos"
SEMILLA = 42
VERSION = "1.0.0"


def _guardar(modelo, nombre: str, metricas: dict) -> Path:
    """Serializa el objeto Python a un archivo .joblib en modelos/."""
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_MODELOS / nombre
    # joblib.dump escribe el objeto (arboles, pesos, scaler) en binario.
    # Es el formato habitual de scikit-learn. No es un CSV.
    joblib.dump(
        {
            "modelo": modelo,
            "features": COLUMNAS_FEATURES,
            "target": COLUMNA_TARGET,
            "version": VERSION,
            "metricas": metricas,
        },
        ruta,
    )
    return ruta


def main() -> None:
    print("Construyendo features desde data/raw/*.csv ...")
    tabla = construir_tabla_features()
    x, y = matriz_xy(tabla)
    print(f"  filas={len(tabla):,}  columnas={list(x.columns)}")

    # 80% entrena, 20% evalua. random_state fijo = mismo corte para todos.
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=SEMILLA
    )

    print("\n1) RandomForestRegressor (scikit-learn clasico)")
    t0 = time.perf_counter()
    bosque = RandomForestRegressor(
        n_estimators=40,   # 40 arboles: suficiente para la demo, rapido en clase
        max_depth=8,       # limita el tamano para no memorizar
        random_state=SEMILLA,
        n_jobs=-1,         # usa todos los nucleos del PC
    )
    bosque.fit(x_tr, y_tr)  # AQUI se "aprende". Despues solo se predice.
    pred_bosque = bosque.predict(x_te)
    met_bosque = {
        "mae": round(float(mean_absolute_error(y_te, pred_bosque)), 2),
        "r2": round(float(r2_score(y_te, pred_bosque)), 3),
        "segundos": round(time.perf_counter() - t0, 2),
    }
    ruta_bosque = _guardar(bosque, "gasto_sklearn.joblib", met_bosque)
    print(f"   MAE={met_bosque['mae']}  R2={met_bosque['r2']}  -> {ruta_bosque.name}")

    print("\n2) MLPRegressor (red neuronal de scikit-learn)")
    # Pipeline: primero escala (las redes sufren si gasto=8000 y compras=12
    # viajan en escalas distintas), luego la red.
    t0 = time.perf_counter()
    red = Pipeline(
        steps=[
            ("escala", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(16, 8),  # 2 capas ocultas, pequenas
                    max_iter=300,
                    random_state=SEMILLA,
                    early_stopping=True,
                ),
            ),
        ]
    )
    red.fit(x_tr, y_tr)
    pred_red = red.predict(x_te)
    met_red = {
        "mae": round(float(mean_absolute_error(y_te, pred_red)), 2),
        "r2": round(float(r2_score(y_te, pred_red)), 3),
        "segundos": round(time.perf_counter() - t0, 2),
    }
    ruta_red = _guardar(red, "gasto_mlp.joblib", met_red)
    print(f"   MAE={met_red['mae']}  R2={met_red['r2']}  -> {ruta_red.name}")

    # Lookup para la API: predecir con solo el cliente_id.
    lookup = tabla[["cliente_id", *COLUMNAS_FEATURES, COLUMNA_TARGET]]
    ruta_lookup = CARPETA_MODELOS / "features_clientes.csv"
    lookup.to_csv(ruta_lookup, index=False)
    print(f"\nLookup clientes -> {ruta_lookup.name} ({len(lookup):,} filas)")

    meta = {
        "version": VERSION,
        "fecha_corte": FECHA_CORTE.isoformat(),
        "horizonte_dias": HORIZONTE_DIAS,
        "features": COLUMNAS_FEATURES,
        "target": COLUMNA_TARGET,
        "sklearn": met_bosque,
        "mlp": met_red,
        "n_filas": int(len(tabla)),
    }
    ruta_meta = CARPETA_MODELOS / "meta.json"
    ruta_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadatos -> {ruta_meta.name}")
    print("\nListo. La API cargara estos archivos al arrancar.")


if __name__ == "__main__":
    main()
