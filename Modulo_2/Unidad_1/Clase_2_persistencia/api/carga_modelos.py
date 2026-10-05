"""
Carga los .joblib UNA VEZ cuando arranca el servidor.

No se reentrena en cada peticion. En produccion el archivo lo dejo el
pipeline de entrenamiento (nuestro scripts/entrenar_modelos.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features_modelo import COLUMNAS_FEATURES

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "modelos"


class ErrorModeloNoEncontrado(RuntimeError):
    """Se lanza si alguien arranca la API sin haber entrenado."""


def _exigir(ruta: Path) -> Path:
    if not ruta.exists():
        raise ErrorModeloNoEncontrado(
            f"No esta {ruta.name}. En la raiz del proyecto ejecuta:\n"
            "  python scripts/entrenar_modelos.py"
        )
    return ruta


def cargar_paquete(nombre_archivo: str) -> dict:
    """
    joblib.load lee el diccionario que guardo entrenar_modelos.py:

        {"modelo": objeto sklearn, "features": [...], "version": "1.0.0", ...}
    """
    ruta = _exigir(CARPETA / nombre_archivo)
    paquete = joblib.load(ruta)
    if paquete.get("features") != COLUMNAS_FEATURES:
        raise RuntimeError(
            f"{nombre_archivo} se entreno con otras columnas: {paquete.get('features')}"
        )
    return paquete


def cargar_lookup() -> pd.DataFrame:
    """Tabla cliente_id -> features. Permite predecir solo con el id."""
    ruta = _exigir(CARPETA / "features_clientes.csv")
    return pd.read_csv(ruta).set_index("cliente_id")


def cargar_meta() -> dict:
    ruta = _exigir(CARPETA / "meta.json")
    return json.loads(ruta.read_text(encoding="utf-8"))


def features_de_cliente(lookup: pd.DataFrame, cliente_id: int) -> dict:
    """
    Busca las 4 cifras de un cliente. Si no esta, KeyError
    (la ruta de la API lo convierte en HTTP 404).
    """
    fila = lookup.loc[cliente_id]
    return {
        "num_compras": int(fila["num_compras"]),
        "gasto_historico": float(fila["gasto_historico"]),
        "ticket_medio": float(fila["ticket_medio"]),
        "cantidad_total": int(fila["cantidad_total"]),
    }
