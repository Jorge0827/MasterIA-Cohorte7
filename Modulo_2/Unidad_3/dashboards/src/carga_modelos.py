"""
Carga los .joblib y el CSV de fichas (lookup).

La app Streamlit llama a esto UNA vez (con @st.cache_resource).
El notebook lo llama a mano para ver el mismo gesto sin interfaz.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features_modelo import COLUMNAS_FEATURES

# streamlit/src/ → raíz = streamlit/
RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "modelos"


class ErrorModeloNoEncontrado(RuntimeError):
    """Se lanza si faltan los artefactos. En clase: ejecutar el script de entrenamiento."""


def _exigir(ruta: Path) -> Path:
    """Si el archivo no está, error claro (mejor que un FileNotFoundError críptico)."""
    if not ruta.exists():
        raise ErrorModeloNoEncontrado(
            f"No está {ruta.name}. En la carpeta streamlit/ ejecuta:\n"
            "  python scripts/entrenar_modelos.py"
        )
    return ruta


def cargar_paquete(nombre_archivo: str) -> dict:
    """
    joblib.load reconstruye el diccionario que guardó entrenar_modelos.py:

        {"modelo": objeto sklearn, "features": [...], "version": "1.0.0", ...}
    """
    ruta = _exigir(CARPETA / nombre_archivo)
    paquete = joblib.load(ruta)
    if paquete.get("features") != COLUMNAS_FEATURES:
        raise RuntimeError(
            f"{nombre_archivo} se entrenó con otras columnas: {paquete.get('features')}"
        )
    return paquete


def cargar_lookup() -> pd.DataFrame:
    """Tabla cliente_id → features. Permite elegir un cliente en un desplegable."""
    ruta = _exigir(CARPETA / "features_clientes.csv")
    return pd.read_csv(ruta).set_index("cliente_id")


def cargar_meta() -> dict:
    """Versión, fecha de corte, MAE. Texto; se puede abrir en el editor."""
    ruta = _exigir(CARPETA / "meta.json")
    return json.loads(ruta.read_text(encoding="utf-8"))


def features_de_cliente(lookup: pd.DataFrame, cliente_id: int) -> dict:
    """
    Saca las 4 cifras del pasado de un cliente.
    El target (gasto_siguiente_periodo) NO se devuelve: el modelo no lo puede ver.
    """
    fila = lookup.loc[cliente_id]
    return {
        "num_compras": int(fila["num_compras"]),
        "gasto_historico": float(fila["gasto_historico"]),
        "ticket_medio": float(fila["ticket_medio"]),
        "cantidad_total": int(fila["cantidad_total"]),
    }
