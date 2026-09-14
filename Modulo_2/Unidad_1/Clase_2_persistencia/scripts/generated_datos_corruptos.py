"""
Genera CSV "rotos" para la demo de calidad de datos (Clase 3).

NO toca los CSV oficiales de data/raw/. Crea copias pequenas y modificadas
en data/raw/corruptos/ para mostrar tres fallos tipicos:

  1. clientes_sin_email.csv      -> falta una columna obligatoria
  2. ventas_columna_extra.csv    -> aparece una columna nueva (schema drift)
  3. ventas_precio_texto.csv     -> un numero se convierte en texto

Uso:
    python scripts/generate_datos_corruptos.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import CARPETA_DATOS
from src.pipeline import CARPETA_CORRUPTOS


def main() -> None:
    clientes_oficial = CARPETA_DATOS / "clientes.csv"
    ventas_oficial = CARPETA_DATOS / "ventas.csv"

    if not clientes_oficial.exists() or not ventas_oficial.exists():
        raise FileNotFoundError(
            "Faltan los CSV oficiales. Ejecuta primero:\n"
            "  python scripts/generate_dataset.py"
        )

    CARPETA_CORRUPTOS.mkdir(parents=True, exist_ok=True)

    # Usamos pocas filas: la demo debe ser instantanea en clase.
    clientes = pd.read_csv(clientes_oficial, nrows=30)
    ventas = pd.read_csv(ventas_oficial, nrows=30)

    # 1) Falta la columna email: el contrato la marca como obligatoria.
    clientes.drop(columns=["email"]).to_csv(
        CARPETA_CORRUPTOS / "clientes_sin_email.csv",
        index=False,
    )

    # 2) Columna extra: alguien cambio el formato del archivo de origen.
    ventas_extra = ventas.copy()
    ventas_extra["promocion"] = "black_friday"
    ventas_extra.to_csv(CARPETA_CORRUPTOS / "ventas_columna_extra.csv", index=False)

    # 3) Tipo corrupto: un precio deja de ser numero.
    ventas_texto = ventas.copy()
    ventas_texto["precio_unitario"] = ventas_texto["precio_unitario"].astype(str)
    ventas_texto.loc[0, "precio_unitario"] = "abc"
    ventas_texto.to_csv(CARPETA_CORRUPTOS / "ventas_precio_texto.csv", index=False)

    print("CSV corruptos creados en", CARPETA_CORRUPTOS)
    for archivo in sorted(CARPETA_CORRUPTOS.glob("*.csv")):
        print(f"  - {archivo.name}")


if __name__ == "__main__":
    main()
