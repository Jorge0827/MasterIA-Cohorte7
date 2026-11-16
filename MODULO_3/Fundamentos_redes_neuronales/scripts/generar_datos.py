"""
Genera data/sensores_motores.csv.

Uso (desde la carpeta redes_neuronales/):
    python scripts/generar_datos.py
    python scripts/generar_datos.py --filas 20000

Este archivo no tiene lógica propia: solo LANZA generar_dataset() de
src/datos.py desde la terminal. Separar "la lógica" (src/) de "el lanzador"
(scripts/) es lo habitual en proyectos reales.
"""

from __future__ import annotations

import argparse   # Lee las opciones escritas en la terminal (--filas, --semilla).
import sys
from pathlib import Path

# Al ejecutar `python scripts/generar_datos.py`, Python solo busca módulos en
# la carpeta scripts/. Añadimos la carpeta padre (redes_neuronales/) a la lista
# de búsqueda sys.path para que `from src.datos import ...` funcione.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `# noqa: E402` le dice al revisor de estilo que el import va después de
# sys.path a propósito (normalmente los imports van arriba del todo).
from src.datos import CLASES, TARGET, generar_dataset, guardar_dataset  # noqa: E402


def main() -> None:
    """Lee las opciones de la terminal, genera el dataset y muestra un resumen."""
    # ArgumentParser define qué opciones acepta el script.
    parser = argparse.ArgumentParser(description="Genera el dataset de sensores de motores.")
    # add_argument(nombre, type, default, help):
    #   type    -> convierte el texto de la terminal a int.
    #   default -> valor si el usuario no escribe la opción.
    #   help    -> texto que sale con `python scripts/generar_datos.py --help`.
    parser.add_argument("--filas", type=int, default=6000, help="Número de lecturas a simular.")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla para reproducibilidad.")
    # parse_args() lee lo escrito; args.filas y args.semilla quedan disponibles.
    args = parser.parse_args()

    df = generar_dataset(n=args.filas, semilla=args.semilla)
    ruta = guardar_dataset(df)

    print(f"CSV guardado en: {ruta}")
    print(f"Filas: {len(df):,}")
    print("Reparto de clases:")
    # .map(CLASES) cambia 0/1/2 por normal/desgaste/falla.
    # value_counts(normalize=True) da la proporción de cada valor; .mul(100) la pasa a %.
    reparto = df[TARGET].map(CLASES).value_counts(normalize=True).mul(100).round(1)
    for clase, pct in reparto.items():
        print(f"  {clase:<9} {pct:5.1f} %")


# Este bloque solo se ejecuta si lanzas el archivo directamente
# (python scripts/generar_datos.py), no si otro archivo lo importa.
if __name__ == "__main__":
    main()
