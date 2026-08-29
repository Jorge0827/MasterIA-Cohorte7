"""
Carga los CSV de data/raw/ en las tablas de PostgreSQL.

Requisitos previos:
    1) python scripts/generate_dataset.py   (crear los CSV)
    2) python scripts/create_schema.py      (crear las tablas)

Uso:
    python scripts/load_data.py
    python scripts/load_data.py --sin-vaciar   # anade filas sin borrar las anteriores
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auditoria import ultimas_auditorias  # noqa: E402
from src.carga import cargar_todo, contar_filas  # noqa: E402
from src.db import ErrorDeConexion  # noqa: E402
from src.schema import ORDEN_CARGA  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga los CSV en PostgreSQL.")
    parser.add_argument(
        "--sin-vaciar",
        action="store_true",
        help="No vacia las tablas antes de insertar",
    )
    parser.add_argument(
        "--metodo",
        choices=["copy", "pandas"],
        default="copy",
        help="copy = carga masiva de PostgreSQL (rapido); pandas = to_sql (didactico)",
    )
    args = parser.parse_args()

    inicio = time.perf_counter()

    try:
        cargar_todo(vaciar_antes=not args.sin_vaciar, metodo=args.metodo)
    except ErrorDeConexion as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    print("\nFilas guardadas en PostgreSQL:")
    for tabla in ORDEN_CARGA:
        print(f"  {tabla:<12} {contar_filas(tabla):>8,}")

    print(f"\nTiempo total: {time.perf_counter() - inicio:.1f} s")

    # La carga se audita automaticamente: lo comprobamos aqui mismo.
    print("\nRegistros de auditoria generados:")
    columnas = ["proceso", "tabla_afectada", "registros_procesados", "estado", "duracion_segundos"]
    print(ultimas_auditorias(len(ORDEN_CARGA))[columnas].to_string(index=False))


if __name__ == "__main__":
    main()