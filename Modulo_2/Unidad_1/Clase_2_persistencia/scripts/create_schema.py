"""
Crea la base de datos y las 5 tablas del proyecto en PostgreSQL.

Es idempotente: se puede ejecutar tantas veces como quieras. Si la base de datos
o las tablas ya existen, no las duplica ni borra nada.

Uso:
    python scripts/create_schema.py
    python scripts/create_schema.py --reset    # borra las tablas y las vuelve a crear
"""

import argparse
import sys
from pathlib import Path

# Los scripts se ejecutan desde la carpeta scripts/, pero importan codigo de src/.
# Anadimos la raiz del proyecto a la lista de rutas donde Python busca modulos.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import POSTGRES_DB, resumen_configuracion  # noqa: E402
from src.db import ErrorDeConexion, crear_base_de_datos_si_no_existe, get_engine  # noqa: E402
from src.schema import borrar_tablas, crear_tablas  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea el esquema en PostgreSQL.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra las tablas existentes antes de crearlas (se pierden los datos)",
    )
    args = parser.parse_args()

    print("Configuracion detectada:")
    print(resumen_configuracion())
    print()

    # Paso 1: asegurar que existe la base de datos.
    # Capturamos el error de conexion para mostrar un mensaje claro en lugar de
    # un traceback de 50 lineas.
    try:
        creada = crear_base_de_datos_si_no_existe()
    except ErrorDeConexion as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    print(f"Base de datos '{POSTGRES_DB}' {'creada' if creada else 'ya existia'}.")

    engine = get_engine()

    # Paso 2 (opcional): limpiar el esquema anterior.
    if args.reset:
        print("Borrando tablas anteriores (--reset)...")
        borrar_tablas(engine)

    # Paso 3: crear las tablas.
    tablas = crear_tablas(engine)
    print("\nTablas disponibles en la base de datos:")
    for tabla in tablas:
        print(f"  - {tabla}")


if __name__ == "__main__":
    main()