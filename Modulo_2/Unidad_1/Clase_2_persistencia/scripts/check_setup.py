"""
Comprueba que todo esta listo para la clase.

Ejecuta este script antes de empezar (o pideselo a los estudiantes para que
verifiquen su instalacion). Revisa, en orden:

    1. Las librerias de Python
    2. El archivo .env
    3. Los CSV generados
    4. La conexion a PostgreSQL
    5. Las tablas y sus filas
    6. La conexion a MongoDB Atlas

No modifica nada: solo mira y informa.

Uso:
    python scripts/check_setup.py
"""

import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Contadores globales para el resumen final.
CORRECTOS = 0
PROBLEMAS: list[str] = []


def comprobar(descripcion: str, correcto: bool, detalle: str = "", critico: bool = True) -> bool:
    """Imprime una linea del checklist y lleva la cuenta de los problemas."""
    global CORRECTOS

    if correcto:
        CORRECTOS += 1
        marca = "[ OK ]"
    else:
        marca = "[FALTA]" if critico else "[AVISO]"
        if critico:
            PROBLEMAS.append(descripcion)

    print(f"  {marca} {descripcion}")
    # El detalle solo se muestra cuando hay algo que resolver: si todo va bien,
    # sugerir un comando de arreglo solo despista.
    if detalle and not correcto:
        print(f"         {detalle}")
    return correcto


def revisar_librerias() -> None:
    print("\n1. LIBRERIAS DE PYTHON")

    # Nombre para importar -> (nombre que se instala, nombre para pedir la version)
    # Los tres pueden ser distintos: se importa "dotenv", se instala
    # "python-dotenv" y la version se consulta como "python-dotenv".
    librerias = {
        "pandas": ("pandas", "pandas"),
        "numpy": ("numpy", "numpy"),
        "sqlalchemy": ("SQLAlchemy", "SQLAlchemy"),
        "psycopg": ("psycopg[binary]", "psycopg"),
        "pymongo": ("pymongo", "pymongo"),
        "dotenv": ("python-dotenv", "python-dotenv"),
    }

    for modulo, (paquete, distribucion) in librerias.items():
        try:
            __import__(modulo)
        except ImportError:
            comprobar(paquete, False, f"Instala con: pip install {paquete}")
            continue

        # No todas las librerias exponen __version__, asi que preguntamos a pip
        # (importlib.metadata lee los metadatos de la instalacion).
        try:
            version = metadata.version(distribucion)
        except metadata.PackageNotFoundError:
            version = "instalada"

        comprobar(f"{paquete:<16} {version}", True)

    print(f"\n  Python {sys.version.split()[0]} en {Path(sys.executable).parent.name}")


def revisar_env() -> bool:
    print("\n2. ARCHIVO .env")

    raiz = Path(__file__).resolve().parents[1]
    if not comprobar(
        "El archivo .env existe",
        (raiz / ".env").exists(),
        "Crealo con: Copy-Item .env.example .env",
    ):
        return False

    try:
        from src.config import resumen_configuracion

        comprobar("Las variables se leen correctamente", True)
        for linea in resumen_configuracion().splitlines():
            print(f"         {linea}")
        return True
    except Exception as error:
        comprobar("Las variables se leen correctamente", False, str(error))
        return False


def revisar_csv() -> None:
    print("\n3. DATOS CSV")

    from src.config import CARPETA_DATOS

    esperado = {"clientes.csv": 10_000, "productos.csv": 2_000, "ventas.csv": 200_000}

    for archivo, filas_esperadas in esperado.items():
        ruta = CARPETA_DATOS / archivo
        if not ruta.exists():
            comprobar(archivo, False, "Genera los CSV con: python scripts/generate_dataset.py")
            continue

        # Contamos lineas sin cargar el archivo en memoria, y restamos la cabecera.
        with ruta.open(encoding="utf-8") as fichero:
            filas = sum(1 for _ in fichero) - 1

        comprobar(f"{archivo:<15} {filas:>7,} filas", filas == filas_esperadas,
                  "" if filas == filas_esperadas else f"Se esperaban {filas_esperadas:,}")


def revisar_postgres() -> bool:
    print("\n4. CONEXION A POSTGRESQL")

    from src.db import ErrorDeConexion, probar_conexion

    try:
        version = probar_conexion()
        # Nos quedamos con la parte corta: "PostgreSQL 17.4"
        comprobar(" ".join(version.split()[:2]), True)
        return True
    except ErrorDeConexion as error:
        comprobar("Conexion a PostgreSQL", False, str(error).replace("\n", "\n         "))
        return False
    except Exception as error:
        comprobar("Conexion a PostgreSQL", False, str(error))
        return False


def revisar_tablas() -> None:
    print("\n5. TABLAS Y DATOS EN POSTGRESQL")

    from src.carga import contar_filas
    from src.db import get_engine
    from src.schema import listar_tablas

    tablas_esperadas = ["auditoria", "clientes", "predicciones", "productos", "ventas"]
    existentes = listar_tablas(get_engine())

    faltan = [t for t in tablas_esperadas if t not in existentes]
    if not comprobar(
        f"Las 5 tablas existen ({len(existentes)} encontradas)",
        not faltan,
        "" if not faltan else f"Faltan {faltan}. Ejecuta: python scripts/create_schema.py",
    ):
        return

    # Las tres primeras deben tener datos; predicciones y auditoria pueden estar vacias.
    for tabla, minimo in [("clientes", 10_000), ("productos", 2_000), ("ventas", 200_000)]:
        filas = contar_filas(tabla)
        comprobar(
            f"{tabla:<13} {filas:>7,} filas",
            filas >= minimo,
            "" if filas >= minimo else "Carga los datos con: python scripts/load_data.py",
        )

    for tabla in ["predicciones", "auditoria"]:
        print(f"         {tabla:<13} {contar_filas(tabla):>7,} filas (se llenan en clase)")


def revisar_mongo() -> None:
    print("\n6. CONEXION A MONGODB ATLAS")

    from src import mongo_db

    try:
        mongo_db.probar_conexion()
        comprobar("Conectado a MongoDB Atlas", True)
        print(f"         Documentos en model_metadata: {mongo_db.contar_documentos()}")
    except RuntimeError as error:
        # No es critico: la parte de PostgreSQL de la clase funciona sin Mongo.
        comprobar(
            "Conexion a MongoDB Atlas",
            False,
            str(error).splitlines()[0],
            critico=False,
        )
        print("         El bloque 9 del notebook no se podra ejecutar,")
        print("         pero el resto de la clase funciona con normalidad.")


def main() -> None:
    print("=" * 62)
    print("COMPROBACION DEL ENTORNO - Clase de Persistencia y Bases de Datos")
    print("=" * 62)

    revisar_librerias()

    # Si el .env no se puede leer, no tiene sentido seguir con la base de datos.
    if revisar_env():
        revisar_csv()
        if revisar_postgres():
            revisar_tablas()
        revisar_mongo()

    print("\n" + "=" * 62)
    if PROBLEMAS:
        print(f"HAY {len(PROBLEMAS)} PUNTO(S) QUE RESOLVER:")
        for problema in PROBLEMAS:
            print(f"  - {problema}")
        print("\nRevisa los mensajes de arriba o consulta docs/INSTALACION.md")
        sys.exit(1)

    print(f"TODO LISTO ({CORRECTOS} comprobaciones correctas). Puedes empezar la clase.")
    print("=" * 62)


if __name__ == "__main__":
    main()
