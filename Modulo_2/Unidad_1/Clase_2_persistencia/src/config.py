"""
Configuracion central del proyecto.

Aqui leemos UNA SOLA VEZ el archivo .env y dejamos disponibles:

- las rutas de carpetas del proyecto,
- la URL de conexion a PostgreSQL,
- los datos de conexion a MongoDB Atlas.

Regla de oro: las credenciales NUNCA se escriben en el codigo ni en el notebook.
Viven en el archivo .env, que esta excluido de git (ver .gitignore).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL

# -----------------------------------------------------------------------------
# Rutas del proyecto
# -----------------------------------------------------------------------------

# Este archivo esta en src/, asi que la raiz del proyecto es un nivel arriba.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ_PROYECTO / "data" / "raw"
CARPETA_SQL = RAIZ_PROYECTO / "sql"

# load_dotenv lee el archivo .env y mete sus valores en las variables de entorno
# del proceso, para poder leerlos despues con os.getenv().
load_dotenv(RAIZ_PROYECTO / ".env")


# -----------------------------------------------------------------------------
# Lectura de variables de entorno
# -----------------------------------------------------------------------------

def _leer_variable(nombre: str, valor_por_defecto: str | None = None) -> str:
    """
    Devuelve el valor de una variable de entorno.

    Si no existe y no hay valor por defecto, lanza un error CLARO. Es mucho mejor
    fallar aqui con un mensaje entendible que ver un error raro de conexion.
    """
    valor = os.getenv(nombre, valor_por_defecto)
    if valor is None or valor == "":
        raise RuntimeError(
            f"Falta la variable '{nombre}'. "
            "Copia .env.example como .env y rellena tus credenciales."
        )
    return valor


# --- PostgreSQL ---------------------------------------------------------------

POSTGRES_HOST = _leer_variable("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(_leer_variable("POSTGRES_PORT", "5432"))
POSTGRES_DB = _leer_variable("POSTGRES_DB", "ventas_ia")
POSTGRES_USER = _leer_variable("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = _leer_variable("POSTGRES_PASSWORD")

# SQLAlchemy necesita una "URL de conexion" con esta forma:
#   postgresql+psycopg://usuario:password@host:puerto/base_de_datos
#
# La construimos con URL.create en lugar de concatenar texto, porque asi
# SQLAlchemy escapa correctamente contrasenas con caracteres especiales (@, /, #...).
POSTGRES_URL = URL.create(
    drivername="postgresql+psycopg",  # dialecto postgresql + driver psycopg 3
    username=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
)


# --- MongoDB Atlas ------------------------------------------------------------

# Se leen de forma "perezosa": si alguien solo trabaja con PostgreSQL, no
# queremos obligarle a tener configurado MongoDB. Por eso usamos os.getenv
# directamente y validamos mas tarde, en src/mongo_db.py.
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "ml_metadata")
COLECCION_METADATOS = "model_metadata"


def resumen_configuracion() -> str:
    """
    Devuelve un resumen legible de la configuracion, SIN mostrar contrasenas.
    Util para comprobar en el notebook que el .env se ha leido bien.
    """
    mongo_configurado = "si" if MONGODB_URI else "no"
    return (
        f"PostgreSQL : {POSTGRES_USER}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}\n"
        f"MongoDB    : base '{MONGODB_DB}' (URI configurada: {mongo_configurado})\n"
        f"Datos CSV  : {CARPETA_DATOS}"
    )
