"""
Conexion a PostgreSQL con SQLAlchemy.

Conceptos clave que usaremos toda la clase:

- ENGINE: es la "fabrica de conexiones". Se crea una sola vez y se reutiliza.
          No abre una conexion al crearse: mantiene un pool de conexiones.
- CONNECTION: una conexion concreta, que abrimos con `with engine.connect() as con:`
          y se cierra automaticamente al salir del bloque.
"""

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from src.config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_URL, POSTGRES_USER

# Guardamos el engine en una variable de modulo para no crear uno nuevo cada vez
# que llamemos a get_engine(). Esto se llama "patron singleton" y aqui lo usamos
# en su version mas simple posible.
_engine: Engine | None = None


def get_engine(echo: bool = False) -> Engine:
    """
    Devuelve el engine de SQLAlchemy conectado a PostgreSQL.

    echo=True hace que SQLAlchemy imprima el SQL que ejecuta: muy util en clase
    para ver que sentencias genera realmente.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            POSTGRES_URL,
            echo=echo,
            pool_pre_ping=True,  # comprueba que la conexion sigue viva antes de usarla
        )
    return _engine


class ErrorDeConexion(RuntimeError):
    """
    Error de conexion con un mensaje entendible.

    Los errores de SQLAlchemy son larguisimos y asustan. Cuando falla la conexion
    lanzamos este error con una explicacion corta y las posibles causas.
    """


def _explicar_error_conexion(error: Exception) -> ErrorDeConexion:
    """Traduce un error de conexion de PostgreSQL a un mensaje util."""
    detalle = str(error)

    if "password" in detalle.lower() or "autentificaci" in detalle.lower():
        causa = (
            f"La contrasena del usuario '{POSTGRES_USER}' no es correcta.\n"
            "Revisa POSTGRES_PASSWORD en tu archivo .env"
        )
    elif "could not connect" in detalle.lower() or "no se pudo conectar" in detalle.lower():
        causa = (
            f"No hay nadie escuchando en {POSTGRES_HOST}:{POSTGRES_PORT}.\n"
            "Comprueba que el servicio de PostgreSQL esta arrancado."
        )
    else:
        causa = "Revisa host, puerto, usuario y contrasena en tu archivo .env"

    return ErrorDeConexion(
        f"No se ha podido conectar a PostgreSQL en "
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}.\n\n{causa}"
    )


def crear_base_de_datos_si_no_existe() -> bool:
    """
    Crea la base de datos del proyecto (POSTGRES_DB) si todavia no existe.

    Detalle importante: no se puede hacer CREATE DATABASE estando conectado a la
    base que quieres crear. Por eso nos conectamos a la base de sistema
    'postgres' y ademas usamos AUTOCOMMIT, porque CREATE DATABASE no puede
    ejecutarse dentro de una transaccion.

    Devuelve True si la ha creado, False si ya existia.
    """
    # Reutilizamos la URL de conexion cambiando solo el nombre de la base.
    url_servidor = POSTGRES_URL.set(database="postgres")
    engine_servidor = create_engine(url_servidor, isolation_level="AUTOCOMMIT")

    try:
        with engine_servidor.connect() as conexion:
            existe = conexion.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :nombre"),
                {"nombre": POSTGRES_DB},
            ).scalar()

            if existe:
                return False

            # El nombre de una base de datos no se puede pasar como parametro
            # (:nombre), asi que lo insertamos en el texto. Es seguro porque el
            # valor viene de nuestro propio .env, no de un usuario externo.
            conexion.execute(text(f'CREATE DATABASE "{POSTGRES_DB}"'))
            return True
    except OperationalError as error:
        raise _explicar_error_conexion(error) from None


def probar_conexion() -> str:
    """
    Comprueba que podemos hablar con PostgreSQL y devuelve su version.

    Es el "hola mundo" de la persistencia: si esto funciona, el resto tambien.
    """
    engine = get_engine()
    try:
        with engine.connect() as conexion:
            # text() convierte una cadena en una sentencia SQL ejecutable.
            # scalar() devuelve el primer valor de la primera fila.
            version = conexion.execute(text("SELECT version()")).scalar()
    except OperationalError as error:
        raise _explicar_error_conexion(error) from None
    return version


def consultar(sql: str, parametros: dict | None = None) -> pd.DataFrame:
    """
    Ejecuta una consulta SELECT y devuelve el resultado como DataFrame de Pandas.

    Los parametros se pasan SIEMPRE en el diccionario, nunca concatenando texto:

        consultar("SELECT * FROM clientes WHERE pais = :pais", {"pais": "Espana"})

    Asi evitamos inyeccion SQL y ganamos legibilidad.
    """
    engine = get_engine()
    with engine.connect() as conexion:
        return pd.read_sql_query(text(sql), conexion, params=parametros)


def ejecutar(sql: str, parametros: dict | None = None) -> int:
    """
    Ejecuta una sentencia que MODIFICA datos (INSERT, UPDATE, DELETE) o el esquema.

    Usamos engine.begin() en lugar de engine.connect(): begin() abre una
    TRANSACCION y hace COMMIT automatico al terminar el bloque sin errores
    (o ROLLBACK si hay excepcion). Devuelve el numero de filas afectadas.
    """
    engine = get_engine()
    with engine.begin() as conexion:
        resultado = conexion.execute(text(sql), parametros or {})
        return resultado.rowcount