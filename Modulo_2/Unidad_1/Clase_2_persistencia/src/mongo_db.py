"""
MongoDB Atlas: metadatos de modelos.

Por que otra base de datos si ya tenemos PostgreSQL?

Los datos de ventas son perfectos para SQL: siempre tienen las mismas columnas y
necesitamos integridad y JOINs. Pero los metadatos de un modelo NO son asi:

    - un modelo de arboles guarda "max_depth" y "n_estimators"
    - una red neuronal guarda "capas", "learning_rate", "dropout"...
    - manana probamos otro algoritmo con hiperparametros que hoy no existen

En SQL eso significaria cambiar la tabla cada vez (ALTER TABLE) o llenarla de
columnas vacias. En MongoDB cada documento puede tener su propia forma. Eso es
el ESQUEMA FLEXIBLE, y es la razon de usar NoSQL aqui.

Vocabulario equivalente:

    PostgreSQL          MongoDB
    ----------          -------
    base de datos  ->   base de datos
    tabla          ->   coleccion
    fila           ->   documento (parecido a un dict de Python / JSON)
    columna        ->   campo

Importante: aqui guardamos METADATOS, nunca las ventas. Los datos transaccionales
se quedan en PostgreSQL.
"""

from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from src.config import COLECCION_METADATOS, MONGODB_DB, MONGODB_URI

# Igual que con el engine de SQLAlchemy, creamos un unico cliente y lo
# reutilizamos: abrir conexiones nuevas todo el tiempo es lento y caro.
_cliente: MongoClient | None = None


def get_cliente() -> MongoClient:
    """
    Devuelve el cliente de MongoDB Atlas.

    serverSelectionTimeoutMS=5000 evita que el programa se quede colgado un
    minuto si la URI es incorrecta o la IP no esta autorizada en Atlas:
    falla en 5 segundos con un error claro.
    """
    global _cliente

    if not MONGODB_URI:
        raise RuntimeError(
            "Falta MONGODB_URI en el archivo .env.\n"
            "Copiala desde MongoDB Atlas: Connect > Drivers > Python."
        )

    if _cliente is None:
        _cliente = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)

    return _cliente


def get_coleccion() -> Collection:
    """
    Devuelve la coleccion `model_metadata`.

    Detalle curioso de MongoDB: no hace falta crear la base ni la coleccion.
    Se crean solas en el momento de insertar el primer documento.
    """
    cliente = get_cliente()
    return cliente[MONGODB_DB][COLECCION_METADATOS]


def probar_conexion() -> str:
    """
    Comprueba que Atlas responde, con el comando 'ping'.

    Es el equivalente al SELECT version() de PostgreSQL: si esto funciona,
    la conexion y las credenciales son correctas.
    """
    try:
        cliente = get_cliente()
        cliente.admin.command("ping")
    except PyMongoError as error:
        raise RuntimeError(
            "No se ha podido conectar a MongoDB Atlas.\n"
            "Comprueba que:\n"
            "  1) MONGODB_URI en .env tiene el usuario y la password correctos.\n"
            "  2) Tu IP esta autorizada en Atlas (Network Access > IP Access List).\n"
            f"\nDetalle tecnico: {error}"
        ) from None

    return f"Conectado a MongoDB Atlas. Base: '{MONGODB_DB}', coleccion: '{COLECCION_METADATOS}'"


def guardar_metadatos_modelo(
    nombre: str,
    version: str,
    metricas: dict,
    hiperparametros: dict,
    configuracion_extra: dict | None = None,
) -> str:
    """
    Guarda un documento con los metadatos de un modelo y devuelve su _id.

    Fijate en que `metricas`, `hiperparametros` y `configuracion_extra` son
    diccionarios libres: cada modelo puede traer las claves que necesite. Eso
    seria imposible en una tabla SQL sin ir cambiando su estructura.
    """
    documento = {
        "nombre": nombre,
        "version": version,
        # Guardamos la fecha en UTC. Es la buena practica cuando el equipo o los
        # servidores estan en zonas horarias distintas.
        "fecha_entrenamiento": datetime.now(timezone.utc),
        "metricas": metricas,
        "hiperparametros": hiperparametros,
    }

    # Solo anadimos el campo si nos han pasado algo: no ensuciamos el documento
    # con claves vacias.
    if configuracion_extra:
        documento["configuracion"] = configuracion_extra

    resultado = get_coleccion().insert_one(documento)

    # MongoDB genera un identificador propio (_id de tipo ObjectId).
    # Lo convertimos a texto para poder imprimirlo o guardarlo comodamente.
    return str(resultado.inserted_id)


def buscar_modelo(nombre: str, version: str | None = None) -> dict | None:
    """
    Busca un modelo por nombre y, si se indica, por version.

    El "filtro" de MongoDB es un diccionario: {"nombre": "modelo_x"}.
    Es el equivalente del WHERE de SQL.
    """
    filtro: dict = {"nombre": nombre}
    if version:
        filtro["version"] = version

    # find_one devuelve un solo documento (o None si no encuentra nada).
    # Ordenamos por fecha descendente para quedarnos con el entrenamiento mas reciente.
    return get_coleccion().find_one(filtro, sort=[("fecha_entrenamiento", -1)])


def listar_modelos(limite: int = 20) -> list[dict]:
    """Devuelve los ultimos modelos registrados, del mas reciente al mas antiguo."""
    # find() devuelve un cursor (perezoso); list() lo recorre y materializa.
    cursor = get_coleccion().find().sort("fecha_entrenamiento", -1).limit(limite)
    return list(cursor)


def contar_documentos() -> int:
    """Cuantos documentos hay en la coleccion model_metadata."""
    # El filtro vacio {} significa "todos", igual que un SELECT count(*) sin WHERE.
    return get_coleccion().count_documents({})


def borrar_modelo(nombre: str, version: str) -> int:
    """
    Borra los documentos de un modelo y version concretos.

    Util para repetir el ejemplo en clase sin acumular duplicados.
    Devuelve cuantos documentos se han borrado.
    """
    resultado = get_coleccion().delete_many({"nombre": nombre, "version": version})
    return resultado.deleted_count
