"""
Carga de datos: del CSV a PostgreSQL.

Este es el corazon de la clase. El recorrido es:

    archivo CSV  ->  Pandas (memoria)  ->  SQLAlchemy  ->  PostgreSQL (disco)

Idea clave: Pandas trabaja en MEMORIA. Cuando cierras Python, el DataFrame
desaparece. PostgreSQL trabaja en DISCO: los datos siguen ahi manana, y otras
personas o aplicaciones pueden consultarlos. Eso es la PERSISTENCIA.

Cada carga queda registrada en la tabla `auditoria` automaticamente.
"""

import io

import pandas as pd
from sqlalchemy import func, select, text

from src.auditoria import auditar
from src.config import CARPETA_DATOS
from src.db import get_engine
from src.schema import ORDEN_CARGA, metadata

# Que CSV va a que tabla, y que columnas hay que interpretar como fechas.
ARCHIVOS = {
    "clientes": {"csv": "clientes.csv", "fechas": ["fecha_registro"]},
    "productos": {"csv": "productos.csv", "fechas": []},
    "ventas": {"csv": "ventas.csv", "fechas": ["fecha_venta"]},
}

# Filas por lote al insertar. No mandamos las 200.000 filas de golpe:
# se envian por bloques para no agotar la memoria ni el limite de parametros
# que acepta PostgreSQL en una sola sentencia.
TAMANO_LOTE = 5_000


def leer_csv(nombre_tabla: str) -> pd.DataFrame:
    """
    Lee el CSV correspondiente a una tabla y lo devuelve como DataFrame.

    parse_dates convierte las columnas de fecha de texto a fecha real. Si no lo
    hicieramos, Pandas las trataria como cadenas de texto.
    """
    if nombre_tabla not in ARCHIVOS:
        raise ValueError(f"No conozco la tabla '{nombre_tabla}'. Opciones: {list(ARCHIVOS)}")

    configuracion = ARCHIVOS[nombre_tabla]
    ruta = CARPETA_DATOS / configuracion["csv"]

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}.\nGenera los datos con: python scripts/generate_dataset.py"
        )

    return pd.read_csv(ruta, parse_dates=configuracion["fechas"])


def vaciar_tabla(nombre_tabla: str) -> None:
    """
    Borra todas las filas de una tabla, dejando la tabla en pie.

    TRUNCATE es mucho mas rapido que DELETE porque no va fila por fila.
    CASCADE vacia tambien las tablas que dependen de esta por clave ajena
    (si vaciamos clientes, hay que vaciar ventas: no puede haber ventas
    apuntando a clientes que ya no existen).

    OJO en clase: vaciar `clientes` arrastra tambien `ventas` y `predicciones`,
    porque ambas apuntan a clientes. Por eso el orden natural es cargar primero
    los datos y registrar las predicciones despues.
    """
    engine = get_engine()
    with engine.begin() as conexion:
        conexion.execute(text(f"TRUNCATE TABLE {nombre_tabla} CASCADE"))


def contar_filas(nombre_tabla: str) -> int:
    """Devuelve cuantas filas hay guardadas en una tabla de PostgreSQL."""
    engine = get_engine()
    tabla = metadata.tables[nombre_tabla]

    # Equivalente a: SELECT count(*) FROM <tabla>
    with engine.connect() as conexion:
        return conexion.execute(select(func.count()).select_from(tabla)).scalar_one()


def insertar_con_pandas(datos: pd.DataFrame, nombre_tabla: str) -> None:
    """
    Inserta un DataFrame usando `to_sql` de Pandas.

    Es la forma mas comoda y la que hay que conocer primero:

        if_exists="append" -> anade filas y NO recrea la tabla, asi conservamos
                              las claves primarias, ajenas y los CHECK.
        index=False        -> no guardamos el indice de Pandas como columna.
        method="multi"     -> agrupa varias filas en cada INSERT (mas rapido).

    Limitacion: genera sentencias INSERT, y eso tiene un coste. Para 10.000
    filas va perfecto; para millones de filas se queda corto.
    """
    datos.to_sql(
        nombre_tabla,
        get_engine(),
        if_exists="append",
        index=False,
        chunksize=TAMANO_LOTE,
        method="multi",
    )


def insertar_con_copy(datos: pd.DataFrame, nombre_tabla: str) -> None:
    """
    Inserta un DataFrame usando el comando COPY de PostgreSQL.

    COPY es la carga masiva nativa de PostgreSQL: en lugar de ejecutar miles de
    INSERT, envia los datos como un flujo continuo. Suele ser entre 10 y 30
    veces mas rapido, y es lo que se usa de verdad en un pipeline de datos.

    Como funciona aqui:
      1) convertimos el DataFrame a texto CSV en memoria (sin tocar el disco),
      2) le pedimos a PostgreSQL que lea ese flujo con COPY ... FROM STDIN.
    """
    # header=False porque COPY espera solo datos, no la fila de nombres.
    # Los valores nulos quedan como cadena vacia, que COPY interpreta como NULL.
    buffer = io.StringIO()
    datos.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    # Indicamos las columnas explicitamente para no depender de su orden en la tabla.
    columnas = ", ".join(datos.columns)
    sentencia = f"COPY {nombre_tabla} ({columnas}) FROM STDIN WITH (FORMAT CSV)"

    # COPY no existe en SQLAlchemy: es especifico de PostgreSQL. Por eso bajamos
    # a la conexion "cruda" del driver psycopg con raw_connection().
    conexion_raw = get_engine().raw_connection()
    try:
        with conexion_raw.cursor() as cursor:
            with cursor.copy(sentencia) as copia:
                copia.write(buffer.read())
        conexion_raw.commit()
    finally:
        # Devolvemos la conexion al pool aunque algo falle.
        conexion_raw.close()


def cargar_tabla(nombre_tabla: str, vaciar_antes: bool = True, metodo: str = "copy") -> int:
    """
    Carga un CSV completo en su tabla de PostgreSQL y audita el proceso.

    metodo="pandas" -> usa to_sql (didactico, mas lento)
    metodo="copy"   -> usa COPY de PostgreSQL (lo que se usa en produccion)

    vaciar_antes=True hace que la funcion sea repetible: si la ejecutas dos
    veces no acaba con las filas duplicadas (la clave primaria daria error).

    Devuelve el numero de filas insertadas.
    """
    if metodo not in {"pandas", "copy"}:
        raise ValueError("metodo debe ser 'pandas' o 'copy'")

    # Todo lo que pase dentro del `with` queda registrado en la tabla auditoria,
    # con su duracion y su estado, incluso si falla.
    with auditar(proceso="carga_csv", operacion="INSERT", tabla_afectada=nombre_tabla) as info:
        datos = leer_csv(nombre_tabla)

        if vaciar_antes:
            vaciar_tabla(nombre_tabla)

        if metodo == "copy":
            insertar_con_copy(datos, nombre_tabla)
        else:
            insertar_con_pandas(datos, nombre_tabla)

        info["registros"] = len(datos)
        info["mensaje"] = (
            f"Cargadas {len(datos)} filas desde {ARCHIVOS[nombre_tabla]['csv']} "
            f"(metodo={metodo})"
        )

    return len(datos)


def cargar_todo(vaciar_antes: bool = True, metodo: str = "copy") -> dict[str, int]:
    """
    Carga las tres tablas de negocio en el orden correcto.

    El ORDEN IMPORTA: primero clientes y productos, y solo despues ventas.
    Si intentaramos cargar ventas antes, PostgreSQL las rechazaria porque sus
    claves ajenas apuntarian a clientes y productos que aun no existen.
    """
    resultado = {}
    for nombre_tabla in ORDEN_CARGA:
        print(f"Cargando {nombre_tabla}...")
        resultado[nombre_tabla] = cargar_tabla(
            nombre_tabla, vaciar_antes=vaciar_antes, metodo=metodo
        )
        print(f"  {resultado[nombre_tabla]:,} filas")
    return resultado