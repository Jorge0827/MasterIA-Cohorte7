"""
Auditoria de procesos.

Cada vez que un proceso toca los datos (cargar un CSV, lanzar predicciones,
limpiar una tabla...) dejamos constancia en la tabla `auditoria`: que se hizo,
sobre que tabla, cuantos registros, si fue bien y cuanto tardo.

Por que importa: cuando manana un pipeline falle a las 3 de la madrugada, la
auditoria es lo unico que te dira que paso. En las siguientes clases usaremos
esta misma tabla para monitorizar la calidad de los datos.

Dos formas de usarla:

1) Registro manual, cuando ya sabes el resultado:

    registrar_auditoria("carga_csv", "INSERT", "clientes", 10000, "exito")

2) Registro automatico con un bloque `with`, que mide el tiempo y captura errores:

    with auditar("carga_csv", "INSERT", "clientes") as auditoria:
        filas = cargar_algo()
        auditoria["registros"] = filas
"""

import time
from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from sqlalchemy import insert, select

from src.db import get_engine
from src.schema import auditoria as tabla_auditoria


def registrar_auditoria(
    proceso: str,
    operacion: str,
    tabla_afectada: str | None = None,
    registros_procesados: int = 0,
    estado: str = "exito",
    mensaje: str | None = None,
    duracion_segundos: float | None = None,
) -> int:
    """
    Inserta una fila en la tabla `auditoria` y devuelve su id.

    Parametros:
        proceso              nombre del proceso, ej. "carga_csv"
        operacion            que se hizo: INSERT, UPDATE, SELECT, DELETE...
        tabla_afectada       tabla sobre la que se actuo
        registros_procesados numero de filas tratadas
        estado               "exito" o "error" (lo valida un CHECK en la BD)
        mensaje              texto libre: resumen o error
        duracion_segundos    cuanto tardo el proceso

    La fecha_hora no se envia: la pone PostgreSQL con su DEFAULT now().
    """
    engine = get_engine()

    # insert(tabla).values(...) es la forma de SQLAlchemy de escribir un INSERT.
    # returning() nos devuelve el id generado por PostgreSQL, para no tener que
    # hacer una segunda consulta.
    sentencia = (
        insert(tabla_auditoria)
        .values(
            proceso=proceso,
            operacion=operacion,
            tabla_afectada=tabla_afectada,
            registros_procesados=registros_procesados,
            estado=estado,
            mensaje=mensaje,
            duracion_segundos=duracion_segundos,
        )
        .returning(tabla_auditoria.c.auditoria_id)
    )

    # engine.begin() abre transaccion y hace COMMIT al salir del bloque.
    with engine.begin() as conexion:
        return conexion.execute(sentencia).scalar_one()


@contextmanager
def auditar(
    proceso: str,
    operacion: str,
    tabla_afectada: str | None = None,
) -> Iterator[dict]:
    """
    Gestor de contexto que audita automaticamente un bloque de codigo.

    Mide el tiempo, y registra "exito" si el bloque termina bien o "error" si
    lanza una excepcion (y vuelve a lanzarla, para no ocultar el problema).

    Dentro del bloque recibes un diccionario donde puedes indicar el resultado:

        with auditar("carga_csv", "INSERT", "ventas") as info:
            info["registros"] = 200000
            info["mensaje"] = "carga completa"
    """
    inicio = time.perf_counter()
    info = {"registros": 0, "mensaje": None}

    try:
        # Aqui se ejecuta el codigo que esta dentro del bloque `with`.
        yield info
    except Exception as error:
        duracion = time.perf_counter() - inicio
        registrar_auditoria(
            proceso=proceso,
            operacion=operacion,
            tabla_afectada=tabla_afectada,
            registros_procesados=info["registros"],
            estado="error",
            # Recortamos el mensaje: un traceback completo no aporta aqui.
            mensaje=f"{type(error).__name__}: {error}"[:500],
            duracion_segundos=round(duracion, 3),
        )
        raise  # el error sigue su camino: auditar no significa silenciar
    else:
        duracion = time.perf_counter() - inicio
        registrar_auditoria(
            proceso=proceso,
            operacion=operacion,
            tabla_afectada=tabla_afectada,
            registros_procesados=info["registros"],
            estado="exito",
            mensaje=info["mensaje"],
            duracion_segundos=round(duracion, 3),
        )


def ultimas_auditorias(limite: int = 10) -> pd.DataFrame:
    """Devuelve los ultimos registros de auditoria, del mas reciente al mas antiguo."""
    engine = get_engine()

    # select() construye el SELECT. order_by(...desc()) ordena descendente.
    consulta = (
        select(tabla_auditoria)
        .order_by(tabla_auditoria.c.auditoria_id.desc())
        .limit(limite)
    )

    with engine.connect() as conexion:
        return pd.read_sql_query(consulta, conexion)