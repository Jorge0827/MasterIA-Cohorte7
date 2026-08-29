"""
Registro de predicciones en PostgreSQL.

En esta clase NO entrenamos ningun modelo: lo simulamos. Lo que si es real es
la parte que nos interesa hoy: **donde se guarda una prediccion y con que datos**.

Una prediccion suelta no sirve de nada. Para poder auditar un modelo en
produccion necesitamos guardar, como minimo:

    modelo, version   -> quien predijo (imprescindible para comparar versiones)
    fecha_prediccion  -> cuando
    cliente_id        -> sobre quien
    valor_predicho    -> que dijo el modelo
    confianza         -> lo seguro que estaba
    resultado_real    -> que paso de verdad (se rellena despues)
    estado            -> si la ejecucion fue bien
    tiempo_ejecucion  -> cuanto tardo (latencia)

Con esas columnas podemos medir mas tarde el error del modelo (predicho vs real).
"""

import random
import time

import pandas as pd
from sqlalchemy import insert, select, text, update

from src.db import get_engine
from src.schema import predicciones as tabla_predicciones

# Identificacion del modelo simulado. La version es texto porque suele ser
# semantica ("1.0.0") y porque el modelo se reentrena y cambia de version.
MODELO = "prediccion_gasto_cliente"
VERSION = "1.0.0"


def simular_prediccion(cliente_id: int, semilla: int | None = None) -> dict:
    """
    Simula la prediccion del gasto futuro de un cliente.

    Para que el numero no sea absurdo, partimos del gasto medio REAL del cliente
    en la tabla `ventas` y le aplicamos una variacion aleatoria. Es lo que haria
    un modelo sencillo de verdad: mirar el historico.

    Devuelve un diccionario con lo que hay que guardar en la tabla predicciones.
    """
    # random.Random(semilla) crea su propio generador: asi podemos repetir el
    # mismo resultado en clase sin afectar al resto del programa.
    aleatorio = random.Random(semilla)

    inicio = time.perf_counter()

    # Consultamos el historico del cliente. COALESCE devuelve 0 si el cliente
    # todavia no tiene ventas (evita quedarnos con un None).
    engine = get_engine()
    with engine.connect() as conexion:
        gasto_medio = conexion.execute(
            text("""
                SELECT COALESCE(AVG(total), 0)
                FROM ventas
                WHERE cliente_id = :cliente_id
            """),
            {"cliente_id": cliente_id},
        ).scalar()

    # "Inferencia" simulada: el gasto medio con una variacion de +-30%.
    factor = aleatorio.uniform(0.7, 1.3)
    valor_predicho = round(float(gasto_medio) * factor, 4)

    # Confianza simulada entre 0.60 y 0.99 (la BD exige que este entre 0 y 1).
    confianza = round(aleatorio.uniform(0.60, 0.99), 4)

    # Latencia real de esta funcion, en milisegundos.
    tiempo_ejecucion_ms = int((time.perf_counter() - inicio) * 1000)

    return {
        "cliente_id": cliente_id,
        "valor_predicho": valor_predicho,
        "confianza": confianza,
        "tiempo_ejecucion_ms": tiempo_ejecucion_ms,
    }


def registrar_prediccion(
    cliente_id: int,
    valor_predicho: float,
    confianza: float | None = None,
    tiempo_ejecucion_ms: int | None = None,
    resultado_real: float | None = None,
    estado: str = "ok",
    modelo: str = MODELO,
    version: str = VERSION,
) -> int:
    """
    Guarda una prediccion en PostgreSQL y devuelve el id generado.

    resultado_real se deja en None: cuando predecimos todavia no sabemos que va
    a pasar. Se rellena mas tarde con actualizar_resultado_real().
    """
    engine = get_engine()

    sentencia = (
        insert(tabla_predicciones)
        .values(
            modelo=modelo,
            version=version,
            cliente_id=cliente_id,
            valor_predicho=valor_predicho,
            confianza=confianza,
            resultado_real=resultado_real,
            estado=estado,
            tiempo_ejecucion_ms=tiempo_ejecucion_ms,
        )
        .returning(tabla_predicciones.c.prediccion_id)
    )

    with engine.begin() as conexion:
        return conexion.execute(sentencia).scalar_one()


def predecir_y_registrar(cliente_id: int, semilla: int | None = None) -> int:
    """
    Flujo completo de una prediccion: simular -> guardar.

    Es la funcion que usaremos en el notebook, porque muestra el ciclo real:
    el modelo produce un numero y ese numero se PERSISTE.
    """
    resultado = simular_prediccion(cliente_id, semilla=semilla)
    return registrar_prediccion(**resultado)


def actualizar_resultado_real(prediccion_id: int, resultado_real: float) -> int:
    """
    Anota lo que ocurrio de verdad para una prediccion ya guardada.

    Este UPDATE es lo que permite calcular despues el error del modelo
    (por ejemplo: AVG(ABS(valor_predicho - resultado_real))).
    Devuelve el numero de filas actualizadas.
    """
    engine = get_engine()

    sentencia = (
        update(tabla_predicciones)
        .where(tabla_predicciones.c.prediccion_id == prediccion_id)
        .values(resultado_real=resultado_real)
    )

    with engine.begin() as conexion:
        return conexion.execute(sentencia).rowcount


def ultimas_predicciones(limite: int = 10) -> pd.DataFrame:
    """Devuelve las ultimas predicciones registradas, de la mas nueva a la mas antigua."""
    engine = get_engine()

    consulta = (
        select(tabla_predicciones)
        .order_by(tabla_predicciones.c.prediccion_id.desc())
        .limit(limite)
    )

    with engine.connect() as conexion:
        return pd.read_sql_query(consulta, conexion)