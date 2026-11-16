"""
Dataset de mantenimiento predictivo de motores industriales.

EL PROBLEMA DE NEGOCIO
----------------------
Una planta tiene cientos de motores eléctricos. Cada motor lleva sensores
que mandan cinco lecturas:

    temperatura_c    Temperatura de la carcasa en grados Celsius.
    vibracion_mm_s   Velocidad de vibración en mm/s (la norma ISO 10816 la usa
                     para decir si un motor está sano).
    rpm              Revoluciones por minuto. El motor está diseñado para 1500.
    corriente_a      Corriente eléctrica que consume, en amperios.
    horas_uso        Horas acumuladas desde la última revisión.

Queremos que una red neuronal mire esas cinco lecturas y diga en qué estado
está el motor:

    0 = normal     -> no hacer nada
    1 = desgaste   -> programar una revisión
    2 = falla      -> parar el motor ya

Es CLASIFICACIÓN MULTICLASE (3 clases). Por eso la última capa de la red
usará softmax: devuelve una probabilidad por clase.

POR QUÉ SINTÉTICO
-----------------
Generamos los datos con reglas físicas inventadas pero creíbles. Así sabemos
exactamente qué patrón hay escondido y podemos comprobar si la red lo encuentra.

Las reglas tienen a propósito relaciones NO LINEALES:

* Las rpm son peligrosas en los DOS sentidos: ir muy lento o muy rápido estresa
  el motor. Eso es una "U" (valor absoluto), y una recta no puede dibujar una U.
* Temperatura y vibración se potencian entre sí (un término temp * vibración).
* La corriente alta solo es grave si además el motor gira lento (atasco).

Un modelo lineal se queda corto con esto. Una red con capas ocultas y ReLU no.
Esa diferencia es la demostración central de la clase.

QUÉ HAY EN ESTE ARCHIVO
-----------------------
    Constantes       RAIZ, RUTA_CSV, FEATURES, TARGET, CLASES, RPM_NOMINAL
    generar_dataset  Fabrica las lecturas y les pone etiqueta (0, 1, 2).
    guardar_dataset  Escribe el DataFrame en un CSV.
    cargar_dataset   Lee el CSV (y lo genera si no existe).
    dividir_datos    Parte en train / validación / test y pasa a NumPy.
"""

# Permite escribir anotaciones de tipo modernas (dict[str, ...]) en cualquier
# versión de Python 3. No cambia el comportamiento del código.
from __future__ import annotations

from pathlib import Path   # Path representa rutas de archivos sin pelear con "\" o "/".

import numpy as np         # Cálculo numérico con arrays: aquí sorteamos los sensores.
import pandas as pd        # Tablas (DataFrame): aquí armamos y guardamos el dataset.

# -----------------------------------------------------------------------------
# CONSTANTES DEL MÓDULO
# Se escriben en MAYÚSCULAS por convención: indica "esto no se modifica".
# Cualquier archivo puede importarlas: from src.datos import FEATURES
# -----------------------------------------------------------------------------

# __file__ es la ruta de ESTE archivo (.../redes_neuronales/src/datos.py).
#   .resolve()   -> la convierte en ruta absoluta (completa, desde C:\).
#   .parents[0]  -> la carpeta que lo contiene: .../src
#   .parents[1]  -> la carpeta de arriba:       .../redes_neuronales
# Así la ruta funciona igual en cualquier ordenador, sin escribir C:\Users\...
RAIZ = Path(__file__).resolve().parents[1]

# El operador / de Path une trozos de ruta: RAIZ/data/sensores_motores.csv
RUTA_CSV = RAIZ / "data" / "sensores_motores.csv"

# Orden fijo de las columnas de entrada. La red aprende pesos POR POSICIÓN:
# el peso 0 multiplica a la columna 0. Si mañana cambias el orden, el peso
# que aprendió para la temperatura se aplicaría a las rpm.
FEATURES = ["temperatura_c", "vibracion_mm_s", "rpm", "corriente_a", "horas_uso"]

# Nombre de la columna que queremos predecir.
TARGET = "estado"

# Diccionario índice -> nombre legible. La red solo conoce números (0, 1, 2);
# esto sirve para traducir su respuesta a algo que entienda una persona.
CLASES = {0: "normal", 1: "desgaste", 2: "falla"}

# Velocidad de diseño del motor. Alejarse de ella (por arriba o por abajo) es malo.
RPM_NOMINAL = 1500


def generar_dataset(n: int = 6000, semilla: int = 42) -> pd.DataFrame:
    """
    Simula `n` lecturas de sensores y les asigna un estado (0, 1 o 2).

    Parámetros
    ----------
    n : cuántos motores (filas) fabricar. Por defecto 6000.
    semilla : número que fija el azar. Misma semilla -> mismo CSV en todos los ordenadores.

    Devuelve
    --------
    DataFrame con 5 columnas de sensores + la columna `estado`.

    Pasos:
      1. Se sortea cada sensor con una distribución realista.
      2. Se calcula un "riesgo" oculto combinando los sensores con reglas
         no lineales (ver docstring del módulo).
      3. Se suma ruido: dos motores con las mismas lecturas no siempre
         terminan igual. Por eso NINGÚN modelo llegará al 100 %.
      4. El riesgo se corta en dos umbrales -> 3 clases.
    """
    # Generador de números aleatorios de NumPy. Todo el azar sale de aquí,
    # así que fijando `semilla` se repite exactamente la misma secuencia.
    rng = np.random.default_rng(semilla)

    # 1. LECTURAS DE SENSORES -------------------------------------------------
    # Cada línea devuelve un array de `n` números (uno por motor).

    # rng.normal(media, desviación, cantidad): campana de Gauss.
    # La mayoría de motores andará entre 55 y 75 °C; pocos en los extremos.
    temperatura = rng.normal(65, 10, n)

    # rng.gamma(forma, escala, cantidad): siempre positiva y con "cola larga"
    # a la derecha. Encaja con la vibración: casi todos vibran poco, unos
    # pocos vibran muchísimo, y nunca hay vibración negativa.
    vibracion = rng.gamma(2.0, 1.4, n)

    # Campana alrededor de la velocidad de diseño (1500), desviación 350 rpm.
    rpm = rng.normal(RPM_NOMINAL, 350, n)

    # Campana alrededor de 12 amperios.
    corriente = rng.normal(12, 3, n)

    # rng.uniform(mínimo, máximo, cantidad): cualquier valor del rango es
    # igual de probable. Hay motores recién revisados y otros muy usados.
    # (El guion bajo en 30_000 solo hace el número más legible: es 30000.)
    horas = rng.uniform(0, 30_000, n)

    # 2. RIESGO OCULTO: la "física" que la red tendrá que descubrir ----------
    # np.abs = valor absoluto. Vale 0 en 1500 rpm y crece hacia LOS DOS lados.
    # Dividir por 350 lo expresa en "desviaciones típicas". Esta es la "U"
    # que un modelo lineal no puede representar.
    desviacion_rpm = np.abs(rpm - RPM_NOMINAL) / 350

    # Todas estas operaciones son ELEMENTO A ELEMENTO: se calculan para los
    # 6000 motores a la vez, sin ningún bucle for.
    riesgo = (
        # Lineal: cada grado por encima de 65 suma 0.06 de riesgo.
        0.06 * (temperatura - 65)
        # Lineal: cada mm/s por encima de 2.8 suma 0.45.
        + 0.45 * (vibracion - 2.8)
        # np.maximum(x, 0) se queda con el mayor de los dos, elemento a elemento:
        # si la desviación es menor que 0.8 da 0 (sin efecto); si la supera,
        # el exceso se eleva al cuadrado (el riesgo crece cada vez más rápido).
        + 1.1 * np.maximum(desviacion_rpm - 0.8, 0) ** 2
        # Desgaste por uso: 30 000 horas suman 1.8 de riesgo.
        + 0.00006 * horas
        # (rpm < 1300) da un array de True/False; al multiplicar, True vale 1
        # y False vale 0. Resultado: la corriente alta SOLO cuenta si el motor
        # va lento. Es una condición "Y", que una suma simple no puede expresar.
        + 0.5 * np.maximum(corriente - 15, 0) * (rpm < 1300)
        # Interacción: el producto hace que calor Y vibración juntos sean
        # peor que la suma de cada uno por separado.
        + 0.04 * (temperatura - 65) * (vibracion - 2.8)
    )

    # 3. RUIDO: el mundo real no es determinista.
    # Sumamos una campana pequeña (media 0, desviación 0.25) a cada motor.
    riesgo = riesgo + rng.normal(0, 0.25, n)

    # 4. DOS UMBRALES -> TRES CLASES
    # np.where(condición, si_verdadero, si_falso) elige elemento a elemento.
    # Anidado se lee así:
    #   si riesgo > 2.2           -> 2 (falla)
    #   si no, si riesgo > 1.0    -> 1 (desgaste)
    #   si no                     -> 0 (normal)
    estado = np.where(riesgo > 2.2, 2, np.where(riesgo > 1.0, 1, 0))

    # Armamos la tabla: cada clave del diccionario es una columna.
    # .round(n) deja n decimales, como mandaría un sensor real.
    # OJO: el `riesgo` NO se guarda. La red solo verá los sensores y el
    # estado; tendrá que descubrir la regla oculta por su cuenta.
    return pd.DataFrame(
        {
            "temperatura_c": temperatura.round(1),
            "vibracion_mm_s": vibracion.round(2),
            "rpm": rpm.round(0),
            "corriente_a": corriente.round(2),
            "horas_uso": horas.round(0),
            TARGET: estado,
        }
    )


def guardar_dataset(df: pd.DataFrame, ruta: Path = RUTA_CSV) -> Path:
    """
    Escribe el DataFrame en CSV y devuelve la ruta donde quedó.

    Si la carpeta `data/` no existe, la crea.
    """
    # ruta.parent = la carpeta que contiene el archivo (redes_neuronales/data).
    #   parents=True  -> crea también las carpetas intermedias si faltan.
    #   exist_ok=True -> si ya existe, no da error.
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # index=False: no escribir la columna de índice (0, 1, 2...) de pandas,
    # que no es un dato del motor.
    df.to_csv(ruta, index=False)
    return ruta


def cargar_dataset(ruta: Path = RUTA_CSV) -> pd.DataFrame:
    """
    Lee el CSV. Si todavía no existe, lo genera y lo guarda primero.

    Así el notebook y los scripts funcionan aunque nadie haya ejecutado
    antes `scripts/generar_datos.py`.
    """
    # .exists() devuelve True si el archivo está en disco.
    if not ruta.exists():
        guardar_dataset(generar_dataset(), ruta)
    return pd.read_csv(ruta)


def dividir_datos(
    df: pd.DataFrame, semilla: int = 42
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Separa el dataset en tres partes y lo convierte a arrays de NumPy.

        train       (70 %)  La red ajusta sus pesos con estos datos.
        validacion  (15 %)  Se mira en cada época para detectar sobreajuste.
                            La red NO aprende de ellos.
        test        (15 %)  Se usa UNA vez al final. Es la nota "de examen".

    Devuelve un diccionario: {"train": (X, y), "validacion": (X, y), "test": (X, y)}

    Detalles que importan en redes neuronales:

    * `float32`: TensorFlow trabaja en 32 bits por defecto. Pandas da float64;
      convertir aquí evita avisos y gasta la mitad de memoria.
    * El target se queda como entero (0, 1, 2). Por eso en `compile` usaremos
      `sparse_categorical_crossentropy` y no hace falta one-hot.
    * Estratificar mantiene la misma proporción de fallas en las tres partes.
      Las fallas son minoría: sin estratificar, el test podría quedarse con pocas.
    """
    # Se importa aquí dentro porque solo esta función usa scikit-learn.
    from sklearn.model_selection import train_test_split

    # df[FEATURES] -> solo las 5 columnas de sensores, en el orden fijado.
    # .to_numpy(dtype=...) -> la tabla pasa a ser un array de NumPy (matriz),
    # que es lo que entiende TensorFlow. X tiene shape (6000, 5).
    X = df[FEATURES].to_numpy(dtype="float32")
    # y tiene shape (6000,): un entero por motor.
    y = df[TARGET].to_numpy(dtype="int32")

    # train_test_split solo sabe partir en DOS. Para tener tres, lo usamos dos veces.
    #
    # Primer corte: 70 % train y 30 % "resto".
    #   test_size=0.30     -> qué fracción va al segundo grupo.
    #   random_state       -> semilla: el mismo reparto en cada ejecución.
    #   stratify=y         -> mantiene el % de cada clase en los dos grupos.
    # Devuelve 4 arrays en este orden: X_1, X_2, y_1, y_2.
    X_train, X_resto, y_train, y_resto = train_test_split(
        X, y, test_size=0.30, random_state=semilla, stratify=y
    )

    # Segundo corte: el 30 % "resto" se parte por la mitad -> 15 % y 15 %.
    X_val, X_test, y_val, y_test = train_test_split(
        X_resto, y_resto, test_size=0.50, random_state=semilla, stratify=y_resto
    )

    # Un diccionario con nombre hace el código que lo usa más legible:
    # partes["train"] se entiende mejor que partes[0].
    return {
        "train": (X_train, y_train),
        "validacion": (X_val, y_val),
        "test": (X_test, y_test),
    }
