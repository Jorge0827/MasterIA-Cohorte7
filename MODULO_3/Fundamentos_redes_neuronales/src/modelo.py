"""
Construcción de las redes con Keras (la API de alto nivel de TensorFlow).

En el notebook hacemos la neurona y la capa densa "a mano" con tensores para
entender qué pasa por dentro. Aquí está la versión que se usa en el día a día:
Keras crea los tensores de pesos (variables), hace la multiplicación de
matrices y calcula los gradientes por nosotros.

QUÉ HAY EN ESTE ARCHIVO
-----------------------
    crear_normalizador()       Capa que pone todos los sensores en la misma escala.
    construir_red()            Red profunda: capas ocultas con ReLU + salida softmax.
    construir_modelo_lineal()  Sin capas ocultas: solo softmax (regresión logística).
    compilar()                 Elige optimizador, función de pérdida y métrica.

Comparar la red con el modelo lineal, con los mismos datos, demuestra para qué
sirven las capas ocultas: sin ellas, el modelo solo separa las clases con rectas.
"""

from __future__ import annotations

import numpy as np

# keras vive DENTRO de tensorflow. `from tensorflow import keras` nos da
# acceso a todo: keras.layers (capas), keras.Sequential (modelos),
# keras.optimizers (optimizadores)...
from tensorflow import keras

# Reutilizamos las constantes de datos.py: si mañana añadimos un sensor,
# solo se toca FEATURES y la red se ajusta sola.
from src.datos import CLASES, FEATURES

N_ENTRADAS = len(FEATURES)  # 5 sensores  -> tamaño de la entrada de la red
N_CLASES = len(CLASES)      # 3 estados   -> neuronas de la capa de salida


def crear_normalizador(X_train: np.ndarray) -> keras.layers.Normalization:
    """
    Capa que estandariza cada sensor: (x - media) / desviación.

    POR QUÉ ES OBLIGATORIO EN REDES
    Las horas de uso llegan a 30 000 y la vibración ronda 3. Si entran tal
    cual, en z = w·x + b las horas dominan la suma y los gradientes de cada
    peso tienen escalas muy distintas: el entrenamiento oscila o no converge.
    Tras normalizar, todos los sensores viven alrededor de 0 con desviación 1.

    POR QUÉ SOLO CON TRAIN
    `adapt` calcula la media y la desviación SOLO con train. Si usáramos
    también test, el modelo "vería" información del examen (fuga de datos).

    POR QUÉ DENTRO DEL MODELO
    Al ser una capa, la normalización viaja con el archivo .keras guardado:
    en producción se le pasan lecturas crudas y el modelo las escala solo.
    """
    # Se crea la capa VACÍA: todavía no sabe ninguna media ni desviación.
    # `name` es solo la etiqueta que aparecerá en model.summary().
    normalizador = keras.layers.Normalization(name="normalizacion")

    # .adapt(datos) recorre los datos y GUARDA dentro de la capa la media y la
    # varianza de cada columna (5 medias + 5 varianzas). No es entrenamiento
    # con gradiente: es un cálculo estadístico que se hace una sola vez.
    normalizador.adapt(X_train)
    return normalizador


def construir_red(
    normalizador: keras.layers.Normalization,
    capas_ocultas: tuple[int, ...] = (32, 16),
    activacion: str = "relu",
) -> keras.Model:
    """
    Red neuronal densa (también llamada MLP o perceptrón multicapa).

    Parámetros
    ----------
    normalizador : la capa creada con crear_normalizador().
    capas_ocultas : neuronas de cada capa oculta. (32, 16) = dos capas de 32 y 16.
    activacion : activación de las capas ocultas. "relu" por defecto.

    Arquitectura con los valores por defecto:

        5 sensores
          -> Normalización              (no aprende, solo escala)
          -> Dense(32, ReLU)            capa oculta 1: 32 neuronas
          -> Dense(16, ReLU)            capa oculta 2: 16 neuronas
          -> Dense(3, softmax)          capa de salida: una neurona por clase

    Cada Dense hace exactamente lo del notebook: salida = activación(X @ W + b).

    * ReLU en las capas ocultas: deja pasar los positivos y apaga los negativos.
      Es lo que le da a la red la capacidad de dibujar fronteras curvas.
    * Softmax en la salida: convierte 3 números cualesquiera en 3 probabilidades
      que suman 1. Ejemplo: [0.05, 0.15, 0.80] -> "80 % de que sea falla".

    El tamaño 32 -> 16 es una convención (embudo que va resumiendo la
    información). No hay fórmula: se elige probando, empezando pequeño.

    `activacion="linear"` quita la no linealidad de las capas ocultas. Sirve
    para el experimento de clase: con muchas capas pero sin ReLU, la red rinde
    igual que el modelo lineal, porque apilar capas lineales sigue siendo lineal.
    """
    # keras.Sequential = un modelo que es una PILA de capas: la salida de
    # cada capa es la entrada de la siguiente, en el orden en que se añaden.
    modelo = keras.Sequential(name="red_motores")

    # keras.Input declara la forma de UNA muestra: un vector de 5 sensores.
    # El tamaño del lote no se pone: Keras lo deja libre (aparece como None),
    # así la misma red sirve para 1 motor o para 10 000.
    modelo.add(keras.Input(shape=(N_ENTRADAS,), name="sensores"))

    # Primera capa: escalar los sensores.
    modelo.add(normalizador)

    # Una capa Dense por cada número de `capas_ocultas`.
    # enumerate(..., start=1) da pares (1, 32), (2, 16): el número de capa
    # (para el nombre) y cuántas neuronas lleva.
    for i, neuronas in enumerate(capas_ocultas, start=1):
        # keras.layers.Dense(units, activation, name):
        #   units      -> cuántas neuronas tiene la capa (columnas de W).
        #   activation -> función que se aplica a X @ W + b.
        # Las filas de W (cuántas entradas) Keras las deduce sola de la capa anterior.
        modelo.add(keras.layers.Dense(neuronas, activation=activacion, name=f"oculta_{i}"))

    # Capa de salida: 3 neuronas (una por clase) con softmax -> 3 probabilidades.
    modelo.add(keras.layers.Dense(N_CLASES, activation="softmax", name="salida"))
    return modelo


def construir_modelo_lineal(normalizador: keras.layers.Normalization) -> keras.Model:
    """
    El mismo modelo SIN capas ocultas: 5 entradas -> 3 salidas softmax.

    Matemáticamente es una regresión logística multiclase. Cada clase recibe
    una puntuación que es una suma ponderada de los sensores, o sea, una
    función lineal. No puede representar la "U" de las rpm ni el efecto
    combinado de temperatura y vibración. Lo usamos como punto de comparación.
    """
    modelo = keras.Sequential(name="modelo_lineal")
    modelo.add(keras.Input(shape=(N_ENTRADAS,), name="sensores"))
    modelo.add(normalizador)
    # Directamente de los 5 sensores a las 3 probabilidades, sin nada en medio.
    modelo.add(keras.layers.Dense(N_CLASES, activation="softmax", name="salida"))
    return modelo


def compilar(modelo: keras.Model, tasa_aprendizaje: float = 0.003) -> keras.Model:
    """
    Le dice a la red CÓMO va a aprender. No entrena nada todavía.

    * optimizer=Adam
        La regla para mover los pesos. En cada paso calcula el gradiente
        (hacia dónde sube el error) y mueve cada peso un poco en sentido
        contrario. La tasa de aprendizaje es el tamaño de ese paso:
        muy grande -> salta y no converge; muy pequeña -> tarda una eternidad.
        Adam además adapta el paso de cada peso por separado, por eso suele
        funcionar bien sin mucho ajuste.

    * loss=sparse_categorical_crossentropy
        El número que la red intenta bajar. Mira la probabilidad que la red
        dio a la clase CORRECTA y castiga con -log(probabilidad):
            acertó con 0.9 -> pérdida 0.10   (poco castigo)
            acertó con 0.1 -> pérdida 2.30   (mucho castigo)
        "sparse" significa que las etiquetas son enteros (0, 1, 2) y no one-hot.

    * metrics=accuracy
        Solo para que nosotros lo leamos: % de motores bien clasificados.
        La red NO optimiza el accuracy; optimiza la pérdida.
    """
    # .compile(...) no devuelve nada nuevo: configura el modelo "en el sitio".
    modelo.compile(
        # Objeto optimizador. learning_rate = tamaño del paso en cada corrección.
        optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        # Se pasa como texto: Keras busca la función con ese nombre.
        loss="sparse_categorical_crossentropy",
        # Lista porque se pueden pedir varias métricas a la vez.
        metrics=["accuracy"],
    )
    # Devolvemos el mismo modelo para poder encadenar: compilar(construir_red(...)).
    return modelo
