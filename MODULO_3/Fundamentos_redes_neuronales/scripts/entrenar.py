"""
Entrena la red de motores desde la terminal y la guarda en modelos/.

Es lo mismo que el bloque 5 del notebook, pero como se haría en un proyecto
real: un script que se puede lanzar en un servidor, en un pipeline o cada
noche, sin abrir Jupyter.

Uso (desde la carpeta redes_neuronales/):
    python scripts/entrenar.py
    python scripts/entrenar.py --epocas 60 --capas 64 32
    python scripts/entrenar.py --lineal        # entrena el modelo sin capas ocultas

Pasos del script:
    1. Leer opciones de la terminal.
    2. Fijar la semilla.
    3. Cargar y dividir los datos          (src/datos.py)
    4. Construir y compilar el modelo      (src/modelo.py)
    5. Entrenar con EarlyStopping
    6. Evaluar en test (global y por clase)
    7. Guardar el modelo en modelos/*.keras
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# TensorFlow escribe muchos mensajes informativos en C++ al arrancar.
# "2" = mostrar solo errores. Tiene que ir ANTES de importar tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RAIZ = Path(__file__).resolve().parents[1]
# Igual que en generar_datos.py: permite `from src...` desde la terminal.
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.datos import CLASES, cargar_dataset, dividir_datos  # noqa: E402
from src.modelo import (  # noqa: E402
    compilar,
    construir_modelo_lineal,
    construir_red,
    crear_normalizador,
)


def main() -> None:
    """Ejecuta el entrenamiento completo: datos -> modelo -> fit -> evaluación -> guardado."""
    # 1. OPCIONES DE LA TERMINAL ----------------------------------------------
    parser = argparse.ArgumentParser(description="Entrena la red de mantenimiento predictivo.")
    parser.add_argument("--epocas", type=int, default=40, help="Pasadas completas por train.")
    parser.add_argument("--lote", type=int, default=64, help="Motores por paso de gradiente.")
    # nargs="+" acepta uno o varios números: --capas 64 32 16 -> [64, 32, 16]
    parser.add_argument("--capas", type=int, nargs="+", default=[32, 16], help="Neuronas por capa oculta.")
    # action="store_true": si escribes --lineal vale True; si no, False.
    parser.add_argument("--lineal", action="store_true", help="Entrena el modelo sin capas ocultas.")
    args = parser.parse_args()

    # 2. SEMILLA --------------------------------------------------------------
    # Fija Python, NumPy y TensorFlow a la vez: los pesos iniciales (aleatorios)
    # salen iguales en cada ejecución y los resultados se pueden comparar.
    keras.utils.set_random_seed(42)

    # 3. DATOS ----------------------------------------------------------------
    partes = dividir_datos(cargar_dataset())
    X_train, y_train = partes["train"]
    X_val, y_val = partes["validacion"]
    X_test, y_test = partes["test"]

    # 4. MODELO ---------------------------------------------------------------
    normalizador = crear_normalizador(X_train)
    if args.lineal:
        modelo = construir_modelo_lineal(normalizador)
    else:
        # tuple(...) porque construir_red espera una tupla y argparse da una lista.
        modelo = construir_red(normalizador, capas_ocultas=tuple(args.capas))
    compilar(modelo)
    # .summary() imprime la tabla de capas, shapes y número de parámetros.
    modelo.summary()

    # 5. ENTRENAMIENTO --------------------------------------------------------
    # Un "callback" es una función que Keras llama al final de cada época.
    # EarlyStopping vigila una métrica y para el entrenamiento si deja de mejorar:
    #   monitor="val_loss"         -> vigila la pérdida en validación.
    #   patience=8                 -> espera 8 épocas sin mejora antes de parar.
    #   restore_best_weights=True  -> al parar, vuelve a los pesos de la MEJOR época,
    #                                 no a los de la última.
    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    # .fit() ES el bucle de entrenamiento (el mismo que hicimos a mano con Celsius):
    #   X_train, y_train  -> datos de los que aprende.
    #   validation_data   -> datos que solo mira al final de cada época, sin aprender de ellos.
    #   epochs            -> máximo de pasadas completas por train.
    #   batch_size        -> motores por paso: cada 64 motores calcula gradiente y corrige pesos.
    #   callbacks         -> lista de callbacks (aquí, EarlyStopping).
    #   verbose=2         -> imprime una línea por época.
    modelo.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epocas,
        batch_size=args.lote,
        callbacks=[parada],
        verbose=2,
    )

    # 6. EVALUACIÓN -----------------------------------------------------------
    # .evaluate() calcula la pérdida y las métricas de compile() sobre datos nuevos.
    # Devuelve [pérdida, accuracy] en el orden de compile().
    perdida, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"\nTEST -> pérdida {perdida:.3f} | accuracy {accuracy:.1%}")

    # .predict() devuelve las probabilidades softmax: shape (900, 3).
    # np.argmax(..., axis=1) busca en cada FILA la columna con la mayor
    # probabilidad -> la clase predicha (0, 1 o 2) de cada motor.
    pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)

    # Acierto por clase: el promedio global puede esconder que fallamos justo
    # en las fallas, que es lo más caro para el negocio.
    for indice, nombre in CLASES.items():
        mascara = y_test == indice   # True en los motores que REALMENTE son de esta clase
        print(f"  {nombre:<9} acierto {np.mean(pred[mascara] == indice):.1%}  ({mascara.sum()} motores)")

    # 7. GUARDADO -------------------------------------------------------------
    carpeta = RAIZ / "modelos"
    carpeta.mkdir(exist_ok=True)
    ruta = carpeta / ("motores_lineal.keras" if args.lineal else "motores_red.keras")
    # .save() guarda en un solo archivo la arquitectura, los pesos aprendidos
    # y la normalización. Se recupera con keras.models.load_model(ruta).
    modelo.save(ruta)
    print(f"\nModelo guardado en: {ruta}")


if __name__ == "__main__":
    main()
