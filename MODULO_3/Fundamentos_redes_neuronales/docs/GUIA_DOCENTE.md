# Guía docente — Unidad 1 · Clase 1: Fundamentos de Redes Neuronales (ANN) con TensorFlow

**Duración:** 120 minutos
**Notebook:** `notebooks/01_fundamentos_redes_neuronales.ipynb`
**Código de apoyo:** `src/datos.py`, `src/modelo.py`, `scripts/entrenar.py`

---

## La idea

> *"Hoy no empezamos por `model.fit()`. Empezamos por un número dentro de un tensor
> y terminamos con una red que decide si se para un motor."*

Cada escalón se hace **a mano primero** y luego se muestra que TensorFlow/Keras hace
lo mismo. Así Keras deja de ser una caja negra.

El hilo conductor es **un solo problema** (motores: normal / desgaste / falla).
Los ejemplos de tensores, la neurona y la capa usan sensores de motores, así el bloque 5
no se siente como un tema nuevo.

---

## Antes de clase

- [ ] `conda activate py_env`
- [ ] `python -c "import tensorflow as tf; print(tf.__version__)"` funciona
      (si da `VersionError` de protobuf: ver README)
- [ ] `python scripts/generar_datos.py` (el notebook lo genera solo si falta, pero mejor tenerlo)
- [ ] Ejecutar el notebook entero una vez (*Restart & Run All*, ~3 min) y luego borrar salidas
- [ ] Tener abiertos en el editor: notebook, `src/datos.py`, `src/modelo.py`
- [ ] Pizarra o tablet para dibujar la neurona y la regla de shapes

---

## Minutado

| Min | Bloque | Qué haces | Qué deben llevarse |
|---|---|---|---|
| 0–10 | 0 · Apertura | Tabla ML vs DL. Presentas los 5 sensores y las 3 clases. Ejecutas la celda de entorno. | En DL la red crea las features. |
| 10–30 | 1 · Tensores | Rango/shape/dtype → error int+float → constante inmutable → `*` vs `@` → reducciones y broadcasting → regla de shapes → `tf.Variable` → demo Celsius. | Datos = constantes; pesos = variables; aprender = `assign_sub` guiado por el gradiente. |
| 30–42 | 2 · Neurona | Dibujas la neurona. Celda paso a paso (`x*w`, suma, `+b`, sigmoide). Lote de 4 motores con `@`. | `z = w·x + b`, y un lote entero es un solo `@`. |
| 42–60 | 3 · Activaciones | Funciones a mano vs `tf.nn`. Gráfico. Softmax con barras. Truco del máximo. **Demo del colapso lineal.** | Sin activación, 100 capas = 1 capa. |
| 60–70 | 4 · Capa densa | Fórmula `X @ W + b` y tabla de shapes. Capa manual. Keras con nuestros pesos → `True`. | Contar parámetros: `entradas × neuronas + neuronas`. |
| 70–115 | 5 · Los `.py` y la red real | Recorrido por `datos.py` → `modelo.py` → `entrenar.py`, leyendo cada función **antes** de usarla (ver detalle abajo). | Cómo se organiza un proyecto de DL y que las capas ocultas con ReLU aportan. |
| 115–120 | 6 · Cierre | Seis ideas. Una pregunta de la lista. Anuncias la próxima clase. | — |

---

## Guion por bloque

### Bloque 1 — Tensores (25 min)

- **Arranca con la tabla de rangos** y relaciónala con los motores: "un motor es un vector,
  un lote de motores es una matriz". Esa frase la vas a repetir en el bloque 5.
- **Error int + float:** déjalo fallar en pantalla. Pregunta *"¿por qué NumPy sí lo permite
  y TensorFlow no?"* (en GPU, convertir en silencio es caro y esconde bugs).
- **`*` vs `@`:** haz la esquina `1·10 + 2·30 = 70` en la pizarra. Es la operación que
  hace una neurona.
- **Regla de shapes:** escribe `(3, 5) @ (5, 2) → (3, 2)` y tacha los 5. Diles que es el error
  que más van a ver en su vida con redes.
- **Demo Celsius:** es el momento clave del bloque. Narra los 5 pasos del bucle mientras corre.
  Señala que el multiplicador llega a 1.800 y el sesgo a 32.00 **sin que nadie les dijera la fórmula**.
  Frase: *"Esto es entrenar. `fit()` es este `for` con más variables."*

### Bloque 2 — Neurona (15 min)

- Dibuja la neurona antes de ejecutar nada. Pesos = importancia; sesgo = umbral.
- Ejecuta la celda y **lee cada línea en voz alta**: `x*w`, suma, `+b`, sigmoide → 0.84.
- Pregunta: *"¿qué peso subirían si la vibración es lo más peligroso?"*
- Lote de 4 motores: el motor típico (todo 0) da `z = b = -3` → alerta baja. Muestra
  que el sesgo decide qué pasa cuando no hay señal.

### Bloque 3 — Activaciones (20 min)

- La sigmoide ya apareció en el bloque 2: ahora se explica.
- **Gráfico:** señala la parte plana de la sigmoide → gradiente casi 0 → *vanishing gradient*.
  ReLU: pendiente 1 en positivos → el gradiente pasa limpio.
- **Softmax con barras:** el logit 4 es el doble que el 2 pero se lleva el 84 %.
- **Colapso lineal (no lo saltes):** `True` sin activación, `False` con ReLU. Esta celda
  es la que vuelve en el bloque 5 con datos reales.

### Bloque 4 — Capa densa (15 min)

- Tabla de shapes en la pizarra. Haz que calculen los parámetros antes de ejecutar (18).
- Los ceros de la salida son neuronas "apagadas" por ReLU para ese motor.
- Keras con nuestros pesos → `True`. Frase: *"Keras no hace magia: `kernel` es nuestra W."*

### Bloque 5 — Los `.py` y la red real (45 min)

**Regla del bloque:** ninguna función se usa sin haber leído antes su código. El notebook la
muestra con `ver_codigo(función)` (el texto real del `.py`) y la celda siguiente la explica
línea por línea. Ten el archivo abierto también en el editor.

| Min | Sección | Archivo | Qué haces |
|---|---|---|---|
| 70–73 | intro | — | Tabla notebook vs `.py`, árbol de carpetas, cómo encuentra Python `src` (`sys.path`, `__init__.py`). |
| 73–83 | 5.1 | `src/datos.py` | Constantes (`Path`, `parents[1]`). `generar_dataset` + celda de métodos en miniatura (la U con 5 valores). `guardar`/`cargar`. Lanzador `generar_datos.py` (`argparse`, `__main__`) y lo ejecutas. |
| 83–88 | 5.2–5.3 | `datos.py` | Exploración y gráfico de la U. `dividir_datos`: los dos cortes 70/30 → 15/15 y `stratify` (la celda lo comprueba). |
| 88–96 | 5.4 | `src/modelo.py` | `crear_normalizador` (`adapt`). `construir_red` (`Sequential`, `Input`, `Dense`). `summary()` y la celda de variables: `kernel (5, 32)` es la `W` del bloque 4. |
| 96–104 | 5.5 | `modelo.py` | `compilar` + celda de `-log(p)`. Argumentos de `fit`. Entrenas en vivo. Curvas. |
| 104–112 | 5.6–5.7 | `modelo.py` | `construir_modelo_lineal`. **Tabla de comparación.** Motores A–E. |
| 112–115 | 5.8 | `scripts/entrenar.py` | Solo lo nuevo: `EarlyStopping`, `evaluate`, `save`. Ejecutar y `load_model`. |

- **Abre `src/datos.py` en el editor** y lee el docstring del problema. Explica por qué la
  lógica vive en `.py` y no en el notebook (reutilizable, versionable, usable desde un script).
- **Celda de métodos en miniatura:** es la que hace entendible `generar_dataset`. Detente en la U
  (`|rpm - 1500|`) y en `True * 1 = 1`.
- **Gráfico de la U:** pregunta *"¿qué peso sobre las rpm dibuja esto?"* (ninguno).
- **`summary()`:** que calculen ellos `oculta_1` (5×32+32 = 192) antes de mirar la tabla.
- **`fit` en vivo** (~30 s): mientras corre, explica lote / paso / época con la tabla.
- **Tabla de comparación:** es el clímax. Lineal ≈ red sin ReLU ≈ 76 %; con ReLU ≈ 89 %.
  Señala la columna *desgaste* (≈ 44 % → ≈ 77 %). Conecta con la celda del colapso lineal.
- **Motores D y E:** lento y rápido, la red dice desgaste a los dos; el lineal dice normal.
  Es la U en un ejemplo concreto.
- **`entrenar.py`:** no lo leas entero. Señala que los pasos 3–4 son exactamente lo que acaban
  de hacer y explica solo lo nuevo (`EarlyStopping`, `evaluate`, `save`).

---

## Preguntas para lanzar

- *"Si un tensor es como un array de NumPy, ¿para qué TensorFlow?"* (GPU y gradientes automáticos)
- *"¿Por qué los pesos son `Variable` y los datos `constant`?"*
- *"Si quito ReLU de todas las capas, ¿qué modelo tengo?"* (uno lineal)
- *"¿Por qué softmax solo en la salida?"* (en ocultas no queremos probabilidades, queremos features)
- *"Un motor sale `[0.34, 0.33, 0.33]`. ¿Lo paramos?"* (la red no sabe; en negocio, revisar)

---

## Errores conceptuales frecuentes

| Lo que dicen | Lo correcto |
|---|---|
| "Más capas siempre es mejor" | Sin activación no suman nada; con activación, más capas necesitan más datos. |
| "La red optimiza el accuracy" | Optimiza la **pérdida** (cross-entropy). El accuracy es solo para leer. |
| "Softmax elige la clase" | Softmax da probabilidades; el `argmax` elige. |
| "Normalizar es opcional" | En redes es casi obligatorio: con horas en 30 000 el entrenamiento no converge bien. |
| "Una neurona artificial funciona como una biológica" | Solo se inspira en ella: es una suma ponderada y una función. |

---

## Si vas mal de tiempo

| Recorta | Ahorras | Qué no se pierde |
|---|---|---|
| Celda de inmutabilidad (bloque 1) | 3 min | Se menciona de palabra. |
| Celda de reducciones/broadcasting | 4 min | Vuelve en la normalización del bloque 5. |
| Gráfico temperatura/vibración (deja solo la U) | 2 min | La U es la prueba central. |
| `red_sin_relu` (entrena solo lineal vs red) | 3 min | La celda del colapso lineal ya lo demostró. |
| Leer `generar_datos.py` (5.1): solo ejecutarlo | 2 min | `argparse` se ve igual en `entrenar.py`. |
| Bloque 5.8 (leer y ejecutar `entrenar.py`) | 5 min | Queda en el README para casa. |

**Lo que NO se recorta:** demo Celsius, colapso lineal, tabla de comparación.

## Si te sobra tiempo

- Reto 1 en vivo: `construir_red(normalizador, capas_ocultas=(8,))` vs `(128, 64, 32)`.
- `python scripts/entrenar.py --capas 64 32 16` y comparar con el notebook.
- Tasa de aprendizaje 0.1 y 0.00001: mostrar las curvas.
