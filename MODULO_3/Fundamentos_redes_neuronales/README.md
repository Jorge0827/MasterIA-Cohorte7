# Unidad 1 · Clase 1 — Fundamentos de Redes Neuronales con TensorFlow

Carpeta **aislada**. No usa nada de las clases anteriores: dataset nuevo,
problema nuevo.

**Problema:** mantenimiento predictivo. Cinco sensores de un motor industrial
(temperatura, vibración, rpm, corriente, horas de uso) -> la red dice si el motor
está **normal**, con **desgaste** o en **falla**.

```
tensor -> neurona (w·x + b) -> activación (ReLU / Softmax) -> capa densa (X @ W + b) -> red con Keras
```

## Estructura

```
redes_neuronales/
├── notebooks/
│   └── 01_fundamentos_redes_neuronales.ipynb   # la clase: explorar y explicar
├── src/
│   ├── datos.py      # genera el dataset de motores y lo divide (train / val / test)
│   └── modelo.py     # normalización, red con ReLU, modelo lineal, compilación
├── scripts/
│   ├── generar_datos.py   # crea data/sensores_motores.csv
│   └── entrenar.py        # entrena desde la terminal y guarda modelos/*.keras
├── data/       # CSV generado (no se versiona)
├── modelos/    # modelos entrenados (no se versionan)
└── docs/
    └── GUIA_DOCENTE.md    # minutado de 120 min, guion, preguntas y recortes
```

**Por qué notebook + `.py`:** el notebook es para explorar, ver tensores y gráficos.
El código que se queda (datos, modelo, entrenamiento) vive en `.py`, como en un
proyecto real. El bloque 5 del notebook importa `src/` y el script hace lo mismo
desde la terminal.

## Arranque

```powershell
conda activate py_env
cd C:\Programacion\DataScienceIA\Docente_Dev_Senior_code\files\persisitencia\redes_neuronales
pip install -r requirements.txt

python scripts/generar_datos.py          # 6 000 lecturas, semilla fija
jupyter lab notebooks/01_fundamentos_redes_neuronales.ipynb
```

Desde la terminal (bloque 5.8):

```powershell
python scripts/entrenar.py               # red 32-16 ReLU  -> ~88-89 % en test
python scripts/entrenar.py --lineal      # sin capas ocultas -> ~76 % en test
python scripts/entrenar.py --capas 64 32 16
```

## Resultado que se demuestra en clase

| Modelo | Acierto en test | Acierto en "desgaste" |
|---|---|---|
| Lineal (sin capas ocultas) | ~76 % | ~45 % |
| Red 32-16 **sin** ReLU | ~76 % | ~43 % |
| Red 32-16 **con** ReLU | ~89 % | ~77 % |

Apilar capas sin activación no aporta nada. Con ReLU, la red descubre sola
la relación en U de las rpm (malo ir muy lento y muy rápido).

## Problema conocido de instalación

Si `import tensorflow` falla con
`VersionError: Detected incompatible Protobuf Gencode/Runtime versions`:

```powershell
pip install "protobuf>=6.31.1,<7"
pip install -U streamlit     # streamlit viejo pide protobuf < 6
```

En Windows nativo TensorFlow ≥ 2.11 no usa GPU. Para esta clase la CPU sobra
(cada entrenamiento tarda menos de un minuto).
