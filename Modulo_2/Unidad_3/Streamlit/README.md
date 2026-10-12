# Clase Streamlit — mostrador del modelo de gasto

Carpeta **aislada**. No importa el `src/` ni el `api/` del proyecto padre.
Todo lo que hace falta está aquí (incluido el `.joblib`).

```
ventas.csv (copia)  →  ya se usó para fabricar el .joblib
modelos/*.joblib    →  el cerebro (ya copiado)
app.py              →  un humano ve y toca la predicción
```

## Cómo se da la clase

1. Terminal: `pip install -r requirements.txt` (desde **esta** carpeta).
2. Notebook: `notebooks/01_streamlit_prediccion.ipynb` (explica y predice sin UI).
3. `streamlit run app_hola.py` — primer contacto (widgets).
4. `streamlit run app.py` — predicción con bosque y red.

## Archivos

| Archivo | Qué es |
|---|---|
| `notebooks/01_streamlit_prediccion.ipynb` | Guion de las 2 horas |
| `app_hola.py` | Widgets, sin modelo |
| `app.py` | App de predicción |
| `src/carga_modelos.py` | `joblib.load` + lookup |
| `src/prediccion.py` | Arma la fila y llama a `predict()` |
| `src/features_modelo.py` | Las 4 columnas y el target |
| `scripts/entrenar_modelos.py` | Solo si borras `modelos/` |
| `modelos/` | `.joblib`, fichas, `meta.json` |
| `data/raw/ventas.csv` | Copia; solo para reentrenar |
| `docs/GUIA_DOCENTE.md` | Minutado |
| `docs/LIBRO_TECNICO_STREAMLIT.pdf` | Libreto teórico (teoría oficial + esta clase) |

## Arranque

```powershell
conda activate py_env
cd C:\Programacion\DataScienceIA\Docente_Dev_Senior_code\files\persisitencia\streamlit
pip install -r requirements.txt
streamlit run app.py
```
