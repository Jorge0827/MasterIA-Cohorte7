# Guía docente — Streamlit para servir el modelo

**Duración:** 100–120 minutos  
**Carpeta:** `streamlit/` (aislada: no importa `api/` ni el `src/` del padre)  
**Notebook:** `notebooks/01_streamlit_prediccion.ipynb`  
**App de verdad:** `app.py` (se lanza con `streamlit run`, no con Run del notebook)  
**Libreto teórico (PDF):** `docs/LIBRO_TECNICO_STREAMLIT.pdf`

---

## Dónde vive el modelo (dilo al minuto 5)

No se descarga. Ya está copiado en `modelos/`:

```
.joblib en disco  →  Streamlit lo carga UNA vez (cache)
                  →  cada clic en la web solo hace predict()
```

- **Bosque:** `modelos/gasto_sklearn.joblib`
- **Red:** `modelos/gasto_mlp.joblib`
- **Fichas:** `modelos/features_clientes.csv`

Hoy no entrenamos (salvo que falten archivos). Hoy **dibujamos el mostrador**.

---

## Notebook o Python “normal”

**Los dos.** El notebook **explica** y predice sin interfaz. La app es un
`.py` porque Streamlit **no se sirve desde un `.ipynb`**.

En clase:

1. Abres el notebook. Inventario + `joblib.load` + `predict()`.
2. `streamlit run app_hola.py` — widgets, sin modelo.
3. `streamlit run app.py` — predicción con los dos motores.

---

## Antes de la clase

- [ ] `conda activate py_env`
- [ ] `cd streamlit` y `pip install -r requirements.txt`
- [ ] En `modelos/` están los `.joblib`, el CSV y `meta.json`.
- [ ] **No** hace falta FastAPI ni Uvicorn hoy. Es otra puerta al mismo cerebro.

---

## Minutado

| Bloque | Tema | Min |
|---|---|---|
| 0 | FastAPI vs Streamlit (máquina vs humano) | 12 |
| 1 | El `.joblib` ya está: cargar y predecir sin UI | 20 |
| 2 | Widgets y el rerun (app_hola) | 20 |
| 3 | Caché, formulario, `predict` (código de app.py) | 20 |
| 4 | App completa en el navegador | 20 |
| 5 | Cierre | 5 |

---

## Qué no recortes

- El dibujo **API = camarero para programas / Streamlit = mostrador para personas**.
- `@st.cache_resource` = no reler el joblib en cada clic.
- `st.form` = no predecir en cada tecla.
- Comparar bosque vs red con la **misma** ficha.
- Un cliente con target 0 (el 9) y uno grande (el 8).

## Si vas mal de tiempo

Quita la pestaña “Comparar” y `app_hola.py`. Entra directo a `app.py`.

---

## Frases

- *“FastAPI habla JSON. Streamlit habla sliders.”*
- *“El cerebro es el mismo archivo. Cambia la puerta.”*
- *“Cada clic rerunea el script. Por eso cacheamos el modelo.”*
- *“El CSV de clientes está congelado: en producción la ficha vendría viva.”*
