"""
App de predicción de gasto — clase Streamlit (carpeta aislada).

Arranque (desde ESTA carpeta streamlit/):

    streamlit run app.py

El cerebro es el .joblib (ya copiado en modelos/).
Esta app es el mostrador: un humano elige cifras o un cliente y ve el número.

Streamlit ejecuta este archivo de ARRIBA ABAJO en cada clic.
@st.cache_resource evita reler el .joblib en cada rerun.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.carga_modelos import (
    cargar_lookup,
    cargar_meta,
    cargar_paquete,
    features_de_cliente,
)
from src.features_modelo import COLUMNA_TARGET, COLUMNAS_FEATURES
from src.prediccion import predecir

# --- Configuración de la página (una sola vez, al inicio) -------------------
st.set_page_config(
    page_title="Predicción de gasto",
    page_icon="💶",
    layout="wide",
)


# --- Carga: UNA vez, en caché ----------------------------------------------
# cache_resource = para objetos pesados (el modelo). Se guarda en RAM
# entre reruns. Si no, cada slider haría joblib.load otra vez.
@st.cache_resource
def _modelos() -> dict:
    return {
        "sklearn": cargar_paquete("gasto_sklearn.joblib"),
        "mlp": cargar_paquete("gasto_mlp.joblib"),
    }


# cache_data = para datos (el CSV). Si el archivo no cambia, no se relée.
@st.cache_data
def _lookup() -> pd.DataFrame:
    return cargar_lookup()


@st.cache_data
def _meta() -> dict:
    return cargar_meta()


paquetes = _modelos()
lookup = _lookup()
meta = _meta()

ETIQUETAS = {
    "sklearn": "Bosque (Random Forest)",
    "mlp": "Red neuronal (MLP)",
}


# --- Cabecera --------------------------------------------------------------
st.title("Predicción de gasto a 90 días")
st.write(
    "El modelo **ya está entrenado**. Aquí no hay `fit()`. "
    "Tú eliges la ficha del cliente; el `.joblib` responde un número."
)

# --- Barra lateral: qué cerebro usar + ficha técnica -----------------------
# st.sidebar.xxx pinta el widget a la IZQUIERDA. El valor sigue siendo Python.
motor = st.sidebar.radio(
    "Qué modelo usar",
    options=["sklearn", "mlp"],
    format_func=lambda clave: ETIQUETAS[clave],
)
paquete = paquetes[motor]

st.sidebar.markdown("### Ficha del artefacto")
st.sidebar.write(f"Versión: `{meta['version']}`")
st.sidebar.write(f"Corte: `{meta['fecha_corte']}`")
st.sidebar.write(f"Clientes en lookup: **{len(lookup):,}**")
st.sidebar.write(f"MAE de este motor: **{meta[motor]['mae']}** €")
st.sidebar.caption(
    "MAE alto = el modelo no es bueno. Hoy no importa: hoy servimos, no afinamos."
)

# --- Dos pestañas: a mano vs elegir cliente --------------------------------
# st.tabs devuelve tantos objetos como nombres. Cada `with` es una pestaña.
tab_manual, tab_cliente, tab_comparar = st.tabs(
    ["Ficha a mano", "Elegir un cliente", "Comparar los dos modelos"]
)


# --- Pestaña 1: el humano escribe las 4 cifras -----------------------------
with tab_manual:
    st.subheader("Tú mandas las 4 cifras (como el POST de FastAPI)")
    st.caption(
        "La app NO busca en el CSV. Usa exactamente lo que pongas aquí."
    )

    # st.columns(4) parte la fila en 4 columnas. Más cómodo que apilar.
    c1, c2, c3, c4 = st.columns(4)
    num_compras = c1.number_input("num_compras", min_value=0, value=18, step=1)
    gasto_historico = c2.number_input(
        "gasto_historico (€)", min_value=0.0, value=2100.5, step=50.0
    )
    ticket_medio = c3.number_input(
        "ticket_medio (€)", min_value=0.0, value=116.7, step=5.0
    )
    cantidad_total = c4.number_input("cantidad_total", min_value=0, value=40, step=1)

    # st.form agrupa widgets y NO rerunea en cada tecla: solo al pulsar el botón.
    # Así no llamamos a predict() mientras el alumno todavía está tecleando.
    with st.form("form_manual"):
        st.caption("Pulsa para predecir. Hasta entonces el modelo no corre.")
        lanzar = st.form_submit_button("Predecir gasto a 90 días")

    if lanzar:
        valor = predecir(
            paquete,
            num_compras=int(num_compras),
            gasto_historico=float(gasto_historico),
            ticket_medio=float(ticket_medio),
            cantidad_total=int(cantidad_total),
        )
        # st.metric = número grande con etiqueta. El gesto de "KPI".
        st.metric(
            label=f"Valor predicho · {ETIQUETAS[motor]}",
            value=f"{valor:,.2f} €",
        )
        st.success("Eso es inferencia: el `.joblib` ya estaba en memoria.")


# --- Pestaña 2: elegir un cliente del lookup -------------------------------
with tab_cliente:
    st.subheader("Eliges el id; las 4 cifras salen del CSV")
    st.caption(
        "Igual que `GET /predecir/cliente/42` en FastAPI. "
        "No es producción (el CSV está congelado), pero para clase evita teclear."
    )

    # selectbox: lista desplegable. lookup.index son los cliente_id.
    # Tomamos una muestra para que el desplegable no tenga 9.996 filas lentas.
    # El alumno puede escribir el id a mano en el number_input de al lado.
    col_sel, col_id = st.columns([2, 1])
    ids_muestra = list(lookup.index[:200])
    elegido_lista = col_sel.selectbox(
        "Clientes (primeros 200 del lookup)",
        options=ids_muestra,
        index=min(7, len(ids_muestra) - 1),  # el 8 suele ser un cliente "grande"
    )
    elegido_manual = col_id.number_input(
        "O escribe un cliente_id",
        min_value=int(lookup.index.min()),
        max_value=int(lookup.index.max()),
        value=int(elegido_lista),
        step=1,
    )
    cliente_id = int(elegido_manual)

    if cliente_id not in lookup.index:
        st.error(f"El cliente {cliente_id} no está en el lookup.")
    else:
        cifras = features_de_cliente(lookup, cliente_id)
        real = float(lookup.loc[cliente_id, COLUMNA_TARGET])

        st.write("Ficha del pasado (lo que VE el modelo):")
        st.dataframe(
            pd.DataFrame([cifras], columns=COLUMNAS_FEATURES),
            hide_index=True,
            use_container_width=True,
        )

        valor = predecir(paquete, **cifras)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicho", f"{valor:,.2f} €")
        m2.metric("Real a 90 días (target)", f"{real:,.2f} €")
        m3.metric("Error |pred − real|", f"{abs(valor - real):,.2f} €")
        st.caption(
            "El target NO entra al modelo. Lo enseñamos para comparar en clase."
        )


# --- Pestaña 3: los dos cerebros con la misma ficha ------------------------
with tab_comparar:
    st.subheader("Misma ficha, dos modelos")
    st.caption("El contrato (las 4 columnas) no cambia. Cambia el archivo .joblib.")

    cliente_cmp = st.number_input(
        "cliente_id para comparar",
        min_value=int(lookup.index.min()),
        max_value=int(lookup.index.max()),
        value=8,
        step=1,
        key="cliente_cmp",  # key= evita chocar con el number_input de la otra pestaña
    )
    if cliente_cmp not in lookup.index:
        st.error(f"El cliente {cliente_cmp} no está en el lookup.")
    else:
        cifras = features_de_cliente(lookup, cliente_cmp)
        real = float(lookup.loc[cliente_cmp, COLUMNA_TARGET])
        v_bosque = predecir(paquetes["sklearn"], **cifras)
        v_red = predecir(paquetes["mlp"], **cifras)

        tabla = pd.DataFrame(
            {
                "origen": ["Bosque", "Red (MLP)", "Real (90 días)"],
                "euros": [v_bosque, v_red, real],
            }
        )
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        # st.bar_chart espera un DataFrame con índice = etiquetas del eje X.
        st.bar_chart(tabla.set_index("origen"))
        st.caption(
            "Si los dos modelos discrepan mucho, en producción elegirías uno "
            "(o un ensemble). Hoy solo se ve que se sirven igual."
        )
