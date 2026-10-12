"""
Primera app Streamlit: widgets, sin modelo.

Arranque (desde la carpeta streamlit/):

    streamlit run app_hola.py

Qué hay que ver en clase:
    1. Se abre el navegador solo (puerto 8501).
    2. Cada vez que mueves un slider, Streamlit VUELVE A EJECUTAR
       este archivo entero, de arriba abajo.
    3. Todavía no hay .joblib. Hoy solo tocamos la interfaz.
"""

# streamlit se importa como st por convenio (como pandas → pd).
# st.title, st.slider, st.button... son FUNCIONES que pintan un widget
# y A LA VEZ devuelven el valor que el usuario acaba de elegir.
import streamlit as st

# set_page_config SOLO puede ir una vez y al PRINCIPIO del script.
# page_title = pestaña del navegador. layout="centered" = columna estrecha.
st.set_page_config(page_title="Hola Streamlit", layout="centered")

st.title("Hola, Streamlit")
st.write(
    "Cada vez que tocas un widget, Streamlit **vuelve a ejecutar TODO "
    "este archivo**. No hay un `main()` que espere: el script corre entero."
)

# text_input pinta una caja y DEVUELVE el texto (str).
# value= es el valor inicial la primera vez que se abre la página.
nombre = st.text_input("Tu nombre", value="Jorge")

# number_input pinta un número con flechas. min_value evita negativos.
# value=10 es el arranque. Devuelve int (o float si step es decimal).
compras = st.number_input("Número de compras", min_value=0, value=10, step=1)

# slider pinta una barra. Los tres números: mínimo, máximo, valor inicial.
gasto = st.slider("Gasto histórico (€)", min_value=0, max_value=20000, value=2000)

st.divider()  # línea horizontal, solo estética

# st.button devuelve True SOLO en el clic que lo pulsó.
# En el siguiente rerun (si mueves el slider) vuelve a False.
# Por eso el saludo no se queda "pegado" a menos que uses session_state
# (eso lo vemos en app.py).
if st.button("Saludar"):
    # st.success / st.info / st.warning / st.error = cajas de color.
    st.success(
        f"Hola {nombre}: {compras} compras y {gasto:,} € de historial."
    )
    st.caption("Esto todavía NO es una predicción. Es solo la interfaz.")
else:
    st.info("Pulsa **Saludar**. Hasta entonces no hay acción.")
