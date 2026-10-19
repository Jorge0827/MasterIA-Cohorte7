from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
    
import streamlit as st
from frontend.cliente_api import pedir_por_cliente, pedir_prediccion, pedir_salud

st.set_page_config(page_title="Front -> API", layout="centered")

#title / write
st.title("Front-end hablando con el back-end")
st.write(
    "Envía un JSON a FastAPIP y pinta **carga / error / éxito"
)

salud = pedir_salud()
if salud["estado"] == "error de conexión":
    st.error(salud["mensaje"])
    st.stop()
    
st.sidebar.success("Backend en :8000 " + salud["datos"]["estado"])


motor = st.sidebar.radio("Motor (ruta de la api)", ["sklearn", "red"])
modo = st.radio("Cómo envías los datos", ["POST a mano (4 cifras)", "GET por cliente_id"])


if modo.startswith("POST"):
    c1, c2 = st.columns(2)
    cliente_id = c1.number_input("cliente_id", min_value=1, value=8, step=1)
    num_compras = c2.number_input("num_compras", min_value=0, value=18, step=1)
    gasto_historico = st.number_input("gasto_historico", min_value=0.0, value=2100.5, step=50.0)
    ticket_medio = st.number_input("ticket_medio", min_value=0.0, value=116.7, step=5.0)
    cantidad_total = st.number_input("cantidad_total", min_value=0, value=40, step=1)

    # Mismos nombres que EntradaPrediccion. Este dict es el body del POST.
    cuerpo = {
        "cliente_id": int(cliente_id),
        "num_compras": int(num_compras),
        "gasto_historico": float(gasto_historico),
        "ticket_medio": float(ticket_medio),
        "cantidad_total": int(cantidad_total),
    }

else:
    cliente_id = st.number_input("Cliente_id", min_value=1, value=8, step=1)
    cuerpo = None

with st.form("enviar"):
    enviar = st.form_submit_button("Enviar al backend")
    
if enviar:
    with st.spinner("Hablando con la API......"):
        if cuerpo is not None:
            resultado = pedir_prediccion(cuerpo, motor=motor)
        
        else:
            resultado = pedir_por_cliente(int(cliente_id), motor=motor)
    
    estado = resultado["estado"]
    

#------------------------------------
#Exito ( 200 )
    if estado == "exito":
        datos = resultado["datos"]
        st.success("Repsuesta 200: El backend predijo")
        st.metric("valor predicho", f"{datos['valor_predicho']:,.2f} euros")
        st.caption(
            f"modelo={datos['modelo']} versión={datos['version']}"
            f"latencia={datos['tiempo_ms']} ms"
        )
        

    elif estado == "error de conexión":
        st.error(resultado["mensaje"])
    elif estado == "error de validación":
        st.error("422: El backend rechazó el JSON (pydantic). El modelo no corrió")
        st.json(resultado["mensaje"])
    elif estado == "error_no_encontrado":
        st.warning("404: ese cliente no está en el lookup")
        st.write(resultado["mensaje"])
    else:
        st.error(resultado["mensaje"])
    
