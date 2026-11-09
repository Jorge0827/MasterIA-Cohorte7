"""
Pegamento del laboratorio. `correr` hace tres cosas, en este orden:

    1. Spark arma una fila por cliente (si no arranca, la tabla sale de pandas).
    2. El teléfono HTTP (`cliente_lote`) manda ese lote a la API.
    3. SQLite guarda las predicciones en el historial.

La página de Streamlit es un botón que llama a `correr`.
Desde la raíz del laboratorio:

    streamlit run src/orquestador.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.cliente_lote import pedir_lote
from src.historial import conectar, crear_tabla, insertar_lote, leer_historial, resumir_lotes
from src.spark_features import armar_features


def correr(
    motor_modelo: str = "sklearn",
    n_clientes: int = 50,
    motor_features: str = "spark",
) -> dict:
    """
    Spark, HTTP y SQLite en una sola llamada.

    motor_modelo es la ruta de la API ("sklearn" o "red").
    motor_features es quién arma la tabla ("spark" o "pandas").
    """
    try:
        tabla, usado = armar_features(motor=motor_features, n_clientes=n_clientes)
    except (FileNotFoundError, ValueError) as error:
        return {"estado": "error", "paso": "features", "mensaje": str(error)}

    fichas = tabla.to_dict(orient="records")
    respuesta = pedir_lote(fichas, motor=motor_modelo)
    if respuesta["estado"] != "exito":
        return {
            "estado": "error",
            "paso": "http",
            "mensaje": respuesta.get("mensaje", respuesta["estado"]),
            "motor_features": usado,
            "n_clientes": len(fichas),
        }

    crear_tabla()
    conexion = conectar()
    try:
        lote_id = insertar_lote(
            conexion,
            respuesta["datos"]["predicciones"],
            motor_modelo,
        )
        return {
            "estado": "exito",
            "motor_features": usado,
            "motor_modelo": motor_modelo,
            "n_clientes": len(fichas),
            "n_predicciones": respuesta["datos"]["n"],
            "lote_id": lote_id,
            "filas": leer_historial(conexion, lote_id=lote_id),
            "resumen": resumir_lotes(conexion),
        }
    finally:
        conexion.close()


def pagina() -> None:
    """Botón Correr. Streamlit pinta lo que devuelve `correr`."""
    import streamlit as st

    st.set_page_config(page_title="Correr laboratorio", layout="centered")
    st.title("Correr")
    st.write(
        "Spark resume las ventas, el teléfono manda el lote por HTTP "
        "y SQLite guarda el historial."
    )

    motor_modelo = st.radio("Motor de la API", ["sklearn", "red"], horizontal=True)
    motor_features = st.radio("Cómo se arman las features", ["spark", "pandas"], horizontal=True)
    n_clientes = st.slider("Clientes del lote", min_value=1, max_value=500, value=50)

    if st.button("Correr"):
        with st.spinner("Spark, HTTP y SQLite..."):
            resultado = correr(
                motor_modelo=motor_modelo,
                n_clientes=int(n_clientes),
                motor_features=motor_features,
            )

        if resultado["estado"] == "error":
            st.error(f"{resultado['paso']}: {resultado['mensaje']}")
            return

        st.success(
            f"Lote {resultado['lote_id']} guardado. "
            f"Features con {resultado['motor_features']}. "
            f"API: {resultado['motor_modelo']}."
        )
        st.metric("Predicciones", f"{resultado['n_predicciones']}")
        st.dataframe(resultado["filas"], hide_index=True, use_container_width=True)
        with st.expander("Resumen de lotes en SQLite"):
            st.dataframe(resultado["resumen"], hide_index=True, use_container_width=True)


if __name__ == "__main__":
    pagina()
