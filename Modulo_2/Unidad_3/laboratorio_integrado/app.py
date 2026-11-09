from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
    
    
from pandas.core.common import random_state
import streamlit as st

from src.carga_modelos import cargar_meta, cargar_paquete, cargar_tabla_evaluacion
from src.evaluacion import(
    construir_monitoreo,
    peores_clientes,
    resumir_metricas,
    error_por_tramo,
    histograma_residual,
    particion_oficial,
)

from src.historial import conectar, crear_tabla, leer_historial, resumir_lotes, vaciar
from src.orquestador import correr
from src.reportes import csv_bytes, informe_bytes, informe_markdown


st.set_page_config(page_title="Monitoreo de modelos", layout="wide")

@st.cache_data
def _tablero(solo_holdout: bool) -> tuple:
    
    tabla = cargar_tabla_evaluacion()
    if solo_holdout:
        _, tabla = particion_oficial(tabla)
    sklearn = cargar_paquete("gasto_sklearn.joblib")
    mlp = cargar_paquete("gasto_mlp.joblib")
    return construir_monitoreo(tabla, sklearn, mlp), cargar_meta()

st.title("Dashboard de monitoreo de IA")
st.write(
    "No es un mostrador de una predicción. Es el tablero real "
    "**Rendimiento**: ¿Cuánto falla el modelo, en quién y se puede bajar el informe?"
)

solo_holdout = st.sidebar.checkbox(
    "Solo holdout (20 % oficial)",
    value=True,
    help="Marcado = las mismas filas con las que se midió el meta json."
    "Descamarcado = todos los clientes (el MAE baja: estás viendo el entrenamiento)"
)


motor = st.sidebar.radio("Motor que detalllas en los graficos", ["sklearn", "red"])
motor_col = "sklearn" if motor == "sklearn" else "mlp"


max_puntos = st.sidebar.slider("Puntos en el gráfico rel vs predicho", 200, 2000, 800, 100)

st.sidebar.caption("El tablero lee modelos/. Historial llama a la API y guarda en SQLite.")

monitoreo, meta = _tablero(solo_holdout)
resumen = resumir_metricas(monitoreo)
sk = resumen["sklearn"]
mlp = resumen["mlp"]

# --- FIla de KPIs

st.subheader("Rendimiento")

c1, c2, c3, c4 = st.columns(4)

c1.metric("CLientes evaluados", f"{sk['n']:,}")
c2.metric(
    "MAE Random Forest",
    f"{sk['mae']:,.2f} €",
    delta=f"{sk['mae'] - meta['sklearn']['mae']:+.2f} vs meta"
)
c3.metric(
    "MAE red (MLP)",
    f"{mlp['mae']:,.2f} €",
    delta=f"{mlp['mae'] - meta['mlp']['mae']:+.2f} vs meta"
)
c4.metric("R2 mejor", f"{max(sk['r2'], mlp['r2']):.3f}")


st.info(
    f"meta.json (holdout del entreanmiento):"
    f"RF MAE={meta['sklearn']['mae']} R2={meta['sklearn']['r2']} -"
    f"MLP MAE={meta['mlp']['mae']} R2={meta['mlp']['r2']}."
    "Un R2 de 0.05 significa que: apenas mejoramos a predecir la media"
)

tab_graf, tab_tramos, tab_peores, tab_descarga, tab_historial = st.tabs(
    ["Real vs Predicho", "Errores por tramo", "Peores clientes", "Descargar reporte", "Historial"]
)

with tab_graf:
    st.write(
        "Cada punto es un cliente. La diagonal (real = predicho) sería el "
        "modelo perfecto: todos los puntos sobre y = x. "
        "Si la nube es una mancha horizontal, el modelo casi no usa las "
        "features: predice más o menos lo mismo da igual al cliente"
    )
    
    n_muestra = min(int(max_puntos), len(monitoreo))
    muestra = monitoreo.sample(n=n_muestra, random_state=0)
    
    st.scatter_chart(muestra, x="y_real", y=["pred_sklearn", "pred_mlp"])
    st.caption("Eje X = gasto real a 90 días (€). Eje Y = predicho (los dos motores)")
    
    st.write(
        f"Residuales del motor **(motor)** (predicho - real)."
        "Un histograma centrado en 0 es un modelo que falla arriba y abajo "
        "por igual. Una joroba a la derecha = se pasa. Una cola larguísima "
        "= pocos fallo enormes ( por eso el RMSE es mayor que el MAE)."
    )
    
    st.bar_chart(histograma_residual(monitoreo, motor=motor_col), y="clientes")
    st.caption(
        f"Sesgo {motor}: {resumen[motor_col]['sesgo']:+.2f} € - "
        f"RMSE {resumen[motor_col]['rmse']:,.2f} €"
    )
    
with tab_tramos:
    st.write(
        "MAE por cuartil de gasto_hisotrico. Cuartil = 4 grupos con "
        "el mismo numero de clientes (q1 los que menos gastaron en el pasado,) "
        " Q4 los que más). Si Q4 faññta mucho más que q1, el modelo no vale "
        "igual para las cuentas VIP: un único MAE de 10.17 € te lo escondía "
    )

    t1, t2 = st.columns(2)
    t1.write("Random Forest")
    t1.bar_chart(error_por_tramo(monitoreo, motor="sklearn"), y="mae")
    t2.write("MLP (red)")
    t2.bar_chart(error_por_tramo(monitoreo, motor="mlp"), y="mae")

    st.dataframe(
        error_por_tramo(monitoreo, motor=motor_col),
        use_container_width=True,
    )

with tab_peores:
    st.write(
        "Los 10 clientes con mayor error absoluto. "
        "Columnas: cliente_id, gasto_historico, y_real, "
        "predicción del motor, error (predicho − real) y error absoluto."
    )
    p1, p2 = st.columns(2)
    with p1:
        st.write("Random Forest")
        st.dataframe(
            peores_clientes(monitoreo, motor="sklearn"),
            hide_index=True,
            use_container_width=True,
        )
    with p2:
        st.write("MLP (red)")
        st.dataframe(
            peores_clientes(monitoreo, motor="mlp"),
            hide_index=True,
            use_container_width=True,
        )

with tab_descarga:
    informe = informe_markdown(monitoreo, meta)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Descargar monitoreo (CSV)",
            data=csv_bytes(monitoreo),
            file_name="monitoreo.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Columnas: cliente_id, num_compras, gasto_historico, ticket_medio, "
            "cantidad_total, y_real, pred_sklearn, pred_mlp, error_sklearn, "
            "error_mlp, abs_sklearn, abs_mlp."
        )
    with d2:
        st.download_button(
            "Descargar informe (Markdown)",
            data=informe_bytes(informe),
            file_name="informe_monitoreo.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.caption("El mismo texto que se abre debajo: ganador por MAE, comparación con meta.json y glosario.")

    with st.expander("Informe Markdown"):
        st.markdown(informe)

with tab_historial:
    st.write(
        "Spark arma las fichas, el teléfono las manda a la API "
        "y estos métodos las dejan en data/historial.sqlite."
    )
    motor_api = st.radio("Motor de la API", ["sklearn", "red"], horizontal=True)
    motor_features = st.radio(
        "Cómo se arman las features",
        ["spark", "pandas"],
        horizontal=True,
    )
    n_clientes = st.slider("Clientes del lote", min_value=1, max_value=500, value=50)
    boton_correr, boton_vaciar = st.columns(2)
    correr_lote = boton_correr.button("Correr", use_container_width=True)
    vaciar_lote = boton_vaciar.button("Vaciar historial", use_container_width=True)

    if correr_lote:
        with st.spinner("Spark, HTTP y SQLite..."):
            resultado = correr(
                motor_modelo=motor_api,
                n_clientes=int(n_clientes),
                motor_features=motor_features,
            )
        if resultado["estado"] == "error":
            st.error(f"{resultado['paso']}: {resultado['mensaje']}")
        else:
            st.success(
                f"Lote {resultado['lote_id']} guardado con {resultado['motor_features']}. "
                f"API: {resultado['motor_modelo']}."
            )

    if vaciar_lote:
        conexion = conectar()
        try:
            borradas = vaciar(conexion)
        finally:
            conexion.close()
        st.success(f"Historial vacío. Filas borradas: {borradas}.")

    crear_tabla()
    conexion = conectar()
    try:
        filas = leer_historial(conexion)
        lotes = resumir_lotes(conexion)
    finally:
        conexion.close()

    st.subheader("Lotes")
    if lotes:
        st.dataframe(lotes, hide_index=True, use_container_width=True)
    else:
        st.info("El archivo ya está. Cuando pulses Correr, aquí salen los lotes.")

    st.subheader("Filas")
    if filas:
        st.dataframe(filas, hide_index=True, use_container_width=True)
    

    
