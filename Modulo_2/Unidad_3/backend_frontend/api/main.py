"""
Servidor de inferencia: FastAPI carga los modelos y responde predicciones.

Arranque (desde la RAIZ del proyecto, entorno py_env):

    uvicorn api.main:app --reload --port 8000

Documentacion interactiva (la genera FastAPI sola):

    http://localhost:8000/docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.carga_modelos import (
    ErrorModeloNoEncontrado,
    cargar_lookup,
    cargar_meta,
    cargar_paquete,
    features_de_cliente,
)
from api.esquemas import EntradaPrediccion, SalidaPrediccion
from src.features_modelo import COLUMNAS_FEATURES

# Estado global del proceso: se rellena al arrancar y se reutiliza.
# Un servidor NO debe hacer joblib.load en cada peticion.
ESTADO: dict = {
    "sklearn": None,
    "mlp": None,
    "lookup": None,
    "meta": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta UNA vez al encender (yield) y al apagar.

    Aqui cargamos los .joblib. Si faltan, el servidor ni arranca:
    mejor un error claro a las 8:00 que 500 misteriosos en cada POST.
    """
    try:
        ESTADO["sklearn"] = cargar_paquete("gasto_sklearn.joblib")
        ESTADO["mlp"] = cargar_paquete("gasto_mlp.joblib")
        ESTADO["lookup"] = cargar_lookup()
        ESTADO["meta"] = cargar_meta()
    except ErrorModeloNoEncontrado as error:
        raise RuntimeError(str(error)) from error
    yield
    ESTADO.clear()


app = FastAPI(
    title="API de prediccion de gasto",
    description=(
        "Unidad 2 · Clase 3. Carga un Random Forest y una red (MLP) "
        "entrenados con los CSV de la plataforma de ventas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["salud"])
def raiz() -> dict:
    """Hola del servidor. Sirve para comprobar que uvicorn esta vivo."""
    return {
        "servicio": "prediccion_gasto",
        "docs": "/docs",
        "salud": "/salud",
    }


@app.get("/salud", tags=["salud"])
def salud() -> dict:
    """
    Health check: lo usa un balanceador o el docente para ver
    si los modelos estan en memoria.
    """
    _asegurar_modelos()
    return {
        "estado": "ok",
        "sklearn": ESTADO["sklearn"]["version"] if ESTADO.get("sklearn") else None,
        "mlp": ESTADO["mlp"]["version"] if ESTADO.get("mlp") else None,
        "clientes_en_lookup": int(len(ESTADO["lookup"])) if ESTADO.get("lookup") is not None else 0,
    }


@app.get("/modelos", tags=["salud"])
def modelos() -> dict:
    """Que hay cargado: metricas del entrenamiento y nombres de features."""
    _asegurar_modelos()
    if not ESTADO.get("meta"):
        raise HTTPException(503, "Modelos no cargados.")
    return ESTADO["meta"]


def _asegurar_modelos() -> None:
    """Si el lifespan no corrio (TestClient sin 'with'), carga ahora."""
    if ESTADO.get("sklearn") is not None:
        return
    ESTADO["sklearn"] = cargar_paquete("gasto_sklearn.joblib")
    ESTADO["mlp"] = cargar_paquete("gasto_mlp.joblib")
    ESTADO["lookup"] = cargar_lookup()
    ESTADO["meta"] = cargar_meta()


def _predecir(paquete: dict, entrada: EntradaPrediccion, etiqueta: str) -> SalidaPrediccion:
    """
    Paso comun de los dos POST:

    1. Pydantic YA valido el JSON (si no, no llegamos aqui).
    2. Armamos un DataFrame de 1 fila, en el ORDEN que espera el modelo.
    3. model.predict(...)  ->  numpy array
    4. Devolvemos el contrato de salida.
    """
    inicio = time.perf_counter()
    # Una fila, mismas columnas y mismo orden que en el entrenamiento.
    x = pd.DataFrame(
        [
            {
                "num_compras": entrada.num_compras,
                "gasto_historico": entrada.gasto_historico,
                "ticket_medio": entrada.ticket_medio,
                "cantidad_total": entrada.cantidad_total,
            }
        ],
        columns=COLUMNAS_FEATURES,
    )
    # predict devuelve un array; [0] es el unico valor (una sola fila).
    valor = float(paquete["modelo"].predict(x)[0])
    ms = int((time.perf_counter() - inicio) * 1000)
    return SalidaPrediccion(
        cliente_id=entrada.cliente_id,
        valor_predicho=round(valor, 2),
        modelo=etiqueta,
        version=paquete["version"],
        tiempo_ms=ms,
    )


@app.post("/predecir/sklearn", response_model=SalidaPrediccion, tags=["prediccion"])
def predecir_sklearn(entrada: EntradaPrediccion) -> SalidaPrediccion:
    """Inferencia con el Random Forest. El body lo valida Pydantic."""
    _asegurar_modelos()
    return _predecir(ESTADO["sklearn"], entrada, "gasto_sklearn")


@app.post("/predecir/red", response_model=SalidaPrediccion, tags=["prediccion"])
def predecir_red(entrada: EntradaPrediccion) -> SalidaPrediccion:
    """Inferencia con la red neuronal (MLP). Mismo contrato de entrada."""
    _asegurar_modelos()
    return _predecir(ESTADO["mlp"], entrada, "gasto_mlp")


@app.get("/predecir/cliente/{cliente_id}", response_model=SalidaPrediccion, tags=["prediccion"])
def predecir_por_id(
    cliente_id: int,
    motor: Literal["sklearn", "red"] = "sklearn",
) -> SalidaPrediccion:
    """
    Atajo de clase: solo pasas el id. Las features salen del CSV de lookup
    que genero el script de entrenamiento (no se recalculan en vivo).
    """
    _asegurar_modelos()
    lookup = ESTADO["lookup"]
    if cliente_id not in lookup.index:
        raise HTTPException(
            404,
            f"El cliente {cliente_id} no esta en el lookup. "
            "Usa un id que haya comprado antes de 2025-06-30.",
        )
    nums = features_de_cliente(lookup, cliente_id)
    entrada = EntradaPrediccion(cliente_id=cliente_id, **nums)
    paquete = ESTADO["sklearn"] if motor == "sklearn" else ESTADO["mlp"]
    etiqueta = "gasto_sklearn" if motor == "sklearn" else "gasto_mlp"
    return _predecir(paquete, entrada, etiqueta)


@app.exception_handler(ErrorModeloNoEncontrado)
async def _sin_modelo(_, error: ErrorModeloNoEncontrado) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})
