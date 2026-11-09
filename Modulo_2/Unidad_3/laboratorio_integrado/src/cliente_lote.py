"""
Teléfono HTTP del orquestador. Solo habla con la API: no usa Spark ni SQLite.

Un lote es un POST a /predecir/lote/{motor}. El body es el de EntradaLote:

    {"clientes": [ficha, ficha, ...]}

La API acepta como mucho 500 fichas por llamada. Si llegan más, este
módulo parte la lista y junta las predicciones en una sola respuesta.
"""

from __future__ import annotations

import httpx

URL_BASE = "http://127.0.0.1:8000"
TIMEOUT = 60.0
TAMANO_LOTE = 500
MOTORES = ("sklearn", "red")


def _ficha(fila: dict) -> dict:
    """La ficha que pide EntradaPrediccion, con números que JSON entiende."""
    return {
        "cliente_id": int(fila["cliente_id"]),
        "num_compras": int(fila["num_compras"]),
        "gasto_historico": float(fila["gasto_historico"]),
        "ticket_medio": float(fila["ticket_medio"]),
        "cantidad_total": int(fila["cantidad_total"]),
    }


def _post_lote(fichas: list[dict], motor: str) -> dict:
    """Una llamada. Devuelve estado y, si salió bien, el JSON de SalidaLote."""
    try:
        respuesta = httpx.post(
            f"{URL_BASE}/predecir/lote/{motor}",
            json={"clientes": fichas},
            timeout=TIMEOUT,
        )
    except httpx.CotError:
        return {
            "estado": "error de conexión",
            "mensaje": "El backend no está en el puerto 8000.",
        }
    except httpx.TimeoutException:
        return {
            "estado": "otro_error",
            "mensaje": "La API tardó más de lo previsto.",
        }

    if respuesta.status_code == 200:
        return {"estado": "exito", "datos": respuesta.json()}
    if respuesta.status_code == 422:
        return {"estado": "error de validación", "mensaje": respuesta.json()}
    return {
        "estado": "error_otro",
        "mensaje": f"HTTP {respuesta.status_code}: {respuesta.text}",
    }


def pedir_lote(clientes: list[dict], motor: str = "sklearn") -> dict:
    """
    Manda las fichas y devuelve las predicciones.

    clientes es una lista de dicts con cliente_id y las cuatro features.
    motor es la ruta de la API: "sklearn" o "red".
    """
    if motor not in MOTORES:
        return {
            "estado": "error de validación",
            "mensaje": "motor tiene que ser 'sklearn' o 'red'.",
        }
    if not clientes:
        return {
            "estado": "error de validación",
            "mensaje": "El lote no trae clientes.",
        }

    fichas = [_ficha(fila) for fila in clientes]
    predicciones: list[dict] = []
    for inicio in range(0, len(fichas), TAMANO_LOTE):
        trozo = fichas[inicio : inicio + TAMANO_LOTE]
        respuesta = _post_lote(trozo, motor)
        if respuesta["estado"] != "exito":
            return respuesta
        predicciones.extend(respuesta["datos"]["predicciones"])

    return {
        "estado": "exito",
        "datos": {"n": len(predicciones), "predicciones": predicciones},
    }
