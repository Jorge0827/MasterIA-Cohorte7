"""
Contratos de entrada y salida de la API (Pydantic).

Pydantic revisa el JSON ANTES de que llegue al modelo:

    - tipos (un string donde iba un numero)
    - rangos (gasto negativo, cliente_id = -3)
    - campos obligatorios

Si algo no cumple, FastAPI responde 422 y el modelo NO se ejecuta.
Eso es lo que evita errores tontos en produccion.
"""

from pydantic import BaseModel, Field


class EntradaPrediccion(BaseModel):
    """
    Body JSON de POST /predecir/sklearn y POST /predecir/red.

    Los nombres coinciden con COLUMNAS_FEATURES a proposito:
    el modelo recibe exactamente estas cuatro cifras.
    """

    cliente_id: int = Field(
        ...,
        ge=1,
        description="Identificador del cliente (entero >= 1).",
        examples=[42],
    )
    num_compras: int = Field(..., ge=0, description="Compras hasta la fecha de corte.")
    gasto_historico: float = Field(..., ge=0, description="Suma de importes del pasado.")
    ticket_medio: float = Field(..., ge=0, description="Media del importe por compra.")
    cantidad_total: int = Field(..., ge=0, description="Unidades compradas en el pasado.")


class SalidaPrediccion(BaseModel):
    """Lo que devolvemos al cliente HTTP. Siempre el mismo formato."""

    cliente_id: int
    valor_predicho: float = Field(description="Gasto estimado a 90 dias.")
    modelo: str
    version: str
    tiempo_ms: int = Field(description="Latencia de ESTA peticion, en milisegundos.")


class ErrorValidacion(BaseModel):
    """Documentacion: FastAPI ya genera este cuerpo cuando Pydantic falla."""

    detail: list
