"""
Pipeline de carga con validacion de calidad.

Flujo:

  1. Leer CSV
  2. Validar contra el contrato
  3. Si hay incidencias -> alerta + NO cargar (fail fast)
  4. Si todo OK -> (opcional) cargar a PostgreSQL + auditar

Este modulo CONECTA la Clase 2 (carga + auditoria) con la Clase 3
(calidad + alertas).

IMPORTANTE para no romper la Clase 2:
  cargar_si_ok=False por defecto. Asi puedes validar y alertar sin
  vaciar ni recargar las tablas que ya tienes en PostgreSQL.
"""

from pathlib import Path

import pandas as pd

from src.alertas import generar_alerta, generar_alerta_exito
from src.auditoria import auditar
from src.calidad import ejecutar_todas_las_validaciones, resumen_validacion
from src.carga import ARCHIVOS, cargar_tabla, leer_csv
from src.config import CARPETA_DATOS
from src.contratos import CONTRATOS

# Los CSV "rotos" de la demo viven aqui. No se tocan los CSV oficiales.
CARPETA_CORRUPTOS = CARPETA_DATOS / "corruptos"


class ErrorCalidadDatos(Exception):
    """
    Excepcion propia para problemas de calidad.

    Usamos una excepcion custom (en lugar de Exception generica) para que
    quien capture el error sepa que fue un fallo de calidad, no un error
    de red o de base de datos.
    """

    def __init__(self, incidencias: list[str]):
        self.incidencias = incidencias
        super().__init__("Fallo de calidad: " + "; ".join(incidencias))


def leer_csv_alternativo(ruta: str | Path, nombre_tabla: str) -> pd.DataFrame:
    """
    Lee un CSV que NO es el oficial de data/raw/.

    Se usa para la demo de archivos corruptos. Respeta parse_dates del
    contrato oficial cuando esas columnas existen.
    """
    preview = pd.read_csv(ruta, nrows=0)
    configuracion = ARCHIVOS.get(nombre_tabla, {"fechas": []})
    # Solo parseamos fechas que realmente existan en ESTE archivo.
    fechas = [col for col in configuracion.get("fechas", []) if col in preview.columns]
    return pd.read_csv(ruta, parse_dates=fechas or None)


def validar_csv(
    nombre_tabla: str,
    ruta_alternativa: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Lee un CSV y ejecuta todas las validaciones.

    ruta_alternativa: si queremos validar un CSV corrupto de data/raw/corruptos/.

    Devuelve (DataFrame, lista_de_incidencias).
    """
    if nombre_tabla not in CONTRATOS:
        raise ValueError(f"No hay contrato para '{nombre_tabla}'. Opciones: {list(CONTRATOS)}")

    contrato = CONTRATOS[nombre_tabla]

    if ruta_alternativa:
        datos = leer_csv_alternativo(ruta_alternativa, nombre_tabla)
    else:
        # Flujo normal: usar leer_csv de la Clase 2 (con parse_dates).
        datos = leer_csv(nombre_tabla)

    incidencias = ejecutar_todas_las_validaciones(datos, contrato)
    return datos, incidencias


def ejecutar_pipeline(
    nombre_tabla: str,
    ruta_alternativa: str | Path | None = None,
    cargar_si_ok: bool = False,
    vaciar_antes: bool = True,
) -> dict:
    """
    Pipeline completo: validar -> alertar -> (opcional) cargar.

    cargar_si_ok=False por defecto para no tocar las tablas de la Clase 2
    mientras se demuestra la validacion.

    Devuelve un diccionario con el resultado del proceso.
    """
    resultado = {
        "tabla": nombre_tabla,
        "cargado": False,
        "filas": 0,
        "incidencias": [],
        "mensaje": "",
    }

    # Todo el bloque queda auditado automaticamente (funcion de la Clase 2).
    with auditar(
        proceso="pipeline_calidad",
        operacion="VALIDACION",
        tabla_afectada=nombre_tabla,
    ) as info:
        datos, incidencias = validar_csv(nombre_tabla, ruta_alternativa)
        resultado["incidencias"] = incidencias
        resultado["filas"] = len(datos)
        info["registros"] = len(datos)

        if incidencias:
            generar_alerta(
                proceso="pipeline_calidad",
                tabla=nombre_tabla,
                incidencias=incidencias,
                registros_procesados=len(datos),
            )
            info["mensaje"] = f"RECHAZADO: {len(incidencias)} incidencias"
            resultado["mensaje"] = resumen_validacion(incidencias)
            raise ErrorCalidadDatos(incidencias)

        generar_alerta_exito(
            proceso="pipeline_calidad",
            tabla=nombre_tabla,
            registros=len(datos),
        )

        if cargar_si_ok:
            # Solo si el docente lo pide explicitamente: recarga la tabla oficial.
            filas = cargar_tabla(nombre_tabla, vaciar_antes=vaciar_antes)
            resultado["cargado"] = True
            resultado["filas"] = filas
            info["registros"] = filas
            info["mensaje"] = f"Validado y cargado: {filas} filas"
            resultado["mensaje"] = f"OK y cargado: {filas} filas"
        else:
            info["mensaje"] = f"Validado sin cargar: {len(datos)} filas"
            resultado["mensaje"] = resumen_validacion(incidencias)

    return resultado
