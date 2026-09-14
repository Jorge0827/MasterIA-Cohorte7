"""
Sistema de alertas: convierte incidencias de calidad en registros de auditoria.

Cuando el formato del CSV cambia (schema drift), generamos una ALERTA:
  - se guarda en la tabla auditoria con estado='error'
  - el mensaje describe que cambio detectamos
  - el pipeline puede decidir NO cargar los datos

En produccion real, ademas de auditoria se enviaria email, Slack o PagerDuty.
Aqui nos quedamos con la tabla auditoria de la Clase 2: misma caja negra,
nueva pregunta (calidad, no solo "se cargo o no").
"""

from src.auditoria import registrar_auditoria, ultimas_auditorias


def generar_alerta(
    proceso: str,
    tabla: str,
    incidencias: list[str],
    registros_procesados: int = 0,
) -> int:
    """
    Registra una alerta de calidad en la tabla auditoria.

    Devuelve el auditoria_id generado por PostgreSQL.
    """
    # Unimos todas las incidencias en un solo mensaje legible.
    mensaje = "ALERTA CALIDAD: " + " | ".join(incidencias)

    # Reutilizamos la funcion de la Clase 2: misma tabla, mismo formato.
    return registrar_auditoria(
        proceso=proceso,
        operacion="VALIDACION",
        tabla_afectada=tabla,
        registros_procesados=registros_procesados,
        estado="error",
        mensaje=mensaje[:500],
    )


def generar_alerta_exito(
    proceso: str,
    tabla: str,
    registros: int,
    mensaje: str = "Validacion superada sin incidencias",
) -> int:
    """Registra que la validacion paso correctamente."""
    return registrar_auditoria(
        proceso=proceso,
        operacion="VALIDACION",
        tabla_afectada=tabla,
        registros_procesados=registros,
        estado="exito",
        mensaje=mensaje,
    )


def ultimas_alertas_calidad(limite: int = 10):
    """
    Devuelve las ultimas auditorias de VALIDACION.

    Es un filtro pedagogico sobre ultimas_auditorias(): en clase queremos
    ver solo las alertas de calidad, no las cargas de la Clase 2.
    """
    historial = ultimas_auditorias(limite=50)
    if historial.empty:
        return historial
    return historial[historial["operacion"] == "VALIDACION"].head(limite)
