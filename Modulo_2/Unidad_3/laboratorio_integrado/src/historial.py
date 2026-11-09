"""
Historial de lotes de predicción en SQLite (sqlite3 de la biblioteca estándar).

El lote es el de la API: `SalidaLote.predicciones`. Cada predicción trae
cliente_id, valor_predicho, modelo, version y tiempo_ms. El motor
("sklearn" o "red") lo pone quien llama, igual que en POST /predecir/lote/{motor}.

El SQL vive arriba, en texto. conectar() abre data/historial.sqlite.
crear_tabla() ejecuta el DDL y hace commit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_SQLITE = RAIZ / "data" / "historial.sqlite"

# Tabla del historial. crear_tabla() ejecuta este texto.
DDL_HISTORIAL = """
CREATE TABLE IF NOT EXISTS historial (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lote_id         INTEGER NOT NULL,
    motor           TEXT    NOT NULL,
    modelo          TEXT    NOT NULL,
    version         TEXT    NOT NULL,
    cliente_id      INTEGER NOT NULL,
    valor_predicho  REAL    NOT NULL,
    tiempo_ms       INTEGER NOT NULL,
    creado_en       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SQL_SIGUIENTE_LOTE = """
SELECT COALESCE(MAX(lote_id), 0) + 1 AS siguiente
FROM historial
"""

SQL_INSERTAR = """
INSERT INTO historial (
    lote_id, motor, modelo, version,
    cliente_id, valor_predicho, tiempo_ms
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SQL_LEER = """
SELECT
    id, lote_id, motor, modelo, version,
    cliente_id, valor_predicho, tiempo_ms, creado_en
FROM historial
"""

SQL_RESUMEN = """
SELECT
    lote_id,
    motor,
    modelo,
    version,
    COUNT(*)                          AS n,
    ROUND(AVG(valor_predicho), 2)     AS media_predicha,
    ROUND(MIN(valor_predicho), 2)     AS minimo,
    ROUND(MAX(valor_predicho), 2)     AS maximo,
    SUM(tiempo_ms)                    AS tiempo_ms,
    MIN(creado_en)                    AS creado_en
FROM historial
GROUP BY lote_id, motor, modelo, version
ORDER BY lote_id DESC
"""

SQL_VACIAR = "DELETE FROM historial"
SQL_REINICIAR_IDS = "DELETE FROM sqlite_sequence WHERE name = 'historial'"

MOTORES = ("sklearn", "red")
CLAVES_PREDICCION = (
    "cliente_id",
    "valor_predicho",
    "modelo",
    "version",
    "tiempo_ms",
)


def conectar() -> sqlite3.Connection:
    """Abre data/historial.sqlite y devuelve la conexión."""
    RUTA_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(RUTA_SQLITE)
    conexion.row_factory = sqlite3.Row
    return conexion


def crear_tabla() -> None:
    """Crea la tabla historial. Se puede llamar otra vez: el DDL usa IF NOT EXISTS."""
    conexion = conectar()
    conexion.execute(DDL_HISTORIAL)
    conexion.commit()
    conexion.close()


def _como_dict(prediccion: object) -> dict:
    """Acepta un dict o un SalidaPrediccion (tiene model_dump)."""
    if isinstance(prediccion, dict):
        datos = prediccion
    elif hasattr(prediccion, "model_dump"):
        datos = prediccion.model_dump()
    else:
        raise TypeError(
            "Cada predicción tiene que ser un dict o un modelo con model_dump()."
        )
    faltan = [clave for clave in CLAVES_PREDICCION if clave not in datos]
    if faltan:
        raise ValueError(f"La predicción no trae {faltan}.")
    return datos


def insertar_lote(
    conexion: sqlite3.Connection,
    predicciones: list,
    motor: str,
) -> int:
    """
    Guarda un lote y devuelve su lote_id.

    predicciones es SalidaLote.predicciones: dicts o SalidaPrediccion.
    Todas las filas comparten el mismo lote_id.
    """
    if motor not in MOTORES:
        raise ValueError("motor tiene que ser 'sklearn' o 'red'.")
    if not predicciones:
        raise ValueError("El lote no trae predicciones.")

    filas = [_como_dict(prediccion) for prediccion in predicciones]
    with conexion:
        lote_id = int(conexion.execute(SQL_SIGUIENTE_LOTE).fetchone()["siguiente"])
        valores = []
        for fila in filas:
            valores.append(
                (
                    lote_id,
                    motor,
                    str(fila["modelo"]),
                    str(fila["version"]),
                    int(fila["cliente_id"]),
                    float(fila["valor_predicho"]),
                    int(fila["tiempo_ms"]),
                )
            )
        conexion.executemany(SQL_INSERTAR, valores)
    return lote_id


def leer_historial(
    conexion: sqlite3.Connection,
    lote_id: int | None = None,
    limite: int | None = None,
) -> list[dict]:
    """Lee el historial. El lote más reciente va primero."""
    consulta = SQL_LEER
    parametros: list = []
    if lote_id is not None:
        consulta += "\nWHERE lote_id = ?"
        parametros.append(int(lote_id))
    consulta += "\nORDER BY lote_id DESC, id ASC"
    if limite is not None:
        consulta += "\nLIMIT ?"
        parametros.append(int(limite))
    filas = conexion.execute(consulta, parametros).fetchall()
    return [dict(fila) for fila in filas]


def resumir_lotes(conexion: sqlite3.Connection) -> list[dict]:
    """Una fila por lote: cuántas predicciones, media y rango del valor predicho."""
    filas = conexion.execute(SQL_RESUMEN).fetchall()
    return [dict(fila) for fila in filas]


def vaciar(conexion: sqlite3.Connection) -> int:
    """Borra todas las filas y devuelve cuántas había. El siguiente lote vuelve a empezar en 1."""
    with conexion:
        cursor = conexion.execute(SQL_VACIAR)
        borradas = cursor.rowcount
        conexion.execute(SQL_REINICIAR_IDS)
    return borradas


if __name__ == "__main__":
    crear_tabla()
    print(RUTA_SQLITE)
