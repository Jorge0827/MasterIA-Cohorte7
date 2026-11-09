"""
Qué estamos construyendo

Dos descargas, sin Streamlit en este archivo:

- el monitoreo en CSV (una fila por cliente)
- un informe Markdown: métricas del tablero, ganador por MAE más bajo,
  comparación con meta.json y el glosario de MAE, RMSE, R2 y sesgo

`informe_bytes` solo pasa ese Markdown a bytes UTF-8.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.evaluacion import resumir_metricas


def csv_bytes(monitoreo: pd.DataFrame) -> bytes:
    """El DataFrame de monitoreo como CSV, en bytes UTF-8 y sin índice."""
    # to_csv recibe el DataFrame y devuelve un str: cabecera y una línea por cliente.
    # index=False deja fuera la columna 0, 1, 2… del índice.
    texto = monitoreo.to_csv(index=False)
    # encode recibe ese str y "utf-8"; devuelve bytes listos para guardar o descargar.
    return texto.encode("utf-8")


def informe_markdown(monitoreo: pd.DataFrame, meta: dict) -> str:
    """Informe del lote actual frente a la ficha guardada en meta.json."""
    resumen = resumir_metricas(monitoreo)

    # datetime.now no recibe argumentos; devuelve un datetime con la fecha y la hora de ahora.
    ahora = datetime.now()
    # strftime recibe el patrón y devuelve un str con la fecha ya escrita.
    fecha = ahora.strftime("%Y-%m-%d %H:%M")

    # dict.get recibe la clave y un respaldo; devuelve el valor de meta o ese respaldo.
    version = meta.get("version", "sin versión")
    # dict.get recibe la clave y un respaldo; devuelve la fecha de corte o "sin fecha".
    fecha_corte = meta.get("fecha_corte", "sin fecha")
    # dict.get recibe la clave; devuelve n_filas o None si meta.json no la trae.
    n_meta = meta.get("n_filas")

    sklearn = resumen["sklearn"]
    mlp = resumen["mlp"]
    if sklearn["mae"] < mlp["mae"]:
        ganador, perdedor = "sklearn", "mlp"
    elif mlp["mae"] < sklearn["mae"]:
        ganador, perdedor = "mlp", "sklearn"
    else:
        ganador = perdedor = None

    lineas = [
        "# Informe de monitoreo de gasto",
        "",
        f"Generado: {fecha}",
        f"Versión del artefacto: {version}",
        f"Fecha de corte: {fecha_corte}",
        f"Clientes en este tablero: {sklearn['n']}",
        f"Filas guardadas en meta.json: {n_meta}",
        "",
        "## Métricas del tablero",
        "",
        "Salen de `resumir_metricas`: el mismo `y_real` contra `pred_sklearn` y `pred_mlp`.",
        "",
        "| Motor | n | MAE | RMSE | R2 | Sesgo |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| sklearn | {sklearn['n']} | {sklearn['mae']} | {sklearn['rmse']} "
            f"| {sklearn['r2']} | {sklearn['sesgo']} |"
        ),
        (
            f"| mlp | {mlp['n']} | {mlp['mae']} | {mlp['rmse']} "
            f"| {mlp['r2']} | {mlp['sesgo']} |"
        ),
        "",
        "## Ganador",
        "",
    ]
    if ganador is None:
        lineas.append(f"Empate: sklearn y mlp tienen el mismo MAE ({sklearn['mae']}).")
    else:
        lineas.append(
            f"**{ganador}** gana por MAE más bajo: {resumen[ganador]['mae']} "
            f"frente a {resumen[perdedor]['mae']} de {perdedor}."
        )

    lineas.extend(
        [
            "",
            "## Tablero frente a meta.json",
            "",
            "El tablero mide el lote que recibe ahora. meta.json guarda el MAE y el R2 del entrenamiento.",
            "",
            "| Motor | MAE tablero | MAE meta | MAE tablero − meta | R2 tablero | R2 meta | R2 tablero − meta |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for nombre in ("sklearn", "mlp"):
        tablero = resumen[nombre]
        # dict.get recibe el nombre del motor y {} si esa ficha no está; devuelve un dict.
        ficha = meta.get(nombre, {})
        if not isinstance(ficha, dict):
            ficha = {}
        # dict.get recibe "mae"; devuelve el MAE guardado o None.
        mae_meta = ficha.get("mae")
        # dict.get recibe "r2"; devuelve el R2 guardado o None.
        r2_meta = ficha.get("r2")
        if mae_meta is None:
            mae_meta_txt, delta_mae = "no está en meta.json", "sin dato"
        else:
            mae_meta_txt = str(mae_meta)
            delta_mae = f"{round(float(tablero['mae']) - float(mae_meta), 2):+.2f}"
        if r2_meta is None:
            r2_meta_txt, delta_r2 = "no está en meta.json", "sin dato"
        else:
            r2_meta_txt = str(r2_meta)
            delta_r2 = f"{round(float(tablero['r2']) - float(r2_meta), 3):+.3f}"
        lineas.append(
            f"| {nombre} | {tablero['mae']} | {mae_meta_txt} | {delta_mae} "
            f"| {tablero['r2']} | {r2_meta_txt} | {delta_r2} |"
        )

    lineas.extend(
        [
            "",
            "## Glosario",
            "",
            "- **MAE**: error absoluto medio, en euros. Media de |predicción − real|. Más bajo, más cerca del gasto.",
            "- **RMSE**: raíz del error cuadrático medio. Como el MAE, pero un error grande pesa más.",
            "- **R2**: parte de la variación del gasto que el modelo acompaña. 1 sería un acierto perfecto; cerca de 0, no mejora a predecir la media.",
            "- **Sesgo**: media de (predicción − real). Positivo: el modelo se pasa. Negativo: se queda corto.",
            "",
        ]
    )
    return "\n".join(lineas)


def informe_bytes(informe: str) -> bytes:
    """El Markdown del informe, en bytes UTF-8."""
    # encode recibe el str del informe y "utf-8"; devuelve bytes.
    return informe.encode("utf-8")
