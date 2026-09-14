"""
Validaciones de calidad de datos sobre un DataFrame de Pandas.

Cada funcion devuelve una lista de incidencias (texto legible).
Si la lista esta vacia, todo esta bien. Si tiene elementos, hay problemas.

No lanzamos excepciones aqui: solo REPORTAMOS. Quien decide que hacer
(detener el pipeline, alertar, cargar igual...) es pipeline.py o el notebook.
"""

import pandas as pd

from src.contratos import ContratoDatos


def validar_esquema(datos: pd.DataFrame, contrato: ContratoDatos) -> list[str]:
    """
    Comprueba que el CSV tiene las columnas correctas.

    Detecta tres problemas tipicos de cambio de formato (schema drift):
      - columnas FALTANTES (alguien borro una columna del CSV)
      - columnas EXTRA (alguien agrego columnas nuevas)
    """
    incidencias: list[str] = []

    # set() convierte listas en conjuntos para comparar facilmente.
    columnas_esperadas = set(contrato.nombres_columnas)
    columnas_recibidas = set(datos.columns)

    faltantes = columnas_esperadas - columnas_recibidas
    if faltantes:
        incidencias.append(f"Columnas FALTANTES: {sorted(faltantes)}")

    extras = columnas_recibidas - columnas_esperadas
    if extras:
        incidencias.append(f"Columnas EXTRA (schema drift): {sorted(extras)}")

    return incidencias


def validar_tipos(datos: pd.DataFrame, contrato: ContratoDatos) -> list[str]:
    """
    Comprueba que cada columna tiene un tipo compatible con lo esperado.

    Pandas a veces lee numeros como object si hay un valor de texto mezclado.
    Por eso esta validacion detecta CSV corruptos del tipo precio = "abc".
    """
    incidencias: list[str] = []

    # Mapeo de nuestro contrato a dtypes habituales de Pandas.
    mapa_tipos = {
        "int": ("int64", "int32", "Int64"),
        "float": ("float64", "float32"),
        "str": ("object", "string"),
        "bool": ("bool",),
        "date": ("datetime64[ns]", "dbdate"),
    }

    for col_contrato in contrato.columnas:
        # Si la columna no existe, validar_esquema ya lo reporto.
        if col_contrato.nombre not in datos.columns:
            continue

        tipo_real = str(datos[col_contrato.nombre].dtype)
        tipos_validos = mapa_tipos.get(col_contrato.tipo, ())

        if tipo_real not in tipos_validos:
            incidencias.append(
                f"Columna '{col_contrato.nombre}': tipo esperado "
                f"{col_contrato.tipo}, recibido {tipo_real}"
            )

    return incidencias


def validar_nulos(datos: pd.DataFrame, contrato: ContratoDatos) -> list[str]:
    """Comprueba que las columnas obligatorias no tienen valores nulos."""
    incidencias: list[str] = []

    for col_contrato in contrato.columnas:
        if col_contrato.nombre not in datos.columns:
            continue

        if not col_contrato.obligatoria:
            continue

        # isna() detecta NaN/None; sum() cuenta cuantos True hay.
        num_nulos = int(datos[col_contrato.nombre].isna().sum())
        if num_nulos > 0:
            incidencias.append(
                f"Columna '{col_contrato.nombre}': {num_nulos} valores nulos "
                "(columna obligatoria)"
            )

    return incidencias


def validar_rangos(datos: pd.DataFrame, contrato: ContratoDatos) -> list[str]:
    """Comprueba minimos, maximos y valores permitidos."""
    incidencias: list[str] = []

    for col_contrato in contrato.columnas:
        if col_contrato.nombre not in datos.columns:
            continue

        serie = datos[col_contrato.nombre]

        if col_contrato.minimo is not None:
            try:
                violaciones = int((serie < col_contrato.minimo).sum())
            except TypeError:
                continue
            if violaciones > 0:
                incidencias.append(
                    f"Columna '{col_contrato.nombre}': {violaciones} filas "
                    f"con valor < {col_contrato.minimo}"
                )

        if col_contrato.maximo is not None:
            try:
                violaciones = int((serie > col_contrato.maximo).sum())
            except TypeError:
                continue
            if violaciones > 0:
                incidencias.append(
                    f"Columna '{col_contrato.nombre}': {violaciones} filas "
                    f"con valor > {col_contrato.maximo}"
                )

        if col_contrato.valores_permitidos is not None:
            invalidos = ~serie.isin(col_contrato.valores_permitidos)
            num_invalidos = int(invalidos.sum())
            if num_invalidos > 0:
                incidencias.append(
                    f"Columna '{col_contrato.nombre}': {num_invalidos} filas "
                    "con valor no permitido"
                )

    return incidencias


def validar_duplicados_id(datos: pd.DataFrame, contrato: ContratoDatos) -> list[str]:
    """Comprueba que la primera columna del contrato (el id) no tiene duplicados."""
    incidencias: list[str] = []
    col_id = contrato.columnas[0].nombre

    if col_id in datos.columns:
        num_dup = int(datos[col_id].duplicated().sum())
        if num_dup > 0:
            incidencias.append(f"Columna '{col_id}': {num_dup} ids duplicados")

    return incidencias


def ejecutar_todas_las_validaciones(
    datos: pd.DataFrame,
    contrato: ContratoDatos,
) -> list[str]:
    """
    Ejecuta TODAS las validaciones y devuelve la lista completa de incidencias.

    Esta es la funcion que usamos desde el notebook y desde pipeline.py.
    """
    incidencias: list[str] = []
    incidencias.extend(validar_esquema(datos, contrato))
    incidencias.extend(validar_tipos(datos, contrato))
    incidencias.extend(validar_nulos(datos, contrato))
    incidencias.extend(validar_rangos(datos, contrato))
    incidencias.extend(validar_duplicados_id(datos, contrato))
    return incidencias


def resumen_validacion(incidencias: list[str]) -> str:
    """Devuelve un texto corto para imprimir en clase."""
    if not incidencias:
        return "OK: el archivo cumple el contrato."
    lineas = [f"ALERTA: {len(incidencias)} incidencia(s) de calidad:"]
    lineas.extend(f"  - {item}" for item in incidencias)
    return "\n".join(lineas)
