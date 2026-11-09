from __future__ import annotations

from pathlib import Path


import pandas as pd

from src.features_modelo import COLUMNAS_FEATURES, FECHA_CORTE

RAIZ = Path(__file__).resolve().parents[1]
CSV_VENTAS = RAIZ / "data" / "raw" / "ventas.csv"
CORTE_TXT = FECHA_CORTE.isoformat()


class ErrorSpark(RuntimeError):
    """Spark No arrancó"""
    
def features_con_pandas(n_clientes: int | None = None) -> pd.DataFrame:
    
    if not CSV_VENTAS.exists():
        raise FileNotFoundError(f"No está {CSV_VENTAS}. copia data/raw/ventas.csv")
    
    ventas = pd.read_csv(CSV_VENTAS, parse_dates=["fecha_venta"])
    corte = pd.Timestamp(FECHA_CORTE)
    # Pasado = lo que el modelo puede ver. El futuro no entra en las features.
    pasado = ventas.loc[ventas["fecha_venta"] <= corte]
    
    features = (
        pasado.groupby("cliente_id", as_index=False)
        .agg(
            num_compras=("venta_id", "count"),
            gasto_historico=("total", "sum"),
            ticket_medio=("total", "mean"),
            cantidad_total=("cantidad", "sum"),
        )
    )
    
    features["ticket_medio"] = features["ticket_medio"].round(2)
    features["gasto_historico"] = features["gasto_historico"].round(2)
    
    features["num_compras"] = features["num_compras"].astype(int)
    features["cantidad_total"] = features["cantidad_total"].astype(int)
    features = features.loc[:, ["cliente_id", *COLUMNAS_FEATURES]]
    
    if n_clientes is not None:
        features = features.head(int(n_clientes))
    
    return features.reset_index(drop=True)


def features_con_spark(n_clientes: int | None = None) -> pd.DataFrame:
    """
    Arranca Spark, lee las 200.000 ventas, filtra el pasado y agrupa por cliente.

    Spark recorre el CSV. Al modelo solo llega una fila por cliente,
    con las mismas columnas que features_con_pandas.
    Si Java o PySpark no arrancan, lanza ErrorSpark.
    """
    import os
    import sys

    if not CSV_VENTAS.exists():
        raise FileNotFoundError(f"No está {CSV_VENTAS}. copia data/raw/ventas.csv")

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            DateType,
            DoubleType,
            IntegerType,
            StringType,
            StructField,
            StructType,
        )
    except ImportError as exc:
        raise ErrorSpark(
            "PySpark no está instalado. No se pueden resumir las ventas."
        ) from exc

    try:
        spark = (
            SparkSession.builder.appName("Laboratorio-features-clientes")
            .master("local[*]")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
    except Exception as exc:
        raise ErrorSpark("Spark no arrancó") from exc

    esquema = StructType(
        [
            StructField("venta_id", IntegerType(), True),
            StructField("cliente_id", IntegerType(), True),
            StructField("producto_id", IntegerType(), True),
            StructField("fecha_venta", DateType(), True),
            StructField("cantidad", IntegerType(), True),
            StructField("precio_unitario", DoubleType(), True),
            StructField("descuento", DoubleType(), True),
            StructField("total", DoubleType(), True),
            StructField("canal", StringType(), True),
        ]
    )

    ventas = (
        spark.read.option("header", "true")
        .schema(esquema)
        .csv(CSV_VENTAS.resolve().as_posix())
    )

    # Misma regla que pandas: fecha_venta <= fecha de corte.
    pasado = ventas.filter(F.col("fecha_venta") <= F.lit(CORTE_TXT))

    features = (
        pasado.groupBy("cliente_id")
        .agg(
            F.count("venta_id").alias("num_compras"),
            F.sum("total").alias("gasto_historico"),
            F.sum("cantidad").alias("cantidad_total"),
        )
        .orderBy("cliente_id")
    )

    if n_clientes is not None:
        features = features.limit(int(n_clientes))

    # toPandas trae el resumen (una fila por cliente), no las 200.000 ventas.
    # El ticket se redondea aquí, igual que en pandas: media = suma / compras.
    tabla = features.toPandas()
    tabla["ticket_medio"] = (tabla["gasto_historico"] / tabla["num_compras"]).round(2)
    tabla["gasto_historico"] = tabla["gasto_historico"].round(2)
    tabla["num_compras"] = tabla["num_compras"].astype(int)
    tabla["cantidad_total"] = tabla["cantidad_total"].astype(int)
    return tabla.loc[:, ["cliente_id", *COLUMNAS_FEATURES]].reset_index(drop=True)


def armar_features(
    motor: str = "spark",
    n_clientes: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Arma la tabla de features con el motor pedido.

    Devuelve (tabla, motor_usado). El segundo valor es "spark" o "pandas":
    si se pidió Spark y no arranca, la tabla sale de pandas y se ve "pandas".
    """
    pedido = motor.strip().lower()
    if pedido == "spark":
        try:
            return features_con_spark(n_clientes), "spark"
        except ErrorSpark:
            return features_con_pandas(n_clientes), "pandas"
    if pedido == "pandas":
        return features_con_pandas(n_clientes), "pandas"
    raise ValueError(f"Motor desconocido: {motor}. Usa 'spark' o 'pandas'.")
 

