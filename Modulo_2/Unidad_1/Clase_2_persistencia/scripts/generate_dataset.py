"""
Generador de datos sinteticos para la plataforma de ventas.

Crea tres archivos CSV en data/raw/ con datos ficticios pero COHERENTES entre si:

    clientes.csv    ->  10.000 clientes
    productos.csv   ->   2.000 productos
    ventas.csv      -> 200.000 ventas que apuntan a clientes y productos existentes

Los datos son "reproducibles": usamos una semilla aleatoria fija (SEED), asi que
todos los estudiantes obtienen exactamente los mismos CSV. Por eso los CSV no se
guardan en git: se regeneran con este script.

Uso:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --ventas 50000        # version reducida
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuracion general
# -----------------------------------------------------------------------------

# Semilla fija: garantiza que el dataset sea siempre el mismo (reproducibilidad).
SEED = 42

# Cantidades por defecto que pide el enunciado de la clase.
N_CLIENTES = 10_000
N_PRODUCTOS = 2_000
N_VENTAS = 200_000

# Carpeta de salida: <raiz del proyecto>/data/raw
# Path(__file__) es este archivo; .parents[1] sube de scripts/ a la raiz.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_SALIDA = RAIZ_PROYECTO / "data" / "raw"

# Catalogos pequenos que combinamos para inventar nombres, ciudades, etc.
NOMBRES = [
    "Ana", "Luis", "Maria", "Carlos", "Laura", "Javier", "Elena", "Miguel",
    "Sofia", "Diego", "Lucia", "Pablo", "Carmen", "Andres", "Marta", "Sergio",
    "Paula", "Ruben", "Julia", "Alberto",
]
APELLIDOS = [
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Sanchez", "Perez", "Gomez",
    "Fernandez", "Diaz", "Torres", "Ramirez", "Vargas", "Castro", "Rojas",
    "Moreno", "Herrera", "Silva", "Mendoza", "Ortiz", "Navarro",
]
CIUDADES = [
    ("Madrid", "Espana"), ("Barcelona", "Espana"), ("Valencia", "Espana"),
    ("Bogota", "Colombia"), ("Medellin", "Colombia"), ("Cali", "Colombia"),
    ("Ciudad de Mexico", "Mexico"), ("Guadalajara", "Mexico"),
    ("Buenos Aires", "Argentina"), ("Santiago", "Chile"), ("Lima", "Peru"),
    ("Quito", "Ecuador"),
]
# Segmento de cliente y su probabilidad (los "bronce" son mayoria).
SEGMENTOS = ["bronce", "plata", "oro", "platino"]
PROBS_SEGMENTO = [0.55, 0.25, 0.15, 0.05]

# Categorias de producto con su rango de precios (min, max) en euros.
CATEGORIAS = {
    "Electronica": (50, 1500),
    "Hogar": (10, 400),
    "Ropa": (8, 150),
    "Deportes": (15, 600),
    "Libros": (5, 60),
    "Juguetes": (7, 120),
    "Alimentacion": (1, 40),
}
# Palabras para construir nombres de producto tipo "Auriculares Pro 128".
BASES_PRODUCTO = [
    "Auriculares", "Teclado", "Monitor", "Lampara", "Silla", "Camiseta",
    "Zapatillas", "Balon", "Mochila", "Novela", "Puzzle", "Cafetera",
    "Batidora", "Raqueta", "Sudadera", "Altavoz", "Tablet", "Manual",
]
MODIFICADORES = ["Pro", "Basic", "Plus", "Max", "Eco", "Ultra", "Mini", "Classic"]

# Canal por el que se realiza la venta.
CANALES = ["web", "app", "tienda", "telefono"]
PROBS_CANAL = [0.45, 0.30, 0.20, 0.05]

# Ventana temporal de los datos.
FECHA_INICIO_REGISTRO = "2022-01-01"
FECHA_FIN_REGISTRO = "2024-12-31"
FECHA_FIN_VENTAS = "2025-12-31"


# -----------------------------------------------------------------------------
# Generacion de cada tabla
# -----------------------------------------------------------------------------

def generar_clientes(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Devuelve un DataFrame de clientes con id, datos personales y fecha de registro."""

    # Los ids van de 1 a n: seran la clave primaria en PostgreSQL.
    cliente_id = np.arange(1, n + 1)

    # Elegimos nombre y apellido al azar y los combinamos.
    nombres = rng.choice(NOMBRES, size=n)
    apellidos = rng.choice(APELLIDOS, size=n)
    nombre_completo = [f"{n_} {a_}" for n_, a_ in zip(nombres, apellidos)]

    # El email incluye el id para que sea UNICO (lo pediremos como restriccion en la BD).
    email = [
        f"{n_.lower()}.{a_.lower()}{i}@example.com"
        for n_, a_, i in zip(nombres, apellidos, cliente_id)
    ]

    # Ciudad y pais viajan juntos para que la combinacion sea realista.
    indices_ciudad = rng.integers(0, len(CIUDADES), size=n)
    ciudad = [CIUDADES[i][0] for i in indices_ciudad]
    pais = [CIUDADES[i][1] for i in indices_ciudad]

    segmento = rng.choice(SEGMENTOS, size=n, p=PROBS_SEGMENTO)

    # Fecha de registro: un dia al azar dentro de la ventana definida arriba.
    inicio = pd.Timestamp(FECHA_INICIO_REGISTRO)
    dias_ventana = (pd.Timestamp(FECHA_FIN_REGISTRO) - inicio).days
    fecha_registro = inicio + pd.to_timedelta(rng.integers(0, dias_ventana, size=n), unit="D")

    return pd.DataFrame({
        "cliente_id": cliente_id,
        "nombre": nombre_completo,
        "email": email,
        "ciudad": ciudad,
        "pais": pais,
        "segmento": segmento,
        "fecha_registro": fecha_registro.date,
    })


def generar_productos(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Devuelve un DataFrame de productos con precio coherente con su categoria."""

    producto_id = np.arange(1, n + 1)

    categorias = list(CATEGORIAS.keys())
    categoria = rng.choice(categorias, size=n)

    # Nombre "inventado" pero legible: base + modificador + numero.
    bases = rng.choice(BASES_PRODUCTO, size=n)
    mods = rng.choice(MODIFICADORES, size=n)
    nombre = [f"{b} {m} {i}" for b, m, i in zip(bases, mods, producto_id)]

    # El precio depende de la categoria: sacamos el rango (min, max) de cada fila
    # y generamos un valor uniforme dentro de ese rango.
    precio_min = np.array([CATEGORIAS[c][0] for c in categoria], dtype=float)
    precio_max = np.array([CATEGORIAS[c][1] for c in categoria], dtype=float)
    precio = np.round(rng.uniform(precio_min, precio_max), 2)

    # Coste interno: entre el 55% y el 80% del precio (asi el margen es positivo).
    coste = np.round(precio * rng.uniform(0.55, 0.80, size=n), 2)

    stock = rng.integers(0, 500, size=n)

    # Un 8% de los productos esta descatalogado (activo = False).
    activo = rng.random(size=n) > 0.08

    return pd.DataFrame({
        "producto_id": producto_id,
        "nombre": nombre,
        "categoria": categoria,
        "precio": precio,
        "coste": coste,
        "stock": stock,
        "activo": activo,
    })


def generar_ventas(
    rng: np.random.Generator,
    n: int,
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame de ventas COHERENTE con clientes y productos:

    - cliente_id y producto_id siempre existen (integridad referencial).
    - precio_unitario es el precio real del producto vendido.
    - fecha_venta es siempre POSTERIOR a la fecha de registro del cliente.
    """

    venta_id = np.arange(1, n + 1)

    # Escogemos clientes y productos existentes. Los clientes con mejor segmento
    # compran mas: damos mas peso a "oro" y "platino".
    peso_por_segmento = {"bronce": 1.0, "plata": 1.8, "oro": 3.0, "platino": 5.0}
    pesos_cliente = clientes["segmento"].map(peso_por_segmento).to_numpy()
    pesos_cliente = pesos_cliente / pesos_cliente.sum()

    # rng.choice con "p" respeta esas probabilidades: es un muestreo con reemplazo.
    cliente_id = rng.choice(clientes["cliente_id"].to_numpy(), size=n, p=pesos_cliente)
    producto_id = rng.choice(productos["producto_id"].to_numpy(), size=n)

    # Recuperamos el precio del producto vendido.
    # Al tener ids consecutivos 1..N, el indice del array es producto_id - 1.
    precios = productos["precio"].to_numpy()
    precio_unitario = precios[producto_id - 1]

    # Cantidad: casi siempre 1-3 unidades, rara vez mas.
    cantidad = rng.choice([1, 2, 3, 4, 5, 10], size=n, p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02])

    # Descuento: la mayoria de ventas no lleva descuento.
    descuento = rng.choice([0.0, 0.05, 0.10, 0.20, 0.30], size=n, p=[0.60, 0.15, 0.13, 0.09, 0.03])

    # Importe final de la linea de venta.
    total = np.round(cantidad * precio_unitario * (1 - descuento), 2)

    # Fecha de la venta: un dia al azar entre el registro del cliente y el fin
    # del periodo. Repartimos con una fraccion aleatoria (0-1) de los dias que
    # quedan disponibles para ese cliente, en lugar de sumar dias fijos y
    # recortar: asi las ventas no se acumulan artificialmente en el ultimo dia.
    fechas_registro = pd.to_datetime(clientes["fecha_registro"]).to_numpy()
    fecha_base = pd.to_datetime(fechas_registro[cliente_id - 1])
    dias_disponibles = (pd.Timestamp(FECHA_FIN_VENTAS) - fecha_base).days
    dias_extra = (rng.random(size=n) * dias_disponibles).astype(int)
    fecha_venta = pd.Series(fecha_base + pd.to_timedelta(dias_extra, unit="D"))

    canal = rng.choice(CANALES, size=n, p=PROBS_CANAL)

    return pd.DataFrame({
        "venta_id": venta_id,
        "cliente_id": cliente_id,
        "producto_id": producto_id,
        "fecha_venta": fecha_venta.dt.date.to_numpy(),
        "cantidad": cantidad,
        "precio_unitario": precio_unitario,
        "descuento": descuento,
        "total": total,
        "canal": canal,
    })


# -----------------------------------------------------------------------------
# Punto de entrada
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los CSV sinteticos del proyecto.")
    parser.add_argument("--clientes", type=int, default=N_CLIENTES, help="Numero de clientes")
    parser.add_argument("--productos", type=int, default=N_PRODUCTOS, help="Numero de productos")
    parser.add_argument("--ventas", type=int, default=N_VENTAS, help="Numero de ventas")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla aleatoria")
    args = parser.parse_args()

    inicio = time.perf_counter()

    # Un unico generador aleatorio con semilla: la clave de la reproducibilidad.
    rng = np.random.default_rng(args.seed)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    print("Generando clientes...")
    clientes = generar_clientes(rng, args.clientes)

    print("Generando productos...")
    productos = generar_productos(rng, args.productos)

    print("Generando ventas...")
    ventas = generar_ventas(rng, args.ventas, clientes, productos)

    # index=False evita escribir la columna de indice de Pandas en el CSV.
    clientes.to_csv(CARPETA_SALIDA / "clientes.csv", index=False)
    productos.to_csv(CARPETA_SALIDA / "productos.csv", index=False)
    ventas.to_csv(CARPETA_SALIDA / "ventas.csv", index=False)

    duracion = time.perf_counter() - inicio
    print("\nCSV creados en", CARPETA_SALIDA)
    print(f"  clientes.csv  -> {len(clientes):>7,} filas")
    print(f"  productos.csv -> {len(productos):>7,} filas")
    print(f"  ventas.csv    -> {len(ventas):>7,} filas")
    print(f"Tiempo total: {duracion:.2f} s")


if __name__ == "__main__":
    main()