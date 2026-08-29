"""
Definicion del esquema de la base de datos con SQLAlchemy.

Aqui describimos las 5 tablas del proyecto en Python. SQLAlchemy traduce estas
definiciones al SQL (CREATE TABLE ...) que entiende PostgreSQL.

Usamos el estilo "Core" (objetos Table) en lugar del ORM porque se parece mucho
al SQL real y por tanto es mas facil de entender cuando estas aprendiendo:
cada Column de Python es una columna de la tabla.

Modelo de datos:

    clientes  ─┐
               ├──> ventas          (una venta pertenece a un cliente y un producto)
    productos ─┘
    clientes  ────> predicciones    (una prediccion se hace sobre un cliente)
    auditoria                       (tabla independiente: registra los procesos)
"""

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    func,
    inspect,
)
from sqlalchemy.engine import Engine

# MetaData es el "catalogo" donde se registran todas nuestras tablas.
# Desde el podemos crear o borrar todo el esquema de una sola vez.
metadata = MetaData()


# -----------------------------------------------------------------------------
# Tablas de negocio (los datos que vienen de los CSV)
# -----------------------------------------------------------------------------

clientes = Table(
    "clientes",
    metadata,
    # primary_key=True: identifica de forma unica cada fila.
    # autoincrement=False porque los ids ya vienen dados en el CSV.
    Column("cliente_id", Integer, primary_key=True, autoincrement=False),
    Column("nombre", String(120), nullable=False),
    # unique=True: PostgreSQL rechazara dos clientes con el mismo email.
    Column("email", String(150), nullable=False, unique=True),
    Column("ciudad", String(80)),
    Column("pais", String(80)),
    Column("segmento", String(20)),
    Column("fecha_registro", Date, nullable=False),
    # CHECK: restriccion de dominio, solo admite estos cuatro valores.
    CheckConstraint(
        "segmento IN ('bronce', 'plata', 'oro', 'platino')",
        name="ck_clientes_segmento",
    ),
    comment="Clientes de la plataforma",
)

productos = Table(
    "productos",
    metadata,
    Column("producto_id", Integer, primary_key=True, autoincrement=False),
    Column("nombre", String(150), nullable=False),
    Column("categoria", String(60), nullable=False),
    # Numeric(10, 2) = hasta 10 digitos, 2 de ellos decimales.
    # Para dinero se usa Numeric y NUNCA float, porque float pierde precision.
    Column("precio", Numeric(10, 2), nullable=False),
    Column("coste", Numeric(10, 2), nullable=False),
    Column("stock", Integer, nullable=False, default=0),
    Column("activo", Boolean, nullable=False, default=True),
    CheckConstraint("precio >= 0", name="ck_productos_precio_positivo"),
    CheckConstraint("stock >= 0", name="ck_productos_stock_positivo"),
    comment="Catalogo de productos",
)

ventas = Table(
    "ventas",
    metadata,
    Column("venta_id", Integer, primary_key=True, autoincrement=False),
    # ForeignKey: obliga a que el cliente_id exista en la tabla clientes.
    # Esta es la INTEGRIDAD REFERENCIAL, la gran ventaja del modelo relacional.
    Column("cliente_id", Integer, ForeignKey("clientes.cliente_id"), nullable=False),
    Column("producto_id", Integer, ForeignKey("productos.producto_id"), nullable=False),
    Column("fecha_venta", Date, nullable=False),
    Column("cantidad", Integer, nullable=False),
    Column("precio_unitario", Numeric(10, 2), nullable=False),
    Column("descuento", Numeric(4, 2), nullable=False, default=0),
    Column("total", Numeric(12, 2), nullable=False),
    Column("canal", String(20)),
    CheckConstraint("cantidad > 0", name="ck_ventas_cantidad_positiva"),
    CheckConstraint("descuento >= 0 AND descuento <= 1", name="ck_ventas_descuento"),
    comment="Ventas realizadas (tabla de hechos)",
)


# -----------------------------------------------------------------------------
# Tablas de la plataforma de Machine Learning
# -----------------------------------------------------------------------------

predicciones = Table(
    "predicciones",
    metadata,
    # Aqui SI usamos autoincremento: PostgreSQL asigna el id en cada INSERT.
    Column("prediccion_id", Integer, primary_key=True, autoincrement=True),
    Column("modelo", String(80), nullable=False),
    Column("version", String(20), nullable=False),
    # server_default=func.now(): si no enviamos fecha, la pone PostgreSQL.
    Column("fecha_prediccion", DateTime, nullable=False, server_default=func.now()),
    Column("cliente_id", Integer, ForeignKey("clientes.cliente_id"), nullable=False),
    Column("valor_predicho", Numeric(12, 4), nullable=False),
    Column("confianza", Numeric(5, 4)),
    # nullable=True porque el resultado real casi nunca se conoce al predecir:
    # se rellena despues, cuando el hecho ya ha ocurrido.
    Column("resultado_real", Numeric(12, 4), nullable=True),
    Column("estado", String(20), nullable=False, default="ok"),
    Column("tiempo_ejecucion_ms", Integer),
    CheckConstraint("confianza >= 0 AND confianza <= 1", name="ck_predicciones_confianza"),
    CheckConstraint(
        "estado IN ('ok', 'error', 'pendiente')",
        name="ck_predicciones_estado",
    ),
    comment="Registro de predicciones del modelo",
)

auditoria = Table(
    "auditoria",
    metadata,
    Column("auditoria_id", Integer, primary_key=True, autoincrement=True),
    Column("fecha_hora", DateTime, nullable=False, server_default=func.now()),
    # Que proceso se ejecuto, por ejemplo "carga_csv" o "prediccion_batch".
    Column("proceso", String(80), nullable=False),
    # Que hizo: INSERT, SELECT, UPDATE, DELETE...
    Column("operacion", String(30), nullable=False),
    Column("tabla_afectada", String(60)),
    Column("registros_procesados", Integer, nullable=False, default=0),
    Column("estado", String(20), nullable=False),
    # Text (sin longitud) porque un mensaje de error puede ser largo.
    Column("mensaje", Text),
    Column("duracion_segundos", Numeric(10, 3)),
    CheckConstraint("estado IN ('exito', 'error')", name="ck_auditoria_estado"),
    comment="Trazabilidad de los procesos ejecutados sobre los datos",
)


# Orden de carga: primero las tablas "padre" y despues las que tienen claves
# ajenas, porque una venta no puede existir sin su cliente y su producto.
ORDEN_CARGA = ["clientes", "productos", "ventas"]


# -----------------------------------------------------------------------------
# Operaciones sobre el esquema
# -----------------------------------------------------------------------------

def crear_tablas(engine: Engine) -> list[str]:
    """
    Crea en PostgreSQL todas las tablas definidas arriba.

    create_all es idempotente: si la tabla ya existe, no la toca ni da error
    (internamente hace CREATE TABLE IF NOT EXISTS). Por eso se puede ejecutar
    varias veces sin miedo.
    """
    metadata.create_all(engine)
    return listar_tablas(engine)


def listar_tablas(engine: Engine) -> list[str]:
    """Devuelve los nombres de las tablas que existen ahora mismo en la base de datos."""
    # El "inspector" pregunta a PostgreSQL por su propia estructura.
    return sorted(inspect(engine).get_table_names())


def borrar_tablas(engine: Engine) -> None:
    """
    Borra todas las tablas del proyecto. Util para empezar de cero en clase.

    OJO: elimina los datos. drop_all respeta el orden de dependencias, asi que
    borra primero las tablas hijas.
    """
    metadata.drop_all(engine)