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

    Clase 4 (SQL avanzado / historial de entrenamiento):
    historial_entrenamiento  ──> dataset_entrenamiento
         (metadatos del snapshot)    (filas congeladas: features + target)
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


# -----------------------------------------------------------------------------
# Clase 4: historial de datasets de entrenamiento
#
# Idea: NO se entrena un modelo "sobre la tabla ventas de hoy", porque manana
# esa tabla habra cambiado (nuevas ventas, correcciones, recargas). Se CONGELA
# un recorte con fecha_corte, se guarda QUE consulta lo genero y se versiona.
# Asi puedes repetir el mismo entrenamiento dentro de 6 meses.
# -----------------------------------------------------------------------------

historial_entrenamiento = Table(
    "historial_entrenamiento",
    metadata,
    # SERIAL: cada snapshot recibe un id automatico.
    Column("snapshot_id", Integer, primary_key=True, autoincrement=True),
    # Nombre legible del dataset, por ejemplo "features_gasto_90d".
    Column("nombre_dataset", String(80), nullable=False),
    # Version del recorte (1.0.0, 1.0.1...). Permite comparar experimentos.
    Column("version", String(20), nullable=False),
    # Cuando se congelo. La pone PostgreSQL si no la enviamos.
    Column("fecha_creacion", DateTime, nullable=False, server_default=func.now()),
    # Hasta que dia usamos ventas como HISTORICO (el pasado del modelo).
    Column("fecha_corte", Date, nullable=False),
    # El SQL EXACTO que materializo las filas. Es el linaje del dataset.
    Column("consulta_sql", Text, nullable=False),
    Column("n_filas", Integer, nullable=False, default=0),
    Column("n_columnas", Integer),
    Column("descripcion", Text),
    # Filtros u opciones en texto (JSON o clave=valor). No es un JSONB a
    # proposito: en clase basta un string legible.
    Column("parametros", Text),
    # Hash de las filas: si dos snapshots tienen el mismo checksum, son iguales.
    Column("checksum", String(64)),
    Column("estado", String(20), nullable=False, default="activo"),
    CheckConstraint(
        "estado IN ('activo', 'archivado', 'error')",
        name="ck_historial_estado",
    ),
    comment="Metadatos de cada dataset congelado para entrenar modelos",
)

dataset_entrenamiento = Table(
    "dataset_entrenamiento",
    metadata,
    Column("fila_id", Integer, primary_key=True, autoincrement=True),
    # Cada fila pertenece a UN snapshot. Si borras el snapshot, se van las filas.
    Column(
        "snapshot_id",
        Integer,
        ForeignKey("historial_entrenamiento.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    ),
    # El cliente sigue existiendo en la tabla maestra (integridad referencial).
    Column("cliente_id", Integer, ForeignKey("clientes.cliente_id"), nullable=False),
    # --- features (lo que el modelo VE al predecir) ---
    Column("segmento", String(20)),
    Column("pais", String(80)),
    Column("num_compras", Integer, nullable=False),
    Column("gasto_historico", Numeric(12, 2), nullable=False),
    Column("ticket_medio", Numeric(12, 2), nullable=False),
    Column("cantidad_total", Integer, nullable=False),
    Column("dias_desde_ultima_compra", Integer),
    Column("canal_preferido", String(20)),
    Column("categoria_favorita", String(60)),
    # --- target (lo que el modelo INTENTA acertar) ---
    # Gasto del cliente en los 90 dias POSTERIORES a fecha_corte.
    # Si no compro nada en ese periodo, vale 0: eso tambien es informacion.
    Column("gasto_siguiente_periodo", Numeric(12, 2), nullable=False, default=0),
    Column("fecha_corte", Date, nullable=False),
    comment="Filas congeladas (features + target) de un snapshot de entrenamiento",
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
