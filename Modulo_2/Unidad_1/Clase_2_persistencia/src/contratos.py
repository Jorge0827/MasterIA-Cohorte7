"""
Contratos de datos: definicion de lo que ESPERAMOS recibir en cada CSV.

Un contrato responde tres preguntas:

  1. Que columnas debe tener el archivo?
  2. Que tipo de dato es cada columna?
  3. Que reglas de negocio debe cumplir?

Si el CSV no cumple el contrato, el pipeline NO debe cargar a PostgreSQL.
Eso se llama "fail fast": fallar pronto, antes de ensuciar la base de datos.

Este modulo es nuevo de la Clase 3. No modifica el esquema de PostgreSQL
de la Clase 2: describe el archivo de ENTRADA, no la tabla de destino.
"""

from dataclasses import dataclass, field


# @dataclass genera automaticamente __init__, __repr__, etc.
# Es una forma limpia de agrupar configuracion sin escribir mucho codigo.
@dataclass
class ColumnaContrato:
    """Describe una columna esperada en el CSV."""

    nombre: str
    tipo: str  # "int", "float", "str", "date", "bool"
    obligatoria: bool = True
    valores_permitidos: list | None = None
    minimo: float | None = None
    maximo: float | None = None


@dataclass
class ContratoDatos:
    """Contrato completo de un archivo CSV."""

    tabla: str
    archivo_csv: str
    columnas: list[ColumnaContrato] = field(default_factory=list)

    @property
    def nombres_columnas(self) -> list[str]:
        """Lista de nombres de columnas esperadas (util para comparar esquemas)."""
        return [columna.nombre for columna in self.columnas]


# -----------------------------------------------------------------------------
# Contratos de las tres tablas de negocio del proyecto
# -----------------------------------------------------------------------------

CONTRATO_CLIENTES = ContratoDatos(
    tabla="clientes",
    archivo_csv="clientes.csv",
    columnas=[
        ColumnaContrato("cliente_id", "int", minimo=1),
        ColumnaContrato("nombre", "str"),
        ColumnaContrato("email", "str"),
        ColumnaContrato("ciudad", "str", obligatoria=False),
        ColumnaContrato("pais", "str", obligatoria=False),
        ColumnaContrato(
            "segmento",
            "str",
            valores_permitidos=["bronce", "plata", "oro", "platino"],
        ),
        ColumnaContrato("fecha_registro", "date"),
    ],
)

CONTRATO_PRODUCTOS = ContratoDatos(
    tabla="productos",
    archivo_csv="productos.csv",
    columnas=[
        ColumnaContrato("producto_id", "int", minimo=1),
        ColumnaContrato("nombre", "str"),
        ColumnaContrato("categoria", "str"),
        ColumnaContrato("precio", "float", minimo=0),
        ColumnaContrato("coste", "float", minimo=0),
        ColumnaContrato("stock", "int", minimo=0),
        ColumnaContrato("activo", "bool"),
    ],
)

CONTRATO_VENTAS = ContratoDatos(
    tabla="ventas",
    archivo_csv="ventas.csv",
    columnas=[
        ColumnaContrato("venta_id", "int", minimo=1),
        ColumnaContrato("cliente_id", "int", minimo=1),
        ColumnaContrato("producto_id", "int", minimo=1),
        ColumnaContrato("fecha_venta", "date"),
        ColumnaContrato("cantidad", "int", minimo=1),
        ColumnaContrato("precio_unitario", "float", minimo=0),
        ColumnaContrato("descuento", "float", minimo=0, maximo=1),
        ColumnaContrato("total", "float", minimo=0),
        ColumnaContrato(
            "canal",
            "str",
            valores_permitidos=["web", "app", "tienda", "telefono"],
        ),
    ],
)

# Diccionario para buscar contrato por nombre de tabla.
CONTRATOS = {
    "clientes": CONTRATO_CLIENTES,
    "productos": CONTRATO_PRODUCTOS,
    "ventas": CONTRATO_VENTAS,
}