"""Limpieza del dataset Online Retail con bitácora de filas antes/después por paso.

Cada paso es una función pura (recibe y devuelve DataFrames) para poder probarla de
forma aislada. ``limpiar`` las encadena en el orden justificado en el notebook de EDA
y registra el efecto de cada una.

Uso desde la raíz del proyecto (con src/ en PYTHONPATH):
    python -m segmentacion.limpieza
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from segmentacion import config

# Códigos de StockCode que no corresponden a mercancía. Se identificaron en el EDA
# a partir de su descripción: portes (POST, DOT, C2), ajustes manuales (M),
# descuentos (D), muestras (S), comisiones/cargos (BANK CHARGES, AMAZONFEE, CRUK)
# y ajuste de deuda incobrable (B). Se comparan en mayúsculas ("m" == "M").
CODIGOS_NO_PRODUCTO: frozenset[str] = frozenset(
    {"POST", "DOT", "C2", "M", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK", "B"}
)
# Bonos regalo (gift_0001_10, gift_0001_20, ...): no son productos físicos.
PREFIJOS_NO_PRODUCTO: tuple[str, ...] = ("GIFT_",)


@dataclass
class ResultadoLimpieza:
    """Salida de :func:`limpiar`."""

    ventas: pd.DataFrame
    devoluciones: pd.DataFrame
    bitacora: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Predicados
# ---------------------------------------------------------------------------
def es_cancelacion(df: pd.DataFrame) -> pd.Series:
    """Marca las líneas cuya factura empieza por 'C' (cancelación según UCI)."""
    return df["InvoiceNo"].str.upper().str.startswith("C").fillna(False).astype(bool)


def es_codigo_no_producto(codigos: pd.Series) -> pd.Series:
    """Marca los StockCode que no son mercancía (portes, comisiones, bonos, etc.)."""
    mayus = codigos.str.upper().str.strip()
    en_lista = mayus.isin(CODIGOS_NO_PRODUCTO)
    con_prefijo = mayus.str.startswith(PREFIJOS_NO_PRODUCTO)
    return (en_lista | con_prefijo).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Pasos de limpieza
# ---------------------------------------------------------------------------
def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas idénticas en las 8 columnas (registros cargados dos veces)."""
    return df.drop_duplicates()


def eliminar_sin_cliente(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina líneas sin CustomerID y convierte el ID a entero.

    La segmentación es a nivel de cliente: una línea sin ID no puede asignarse
    a ningún cliente.
    """
    salida = df[df["CustomerID"].notna()].copy()
    salida["CustomerID"] = salida["CustomerID"].astype("int64")
    return salida


def eliminar_codigos_no_producto(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina líneas cuyo StockCode no corresponde a un producto."""
    return df[~es_codigo_no_producto(df["StockCode"])]


def separar_cancelaciones(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa ventas y cancelaciones.

    Returns:
        Tupla ``(ventas, cancelaciones)``. Las cancelaciones no se descartan:
        se usan para emparejar compras anuladas y para la tasa de devoluciones.
    """
    mascara = es_cancelacion(df)
    return df[~mascara], df[mascara]


def eliminar_no_positivos(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina líneas de venta con cantidad o precio unitario menor o igual a cero."""
    return df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]


def emparejar_compras_anuladas(
    ventas: pd.DataFrame, cancelaciones: pd.DataFrame
) -> tuple[pd.Index, pd.Series]:
    """Empareja cada cancelación con la compra que anula exactamente.

    Una compra se considera anulada si existe una cancelación posterior (o en el
    mismo minuto) del mismo cliente, mismo producto, misma cantidad (en valor
    absoluto) y mismo precio unitario. Se toma la compra más reciente anterior a
    la cancelación y el emparejamiento es 1 a 1: si dos cancelaciones apuntan a
    la misma compra, solo la primera queda emparejada.

    Args:
        ventas: líneas de venta (cantidad > 0) con índice único.
        cancelaciones: líneas de cancelación (cantidad < 0) con índice único.

    Returns:
        Tupla ``(indices_ventas_anuladas, cancelacion_emparejada)``, donde la
        segunda es una Serie booleana alineada con ``cancelaciones``.
    """
    claves = ["CustomerID", "StockCode", "UnitPrice"]
    izq = (
        cancelaciones.assign(Cantidad=-cancelaciones["Quantity"])[claves + ["InvoiceDate", "Cantidad"]]
        .rename_axis("idx_cancelacion")
        .reset_index()
        .sort_values("InvoiceDate")
    )
    der = (
        ventas.rename(columns={"Quantity": "Cantidad"})[claves + ["InvoiceDate", "Cantidad"]]
        .rename(columns={"InvoiceDate": "FechaVenta"})
        .rename_axis("idx_venta")
        .reset_index()
        .sort_values("FechaVenta")
    )
    cruce = pd.merge_asof(
        izq,
        der,
        left_on="InvoiceDate",
        right_on="FechaVenta",
        by=claves + ["Cantidad"],
        direction="backward",
    )
    cruce = cruce.dropna(subset=["idx_venta"])
    cruce = cruce.drop_duplicates(subset="idx_venta", keep="first")

    anuladas = pd.Index(cruce["idx_venta"].astype("int64"))
    emparejada = pd.Series(cancelaciones.index.isin(cruce["idx_cancelacion"]), index=cancelaciones.index)
    return anuladas, emparejada


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def _registro(paso: str, antes: int, despues: int, justificacion: str) -> dict[str, object]:
    eliminadas = antes - despues
    return {
        "paso": paso,
        "filas_antes": antes,
        "filas_despues": despues,
        "filas_eliminadas": eliminadas,
        "pct_eliminado": round(100 * eliminadas / antes, 2) if antes else 0.0,
        "justificacion": justificacion,
    }


def limpiar(crudo: pd.DataFrame) -> ResultadoLimpieza:
    """Aplica todos los pasos de limpieza y devuelve ventas, devoluciones y bitácora.

    Orden: duplicados -> sin cliente -> códigos no producto -> separar
    cancelaciones -> cantidad/precio <= 0 -> compras anuladas.

    Args:
        crudo: dataset original con tipos normalizados (ver ``carga``).

    Returns:
        ResultadoLimpieza con las ventas limpias (con columna ``Importe``),
        las cancelaciones de producto (con columna ``EmparejaCompra``) y la
        bitácora de filas por paso.
    """
    df = crudo.reset_index(drop=True)
    pasos: list[dict[str, object]] = []

    n = len(df)
    df = eliminar_duplicados(df)
    pasos.append(_registro("1. Duplicados exactos", n, len(df),
                           "Filas idénticas en todas las columnas: doble registro de la misma línea."))

    n = len(df)
    df = eliminar_sin_cliente(df)
    pasos.append(_registro("2. Sin CustomerID", n, len(df),
                           "La unidad de análisis es el cliente; sin ID la línea no se puede asignar."))

    n = len(df)
    df = eliminar_codigos_no_producto(df)
    pasos.append(_registro("3. Códigos que no son producto", n, len(df),
                           "Portes, comisiones, descuentos, ajustes manuales y bonos no reflejan "
                           "comportamiento de compra de mercancía."))

    n = len(df)
    ventas, cancelaciones = separar_cancelaciones(df)
    pasos.append(_registro("4. Separar cancelaciones (factura 'C')", n, len(ventas),
                           "Las cancelaciones no son compras; se guardan aparte para emparejar "
                           "anulaciones y calcular la tasa de devoluciones."))

    n = len(ventas)
    ventas = eliminar_no_positivos(ventas)
    pasos.append(_registro("5. Cantidad o precio <= 0", n, len(ventas),
                           "Una venta válida requiere cantidad y precio positivos (precio 0 = "
                           "regalos/ajustes sin valor monetario)."))

    n = len(ventas)
    anuladas, emparejada = emparejar_compras_anuladas(ventas, cancelaciones)
    ventas = ventas.drop(index=anuladas)
    pasos.append(_registro("6. Compras anuladas por cancelación exacta", n, len(ventas),
                           "Compra revertida después por el mismo cliente (mismo producto, cantidad "
                           "y precio): la transacción no se concretó y, de dejarla, inflaría el gasto."))

    ventas = ventas.assign(Importe=ventas["Quantity"] * ventas["UnitPrice"])
    devoluciones = cancelaciones.assign(
        Importe=cancelaciones["Quantity"] * cancelaciones["UnitPrice"],
        EmparejaCompra=emparejada,
    )
    bitacora = pd.DataFrame(pasos)
    return ResultadoLimpieza(ventas=ventas, devoluciones=devoluciones, bitacora=bitacora)


def guardar(resultado: ResultadoLimpieza) -> None:
    """Persiste ventas, devoluciones y bitácora en data/processed."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    resultado.ventas.to_parquet(config.PARQUET_LIMPIO)
    resultado.devoluciones.to_parquet(config.PARQUET_DEVOLUCIONES)
    resultado.bitacora.to_csv(config.CSV_BITACORA, index=False, encoding="utf-8")


if __name__ == "__main__":
    from segmentacion.carga import obtener_datos_crudos

    res = limpiar(obtener_datos_crudos())
    guardar(res)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(res.bitacora.drop(columns="justificacion").to_string(index=False))
    print(f"\nVentas limpias: {len(res.ventas):,} | Clientes: {res.ventas['CustomerID'].nunique():,}")
    print(f"Devoluciones guardadas: {len(res.devoluciones):,} "
          f"(emparejadas con compra: {int(res.devoluciones['EmparejaCompra'].sum()):,})")
