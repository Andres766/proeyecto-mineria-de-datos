"""Ingeniería de características a nivel de cliente.

Tasa de devoluciones
--------------------
Definición coherente con la limpieza (ver ``limpieza.py``):

    tasa_devoluciones(cliente) = importe devuelto computable / importe comprado

* **Denominador — importe comprado:** suma de ``Importe`` de las ventas limpias del
  cliente. Ya excluye las compras anuladas del paso 6 de la limpieza.
* **Numerador — importe devuelto computable:** importe de las cancelaciones de
  producto del cliente que cumplen dos condiciones:

  1. **No están emparejadas** con una compra anulada (``EmparejaCompra == False``).
     Las emparejadas y sus compras se eliminaron juntas en la limpieza: contarlas
     aquí inflaría la tasa con transacciones que nunca se concretaron.
  2. **Son trazables dentro del periodo:** solo computan las unidades devueltas que
     caben en lo que el cliente compró de ese producto *antes* de la devolución
     (en las ventas limpias). Por cada par cliente-producto, en orden cronológico,
     las unidades computables acumuladas son
     ``min(unidades devueltas acumuladas, unidades compradas hasta esa fecha)``.

  Consecuencias: una devolución de un producto que el cliente no compró antes en el
  periodo (compra anterior al 01-12-2010, fuera de los datos) computa 0, y nadie
  puede devolver más unidades de las que compró.

* **Recorte a [0, 1]:** el tope anterior es en unidades; en importe la tasa podría
  superar 1 si una devolución tuviera un precio unitario mayor que el de su compra.
  Por robustez la tasa se recorta a [0, 1] (``clip``). En los datos actuales ningún
  cliente supera 1, así que el recorte no cambia ningún valor; protege ante datos nuevos.
"""

from __future__ import annotations

import pandas as pd


def devoluciones_computables(ventas: pd.DataFrame, devoluciones: pd.DataFrame) -> pd.DataFrame:
    """Calcula, para cada cancelación, cuántas unidades e importe computan como devolución.

    Args:
        ventas: ventas limpias (``CustomerID``, ``StockCode``, ``Quantity``, ``InvoiceDate``).
        devoluciones: cancelaciones de producto con ``EmparejaCompra`` (salida de la limpieza).

    Returns:
        Copia de ``devoluciones`` con las columnas:
        ``UnidadesDevueltas`` (positivas), ``CompradoAntes`` (unidades compradas del mismo
        producto por el cliente hasta la fecha de la devolución), ``UnidadesComputables`` e
        ``ImporteComputable`` (positivo). Las emparejadas computan 0.
    """
    dev = devoluciones.copy()
    dev["UnidadesDevueltas"] = -dev["Quantity"]

    # Unidades compradas acumuladas por cliente-producto a lo largo del tiempo.
    compras = (
        ventas[["CustomerID", "StockCode", "InvoiceDate", "Quantity"]]
        .sort_values("InvoiceDate")
        .assign(CompradoAntes=lambda d: d.groupby(["CustomerID", "StockCode"])["Quantity"].cumsum())
        .rename(columns={"InvoiceDate": "FechaCompra"})
        .drop(columns="Quantity")
    )
    # Mismos tipos (texto y resolución de fecha) en ambas tablas para que merge_asof acepte las claves.
    compras["StockCode"] = compras["StockCode"].astype("string")
    dev["StockCode"] = dev["StockCode"].astype("string")
    compras["FechaCompra"] = compras["FechaCompra"].astype("datetime64[ns]")
    dev["InvoiceDate"] = dev["InvoiceDate"].astype("datetime64[ns]")

    libres = dev[~dev["EmparejaCompra"]].rename_axis("idx_dev").reset_index().sort_values("InvoiceDate")
    libres = pd.merge_asof(
        libres,
        compras,
        left_on="InvoiceDate",
        right_on="FechaCompra",
        by=["CustomerID", "StockCode"],
        direction="backward",
    )
    libres["CompradoAntes"] = libres["CompradoAntes"].fillna(0).astype("int64")

    # Tope acumulado: lo devuelto no puede superar lo comprado hasta cada fecha.
    libres = libres.sort_values(["CustomerID", "StockCode", "InvoiceDate", "idx_dev"])
    grupo = libres.groupby(["CustomerID", "StockCode"])
    devuelto_acum = grupo["UnidadesDevueltas"].cumsum()
    permitido_acum = devuelto_acum.where(devuelto_acum < libres["CompradoAntes"], libres["CompradoAntes"])
    # min() de dos series no decrecientes es no decreciente; la diferencia da las unidades de cada línea.
    libres["UnidadesComputables"] = (
        permitido_acum - permitido_acum.groupby([libres["CustomerID"], libres["StockCode"]]).shift(fill_value=0)
    ).clip(lower=0).astype("int64")

    libres = libres.set_index("idx_dev")
    dev["CompradoAntes"] = libres["CompradoAntes"].reindex(dev.index).fillna(0).astype("int64")
    dev["UnidadesComputables"] = libres["UnidadesComputables"].reindex(dev.index).fillna(0).astype("int64")
    dev["ImporteComputable"] = dev["UnidadesComputables"] * dev["UnitPrice"]
    return dev


def tasa_devoluciones(ventas: pd.DataFrame, devoluciones: pd.DataFrame) -> pd.Series:
    """Tasa de devoluciones por cliente = importe devuelto computable / importe comprado.

    Args:
        ventas: ventas limpias con columna ``Importe``.
        devoluciones: cancelaciones de producto con ``EmparejaCompra``.

    Returns:
        Serie indexada por ``CustomerID`` (todos los clientes de ``ventas``) recortada a
        [0, 1]; 0 para clientes sin devoluciones computables.
    """
    comprado = ventas.groupby("CustomerID")["Importe"].sum()
    computable = devoluciones_computables(ventas, devoluciones)
    devuelto = computable.groupby("CustomerID")["ImporteComputable"].sum().reindex(comprado.index, fill_value=0.0)
    return (devuelto / comprado).clip(lower=0.0, upper=1.0).rename("tasa_devoluciones")


# ---------------------------------------------------------------------------
# RFM y variables de comportamiento
# ---------------------------------------------------------------------------
def fecha_referencia(ventas: pd.DataFrame) -> pd.Timestamp:
    """Fecha de referencia para la Recency: día siguiente a la última fecha del dataset.

    Se toma el día (sin hora) de la última transacción y se suma 1 día, de modo que la
    compra más reciente tenga Recency = 1 y ningún cliente quede en 0.
    """
    return ventas["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def calcular_rfm(ventas: pd.DataFrame, referencia: pd.Timestamp) -> pd.DataFrame:
    """Calcula Recency, Frequency y Monetary por cliente.

    * **Recency:** días entre la fecha de referencia y el día de la última compra.
    * **Frequency:** número de facturas distintas.
    * **Monetary:** gasto total (suma de ``Importe`` de las ventas limpias).

    Args:
        ventas: ventas limpias con ``CustomerID``, ``InvoiceNo``, ``InvoiceDate`` e ``Importe``.
        referencia: fecha de referencia (ver :func:`fecha_referencia`).

    Returns:
        DataFrame indexado por ``CustomerID`` con columnas ``recency``, ``frequency``, ``monetary``.
    """
    agg = ventas.groupby("CustomerID").agg(
        ultima=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Importe", "sum"),
    )
    agg["recency"] = (referencia - agg["ultima"].dt.normalize()).dt.days
    return agg[["recency", "frequency", "monetary"]]


def calcular_variables_cliente(
    ventas: pd.DataFrame, devoluciones: pd.DataFrame, referencia: pd.Timestamp
) -> pd.DataFrame:
    """Construye la tabla de variables por cliente: RFM + comportamiento + país.

    Variables adicionales:

    * ``ticket_promedio``: gasto medio por factura (monetary / frequency).
    * ``productos_distintos``: número de StockCode distintos comprados.
    * ``antiguedad``: días entre la primera compra y la fecha de referencia.
    * ``dias_entre_compras``: tiempo medio (días) entre facturas consecutivas,
      (última − primera) / (frequency − 1). **No está definida (NaN) si frequency = 1.**
    * ``unidades_por_factura``: unidades totales / frequency.
    * ``tasa_devoluciones``: ver :func:`tasa_devoluciones`.
    * ``pais``: país con más facturas del cliente (solo para perfilar).

    Returns:
        DataFrame indexado por ``CustomerID``.
    """
    tabla = calcular_rfm(ventas, referencia)
    extra = ventas.groupby("CustomerID").agg(
        primera=("InvoiceDate", "min"),
        ultima=("InvoiceDate", "max"),
        productos_distintos=("StockCode", "nunique"),
        unidades=("Quantity", "sum"),
    )
    tabla["ticket_promedio"] = tabla["monetary"] / tabla["frequency"]
    tabla["productos_distintos"] = extra["productos_distintos"]
    tabla["antiguedad"] = (referencia - extra["primera"].dt.normalize()).dt.days
    lapso_dias = (extra["ultima"] - extra["primera"]).dt.total_seconds() / 86_400
    tabla["dias_entre_compras"] = (lapso_dias / (tabla["frequency"] - 1)).where(tabla["frequency"] > 1)
    tabla["unidades_por_factura"] = extra["unidades"] / tabla["frequency"]
    tabla["tasa_devoluciones"] = tasa_devoluciones(ventas, devoluciones)
    facturas_pais = ventas.drop_duplicates("InvoiceNo").groupby(["CustomerID", "Country"]).size()
    tabla["pais"] = facturas_pais.sort_values(ascending=False).reset_index().drop_duplicates("CustomerID").set_index("CustomerID")["Country"]
    return tabla


# ---------------------------------------------------------------------------
# Selección y preprocesamiento para el clustering
# ---------------------------------------------------------------------------
# Variables que entran al clustering (justificación en notebooks/02_features.ipynb):
# monetary = frequency x ticket_promedio, así que en escala log monetary es una
# combinación casi lineal de las otras dos; se usa frequency + ticket_promedio en su lugar.
VARIABLES_CLUSTERING: list[str] = ["recency", "frequency", "ticket_promedio", "productos_distintos"]

# Variables que solo se usan para perfilar los segmentos.
VARIABLES_PERFIL: list[str] = [
    "monetary", "antiguedad", "dias_entre_compras", "unidades_por_factura", "tasa_devoluciones", "pais",
]

PERCENTILES_WINSOR: tuple[float, float] = (1.0, 99.0)


def calcular_vif(datos: pd.DataFrame) -> pd.Series:
    """Factor de inflación de la varianza de cada columna: 1 / (1 - R²).

    R² es el de la regresión lineal (con intercepto) de la columna sobre las demás.
    VIF > 5 indica colinealidad alta; > 10, severa.
    """
    import numpy as np

    matriz = datos.to_numpy(dtype=float)
    resultado = {}
    for i, columna in enumerate(datos.columns):
        y = matriz[:, i]
        diseno = np.column_stack([np.ones(len(matriz)), np.delete(matriz, i, axis=1)])
        coef, *_ = np.linalg.lstsq(diseno, y, rcond=None)
        r2 = 1 - ((y - diseno @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        resultado[columna] = 1 / (1 - r2) if r2 < 1 else float("inf")
    return pd.Series(resultado, name="vif")


def construir_preprocesador(datos: pd.DataFrame, winsorizar: bool = True):
    """Crea y ajusta el pipeline log1p -> winsorizado p1-p99 -> StandardScaler.

    El winsorizado recorta cada variable (ya en escala log) a sus percentiles 1 y 99
    calculados sobre ``datos``; los límites quedan fijos dentro del pipeline. Solo usa
    objetos de scikit-learn y funciones de numpy, de modo que el ``.joblib`` se puede
    cargar sin importar este paquete.

    Args:
        datos: variables de clustering en escala original (columnas ``VARIABLES_CLUSTERING``).
        winsorizar: si es False se omite el recorte (útil para comparar).

    Returns:
        ``sklearn.pipeline.Pipeline`` ya ajustado sobre ``datos``.
    """
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    pasos = [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]
    if winsorizar:
        en_log = np.log1p(datos.to_numpy(dtype=float))
        bajo, alto = np.percentile(en_log, PERCENTILES_WINSOR, axis=0)
        pasos.append(("winsor", FunctionTransformer(
            np.clip, kw_args={"a_min": bajo, "a_max": alto}, feature_names_out="one-to-one")))
    pasos.append(("escalado", StandardScaler()))
    return Pipeline(pasos).fit(datos)

