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
  puede devolver más unidades de las que compró. El tope es en unidades: en importe
  la tasa solo podría superar 1 si una devolución tuviera un precio unitario mayor
  que el de la compra (el máximo observado se verifica en el notebook de features).
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
        Serie indexada por ``CustomerID`` (todos los clientes de ``ventas``), no negativa;
        0 para clientes sin devoluciones computables.
    """
    comprado = ventas.groupby("CustomerID")["Importe"].sum()
    computable = devoluciones_computables(ventas, devoluciones)
    devuelto = computable.groupby("CustomerID")["ImporteComputable"].sum().reindex(comprado.index, fill_value=0.0)
    return (devuelto / comprado).rename("tasa_devoluciones")
