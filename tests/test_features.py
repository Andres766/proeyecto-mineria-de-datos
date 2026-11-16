"""Pruebas de la tasa de devoluciones con casos pequeños de resultado conocido."""

from __future__ import annotations

import pandas as pd
import pytest

from segmentacion.features import devoluciones_computables, tasa_devoluciones

T0 = pd.Timestamp("2011-03-01 10:00")
DIA = pd.Timedelta(days=1)


def ventas(filas: list[tuple]) -> pd.DataFrame:
    """Ventas limpias a partir de (CustomerID, StockCode, Quantity, InvoiceDate, UnitPrice)."""
    df = pd.DataFrame(filas, columns=["CustomerID", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"])
    df["StockCode"] = df["StockCode"].astype("string")
    return df.assign(Importe=df["Quantity"] * df["UnitPrice"])


def devoluciones(filas: list[tuple]) -> pd.DataFrame:
    """Cancelaciones a partir de (CustomerID, StockCode, Quantity<0, InvoiceDate, UnitPrice, EmparejaCompra)."""
    df = pd.DataFrame(
        filas, columns=["CustomerID", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "EmparejaCompra"]
    )
    df["StockCode"] = df["StockCode"].astype("string")
    return df.assign(Importe=df["Quantity"] * df["UnitPrice"]).set_index(pd.RangeIndex(100, 100 + len(df)))


def tasa(filas_ventas: list[tuple], filas_dev: list[tuple]) -> dict[int, float]:
    return tasa_devoluciones(ventas(filas_ventas), devoluciones(filas_dev)).round(6).to_dict()


def test_devolucion_parcial_trazable():
    # Compra 10 x 2 = 20; devuelve 4 x 2 = 8 -> 0,4.
    assert tasa([(1, "A", 10, T0, 2.0)], [(1, "A", -4, T0 + DIA, 2.0, False)]) == {1: 0.4}


def test_cancelacion_emparejada_no_cuenta():
    # La compra anulada ya no está en ventas; su cancelación no debe sumar al numerador.
    assert tasa([(1, "B", 5, T0, 1.0)], [(1, "A", -10, T0 + DIA, 2.0, True)]) == {1: 0.0}


@pytest.mark.parametrize(
    "devolucion, motivo",
    [
        ((1, "Z", -3, T0 + DIA, 2.0, False), "producto nunca comprado en el periodo"),
        ((1, "A", -3, T0 - DIA, 2.0, False), "devolución anterior a la compra"),
        ((2, "A", -3, T0 + DIA, 2.0, False), "otro cliente"),
    ],
)
def test_devolucion_no_trazable_computa_cero(devolucion, motivo):
    res = devoluciones_computables(ventas([(1, "A", 10, T0, 2.0)]), devoluciones([devolucion]))
    assert res["UnidadesComputables"].tolist() == [0], motivo


def test_no_se_devuelve_mas_de_lo_comprado():
    # Compra 5, devuelve 8: solo computan 5 -> tasa 1,0.
    assert tasa([(1, "A", 5, T0, 2.0)], [(1, "A", -8, T0 + DIA, 2.0, False)]) == {1: 1.0}


def test_tope_acumulado_en_varias_devoluciones():
    # Compra 10; devuelve 6 y luego 6: computan 6 y 4.
    res = devoluciones_computables(
        ventas([(1, "A", 10, T0, 1.0)]),
        devoluciones([(1, "A", -6, T0 + DIA, 1.0, False), (1, "A", -6, T0 + 2 * DIA, 1.0, False)]),
    )
    assert res["UnidadesComputables"].tolist() == [6, 4]


def test_compra_posterior_no_habilita_devolucion_anterior():
    # Devuelve 5 el día 1 y compra 5 el día 2: la devolución no es trazable.
    res = devoluciones_computables(
        ventas([(1, "A", 5, T0 + 2 * DIA, 1.0)]), devoluciones([(1, "A", -5, T0 + DIA, 1.0, False)])
    )
    assert res["UnidadesComputables"].tolist() == [0]


def test_compras_acumuladas_amplian_el_tope():
    # Compra 3 y luego 4 (total 7 antes de devolver); devuelve 7 -> computan 7.
    res = devoluciones_computables(
        ventas([(1, "A", 3, T0, 1.0), (1, "A", 4, T0 + DIA, 1.0)]),
        devoluciones([(1, "A", -7, T0 + 2 * DIA, 1.0, False)]),
    )
    assert res["UnidadesComputables"].tolist() == [7]


def test_cliente_sin_devoluciones_tiene_tasa_cero():
    assert tasa(
        [(1, "A", 10, T0, 2.0), (2, "A", 1, T0, 2.0)], [(1, "A", -5, T0 + DIA, 2.0, False)]
    ) == {1: 0.5, 2: 0.0}  # 5 x 2 = 10 sobre 20


def test_tasa_se_recorta_a_uno_si_la_devolucion_es_mas_cara():
    # Compra 5 x 1 = 5; devuelve 5 unidades a 3 £ (precio mayor): 15 / 5 = 3 -> se recorta a 1.
    assert tasa([(1, "A", 5, T0, 1.0)], [(1, "A", -5, T0 + DIA, 3.0, False)]) == {1: 1.0}
