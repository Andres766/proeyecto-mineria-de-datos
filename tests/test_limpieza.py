"""Pruebas del emparejamiento de compras anuladas (exactas 1 a 1 y manuales 'M').

Cada caso usa unas pocas líneas construidas a mano cuyo resultado se conoce de antemano.
"""

from __future__ import annotations

import pandas as pd
import pytest

from segmentacion.limpieza import (
    emparejar_anulaciones_manuales,
    emparejar_compras_anuladas,
    es_codigo_no_producto,
    separar_cancelaciones,
)

T0 = pd.Timestamp("2011-03-01 10:00")
MIN = pd.Timedelta(minutes=1)


def transacciones(filas: list[tuple]) -> pd.DataFrame:
    """Construye un DataFrame con el esquema del dataset a partir de tuplas
    (InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID)."""
    df = pd.DataFrame(filas, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"])
    return df.assign(
        InvoiceNo=df["InvoiceNo"].astype("string"),
        StockCode=df["StockCode"].astype("string"),
        Quantity=df["Quantity"].astype("int64"),
        InvoiceDate=pd.to_datetime(df["InvoiceDate"]),
        UnitPrice=df["UnitPrice"].astype("float64"),
        CustomerID=df["CustomerID"].astype("int64"),
        Description="ARTICULO",
        Country="United Kingdom",
    )


def emparejar(filas: list[tuple]) -> tuple[list[int], list[bool]]:
    """Separa ventas/cancelaciones y devuelve (índices anulados, emparejada por cancelación)."""
    ventas, cancelaciones = separar_cancelaciones(transacciones(filas))
    anuladas, emparejada = emparejar_compras_anuladas(ventas, cancelaciones)
    return sorted(anuladas.tolist()), emparejada.tolist()


# ---------------------------------------------------------------------------
# Cancelación exacta (mismo cliente, producto, cantidad y precio)
# ---------------------------------------------------------------------------
def test_cancelacion_exacta_anula_la_compra():
    anuladas, emparejada = emparejar([
        ("100", "A", 10, T0, 2.0, 1),
        ("C101", "A", -10, T0 + 5 * MIN, 2.0, 1),
    ])
    assert anuladas == [0]
    assert emparejada == [True]


@pytest.mark.parametrize(
    "cancelacion, motivo",
    [
        (("C101", "A", -10, T0 - 5 * MIN, 2.0, 1), "anterior a la compra"),
        (("C101", "A", -10, T0 + 5 * MIN, 2.5, 1), "otro precio"),
        (("C101", "A", -4, T0 + 5 * MIN, 2.0, 1), "devolución parcial (otra cantidad)"),
        (("C101", "B", -10, T0 + 5 * MIN, 2.0, 1), "otro producto"),
        (("C101", "A", -10, T0 + 5 * MIN, 2.0, 2), "otro cliente"),
    ],
)
def test_no_empareja_si_no_es_exacta(cancelacion, motivo):
    anuladas, emparejada = emparejar([("100", "A", 10, T0, 2.0, 1), cancelacion])
    assert anuladas == [], motivo
    assert emparejada == [False], motivo


def test_mismo_minuto_cuenta_como_posterior():
    anuladas, _ = emparejar([
        ("100", "A", 10, T0, 2.0, 1),
        ("C101", "A", -10, T0, 2.0, 1),
    ])
    assert anuladas == [0]


def test_una_compra_solo_puede_anularse_una_vez():
    # Dos cancelaciones idénticas y una sola compra: solo la primera cancelación empareja.
    anuladas, emparejada = emparejar([
        ("100", "A", 10, T0, 2.0, 1),
        ("C101", "A", -10, T0 + 5 * MIN, 2.0, 1),
        ("C102", "A", -10, T0 + 9 * MIN, 2.0, 1),
    ])
    assert anuladas == [0]
    assert emparejada == [True, False]


def test_se_anula_la_compra_mas_reciente_anterior():
    anuladas, _ = emparejar([
        ("100", "A", 10, T0, 2.0, 1),
        ("101", "A", 10, T0 + 60 * MIN, 2.0, 1),
        ("C102", "A", -10, T0 + 90 * MIN, 2.0, 1),
    ])
    assert anuladas == [1]


def test_dos_compras_y_dos_cancelaciones_identicas_se_emparejan_todas():
    # Emparejamiento 1 a 1 real: cada cancelación anula una compra distinta.
    anuladas, emparejada = emparejar([
        ("100", "A", 10, T0, 2.0, 1),
        ("101", "A", 10, T0 + 60 * MIN, 2.0, 1),
        ("C102", "A", -10, T0 + 90 * MIN, 2.0, 1),
        ("C103", "A", -10, T0 + 95 * MIN, 2.0, 1),
    ])
    assert anuladas == [0, 1]
    assert emparejada == [True, True]


# ---------------------------------------------------------------------------
# Anulación manual 'M' (mismo cliente, mismo importe exacto, < 24 h)
# ---------------------------------------------------------------------------
def anular_manual(filas_ventas: list[tuple], filas_manuales: list[tuple]) -> list[int]:
    df = transacciones(filas_ventas + filas_manuales)
    ventas, cancelaciones = separar_cancelaciones(df)
    manuales = cancelaciones[cancelaciones["StockCode"] == "M"]
    return sorted(emparejar_anulaciones_manuales(ventas, manuales).tolist())


VENTA_60 = ("100", "A", 60, T0, 649.5, 1)  # importe 38.970


def test_manual_mismo_importe_en_minutos_anula():
    assert anular_manual([VENTA_60], [("C101", "M", -1, T0 + 3 * MIN, 38970.0, 1)]) == [0]


@pytest.mark.parametrize(
    "manual, motivo",
    [
        (("C101", "M", -1, T0 + pd.Timedelta(hours=25), 38970.0, 1), "más de 24 h después"),
        (("C101", "M", -1, T0 + 3 * MIN, 38969.0, 1), "otro importe"),
        (("C101", "M", -1, T0 - 3 * MIN, 38970.0, 1), "anterior a la compra"),
        (("C101", "M", -1, T0 + 3 * MIN, 38970.0, 2), "otro cliente"),
    ],
)
def test_manual_no_anula_si_no_cumple(manual, motivo):
    assert anular_manual([VENTA_60], [manual]) == [], motivo


def test_manual_importe_compuesto_por_cantidad_y_precio():
    # -2 x 10,25 = 20,50 debe emparejar con una línea de 4 x 5,125 = 20,50.
    assert anular_manual([("100", "A", 4, T0, 5.125, 1)], [("C101", "M", -2, T0 + MIN, 10.25, 1)]) == [0]


def test_manual_es_uno_a_uno():
    # Dos anulaciones manuales del mismo importe y una sola compra: solo se anula una vez.
    assert anular_manual(
        [VENTA_60],
        [("C101", "M", -1, T0 + 3 * MIN, 38970.0, 1), ("C102", "M", -1, T0 + 4 * MIN, 38970.0, 1)],
    ) == [0]


def test_manual_dos_compras_dos_anulaciones():
    assert anular_manual(
        [VENTA_60, ("101", "A", 60, T0 + MIN, 649.5, 1)],
        [("C102", "M", -1, T0 + 3 * MIN, 38970.0, 1), ("C103", "M", -1, T0 + 4 * MIN, 38970.0, 1)],
    ) == [0, 1]


# ---------------------------------------------------------------------------
# Códigos que no son producto
# ---------------------------------------------------------------------------
def test_codigos_no_producto():
    codigos = pd.Series(["POST", "m", "gift_0001_20", "AMAZONFEE", "85123A", "DCGS0003", "PADS"], dtype="string")
    assert es_codigo_no_producto(codigos).tolist() == [True, True, True, True, False, False, False]
