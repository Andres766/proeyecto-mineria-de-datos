"""Pruebas de la tasa de devoluciones con casos pequeños de resultado conocido."""

from __future__ import annotations

import numpy as np
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


# ---------------------------------------------------------------------------
# RFM y variables de comportamiento
# ---------------------------------------------------------------------------
from segmentacion.features import calcular_rfm, calcular_variables_cliente, fecha_referencia  # noqa: E402


def ventas_rfm() -> pd.DataFrame:
    """Cliente 1: 3 facturas (días 1, 11 y 31, a distintas horas); cliente 2: 1 factura."""
    filas = [
        ("F1", 1, "A", 2, "2011-01-01 09:00", 5.0, "France"),
        ("F1", 1, "B", 1, "2011-01-01 09:00", 10.0, "France"),
        ("F2", 1, "A", 4, "2011-01-11 18:00", 5.0, "France"),
        ("F3", 1, "C", 10, "2011-01-31 09:00", 1.0, "Germany"),
        ("F4", 2, "A", 1, "2011-01-21 12:00", 5.0, "Spain"),
    ]
    df = pd.DataFrame(filas, columns=["InvoiceNo", "CustomerID", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.assign(Importe=df["Quantity"] * df["UnitPrice"])


def test_fecha_referencia_es_dia_siguiente_al_ultimo():
    assert fecha_referencia(ventas_rfm()) == pd.Timestamp("2011-02-01")


def test_rfm():
    v = ventas_rfm()
    rfm = calcular_rfm(v, fecha_referencia(v))
    assert rfm.loc[1].tolist() == [1, 3, 50.0]   # recency 1 día, 3 facturas, 10+10+20+10
    assert rfm.loc[2].tolist() == [11, 1, 5.0]


def test_variables_cliente():
    v = ventas_rfm()
    sin_dev = pd.DataFrame({"CustomerID": pd.Series(dtype="int64"), "StockCode": pd.Series(dtype="string"),
                            "Quantity": pd.Series(dtype="int64"), "InvoiceDate": pd.Series(dtype="datetime64[ns]"),
                            "UnitPrice": pd.Series(dtype="float64"), "EmparejaCompra": pd.Series(dtype="bool")})
    x = calcular_variables_cliente(v, sin_dev, fecha_referencia(v))
    c1 = x.loc[1]
    assert c1["ticket_promedio"] == pytest.approx(50 / 3)
    assert c1["productos_distintos"] == 3
    assert c1["antiguedad"] == 31                       # del 01-01 al 01-02
    assert c1["dias_entre_compras"] == pytest.approx(30 / 2)   # 30 días en 2 intervalos
    assert c1["unidades_por_factura"] == pytest.approx(17 / 3)
    assert c1["pais"] == "France"                       # 2 de sus 3 facturas
    assert np.isnan(x.loc[2, "dias_entre_compras"])     # una sola factura: no definida
    assert x["tasa_devoluciones"].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# VIF, preprocesador y baseline RFM
# ---------------------------------------------------------------------------
from segmentacion.features import (  # noqa: E402
    calcular_vif,
    construir_preprocesador,
)


def test_vif_detecta_colinealidad():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    datos = pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(scale=0.01, size=500)})
    vif = calcular_vif(datos)
    assert vif["c"] > 100                          # c es casi a + b
    assert calcular_vif(datos[["a", "b"]]).max() < 1.1   # independientes


def test_preprocesador_log_winsor_y_escalado():
    datos = pd.DataFrame({"x": np.r_[np.arange(1, 100), 1_000_000.0], "y": np.arange(1, 101, dtype=float)})
    pre = construir_preprocesador(datos)
    z = pre.transform(datos)
    assert z.mean(axis=0) == pytest.approx([0, 0], abs=1e-9)
    assert z.std(axis=0) == pytest.approx([1, 1])
    # El valor extremo queda recortado al percentil 99 (en log), no a log1p(1e6).
    limite_alto = pre.named_steps["winsor"].kw_args["a_max"][0]
    assert limite_alto < np.log1p(1_000_000.0)
    sin_winsor = construir_preprocesador(datos, winsorizar=False)
    assert "winsor" not in sin_winsor.named_steps


def test_preprocesador_solo_usa_objetos_de_sklearn_y_numpy():
    import pickle
    pre = construir_preprocesador(pd.DataFrame({"x": np.arange(1.0, 51.0)}))
    clases = {type(paso).__module__.split(".")[0] for _, paso in pre.steps}
    assert clases == {"sklearn"}
    assert "segmentacion" not in pickle.dumps(pre).decode("latin-1")
