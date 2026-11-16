"""Pruebas del perfilamiento con un ejemplo pequeño de resultado conocido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segmentacion.catalogo import CUENTAS_CLAVE, SEGMENTOS
from segmentacion.perfilamiento import (
    facturas_por_mes,
    indicadores_segmento,
    medias_estandarizadas,
    pais_principal,
    percentiles_medios,
    productos_top,
)


@pytest.fixture
def ejemplo():
    """Cliente 1 (segmento 0): una compra en octubre-2011. Cliente 2 (segmento 1): tres compras
    (marzo, septiembre y noviembre de 2011). Cliente 3 (segmento 1): una compra en diciembre-2010."""
    filas = [
        ("F1", 1, "A", 2, "2011-10-05", 10.0, "France"),
        ("F2", 2, "A", 1, "2011-03-01", 10.0, "United Kingdom"),
        ("F2", 2, "B", 1, "2011-03-01", 30.0, "United Kingdom"),
        ("F3", 2, "B", 1, "2011-09-10", 30.0, "United Kingdom"),
        ("F4", 2, "C", 4, "2011-11-20", 6.0, "United Kingdom"),
        ("F5", 3, "A", 1, "2010-12-03", 10.0, "United Kingdom"),
    ]
    ventas = pd.DataFrame(filas, columns=["InvoiceNo", "CustomerID", "StockCode", "Quantity", "InvoiceDate",
                                          "UnitPrice", "Country"])
    ventas["InvoiceDate"] = pd.to_datetime(ventas["InvoiceDate"])
    ventas["Importe"] = ventas["Quantity"] * ventas["UnitPrice"]
    ventas["Description"] = ventas["StockCode"].map({"A": "ART A", "B": "ART B", "C": "ART C"})
    clientes = pd.DataFrame({
        "recency": [66, 20, 372], "frequency": [1, 3, 1], "monetary": [20.0, 90.0, 10.0],
        "ticket_promedio": [20.0, 30.0, 10.0], "productos_distintos": [1, 3, 1],
        "dias_entre_compras": [np.nan, 132.0, np.nan], "tasa_devoluciones": [0.0, 0.1, 0.0],
        "pais": ["France", "United Kingdom", "United Kingdom"],
    }, index=pd.Index([1, 2, 3], name="CustomerID"))
    segmento = pd.Series([0, 1, 1], index=clientes.index)
    return ventas, clientes, segmento


def test_indicadores_segmento(ejemplo):
    ventas, clientes, segmento = ejemplo
    ind = indicadores_segmento(clientes, ventas, segmento)
    assert ind.loc[0, "clientes"] == 1
    assert ind.loc[1, "pct_ingresos"] == pytest.approx(100 * 100 / 120)
    assert ind.loc[1, "pct_compra_90d"] == pytest.approx(50.0)          # cliente 2 sí, cliente 3 no
    assert ind.loc[1, "pct_sin_compra_90d"] == pytest.approx(50.0)
    assert ind.loc[1, "pct_una_compra"] == pytest.approx(50.0)
    assert ind.loc[1, "pct_clientes_compran_temporada"] == pytest.approx(50.0)  # solo el 2 (sep y nov)
    assert ind.loc[1, "pct_facturas_temporada"] == pytest.approx(100 * 2 / 4)   # F3 y F4 de F2..F5
    assert ind.loc[0, "pct_una_compra_en_temporada"] == pytest.approx(100.0)    # su única compra en octubre
    assert ind.loc[1, "pct_una_compra_en_temporada"] == pytest.approx(0.0)      # el 3 compró en dic-2010
    assert ind.loc[0, "pct_captados_temporada"] == pytest.approx(100.0)
    assert ind.loc[1, "ingresos_temporada"] == pytest.approx(30.0 + 24.0)
    assert np.isnan(ind.loc[0, "pct_intervalo_60d"])


def test_facturas_por_mes_y_filtro(ejemplo):
    ventas, _, segmento = ejemplo
    tabla = facturas_por_mes(ventas, segmento)
    assert tabla.loc[1].sum() == pytest.approx(100.0)
    assert tabla.loc[1, pd.Period("2011-03", "M")] == pytest.approx(25.0)
    assert tabla.loc["Total", pd.Period("2011-10", "M")] == pytest.approx(20.0)
    solo_una = facturas_por_mes(ventas, segmento, clientes_incluidos=pd.Index([1, 3]))
    assert solo_una.loc[1, pd.Period("2010-12", "M")] == pytest.approx(100.0)


def test_productos_top_por_importe_y_penetracion(ejemplo):
    ventas, _, segmento = ejemplo
    imp = productos_top(ventas, segmento, n=2, criterio="importe")
    seg1 = imp[imp["segmento"] == 1]
    assert seg1["StockCode"].tolist() == ["B", "C"]                    # 60 y 24 (A suma 20)
    pen = productos_top(ventas, segmento, n=1, criterio="penetracion")
    assert pen.loc[pen["segmento"] == 1, "valor"].iloc[0] == pytest.approx(100.0)  # A: clientes 2 y 3
    with pytest.raises(ValueError):
        productos_top(ventas, segmento, criterio="otro")


def test_pais_principal(ejemplo):
    _, clientes, segmento = ejemplo
    paises = pais_principal(clientes, segmento.to_numpy())
    assert paises.loc[0, "pais_principal"] == "France"
    assert paises.loc[1, "pct_pais_principal"] == pytest.approx(100.0)
    assert paises.loc[1, "segundo_pais"] is None


def test_snake_y_radar(ejemplo):
    _, clientes, segmento = ejemplo
    z = np.array([[1.0, -1.0], [3.0, 1.0], [-1.0, 0.0]])
    snake = medias_estandarizadas(z, segmento.to_numpy(), ["x", "y"])
    assert snake.loc[1].tolist() == pytest.approx([1.0, 0.5])
    # recency 66, 20, 372 -> percentiles 66,7 / 33,3 / 100; invertidos 33,3 / 66,7 / 0.
    radar = percentiles_medios(clientes, segmento.to_numpy(), ["recency"], invertir=("recency",))
    assert radar.loc[0, "recency"] == pytest.approx(100 / 3)
    assert radar.loc[1, "recency"] == pytest.approx((200 / 3 + 0) / 2)


def test_catalogo_completo():
    claves_validas = {"pct_compra_90d", "pct_ingresos", "pct_una_compra_en_temporada", "pct_una_compra",
                      "pct_intervalo_60d", "pct_captados_temporada", "ticket_mediano",
                      "pct_clientes_compran_temporada", "pct_sin_compra_90d", "pct_devuelve", "ingresos_temporada"}
    assert sorted(SEGMENTOS) == [0, 1, 2, 3]
    for grupo in [*SEGMENTOS.values(), CUENTAS_CLAVE]:
        assert grupo["nombre"] and grupo["descripcion"]
        assert 2 <= len(grupo["estrategias"]) <= 3
        for estrategia in grupo["estrategias"]:
            assert {"accion", "momento", "kpi", "indicador"} <= set(estrategia)
            assert estrategia["indicador"] in claves_validas
    nombres = [g["nombre"] for g in SEGMENTOS.values()]
    assert len(set(nombres)) == 4
