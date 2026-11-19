"""Pruebas del renderizado de documentos: formato, numeración y cifras obligatorias."""

from __future__ import annotations

import pytest

from segmentacion.reportes import libras, num, numerar_figuras_y_tablas, pct, reglas_arbol_en_texto, renderizar


def test_formato_espanol():
    assert num(1234567.891, 2) == "1.234.567,89"
    assert pct(24.0516) == "24,1 %"
    assert libras(3098.15, 2) == "3.098,15 £"


def test_una_cifra_inexistente_hace_fallar_el_renderizado():
    with pytest.raises(KeyError, match="cifra_que_no_existe"):
        renderizar("El valor es {cifra_que_no_existe}.", {"otra": "1"})


def test_renderizar_sustituye_cifras():
    assert renderizar("Hay {clientes} clientes.", {"clientes": "4.324"}).strip() == "Hay 4.324 clientes."


def test_numeracion_de_figuras_y_tablas():
    texto = "Intro\n![Primera](a.png)\nTabla: Resumen\n| a |\n|---|\n![Segunda](b.png)\n"
    salida = numerar_figuras_y_tablas(texto)
    assert "![Figura 1. Primera](a.png)" in salida
    assert "*Figura 1. Primera*" in salida
    assert "*Tabla 1. Resumen*" in salida
    assert "*Figura 2. Segunda*" in salida


def test_reglas_del_arbol_en_texto():
    from sklearn.tree import DecisionTreeClassifier, export_text

    arbol = DecisionTreeClassifier(max_depth=2, random_state=0).fit(
        [[1, 10], [2, 20], [8, 10], [9, 40]], ["A", "A", "B", "C"])
    texto = export_text(arbol, feature_names=["Facturas", "Ticket"], class_names=["A", "B", "C"], decimals=1)
    assert reglas_arbol_en_texto(texto) == ["Facturas ≤ 5,0 → **A**", "Facturas > 5,0 y Facturas ≤ 8,5 → **B**",
                                            "Facturas > 5,0 y Facturas > 8,5 → **C**"]
