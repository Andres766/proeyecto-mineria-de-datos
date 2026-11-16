"""Pruebas de la app Streamlit con AppTest: cada página carga sin excepciones y los casos de error se manejan."""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = sorted((RAIZ / "app" / "pages").glob("*.py"))


def ejecutar(ruta: Path) -> "AppTest":
    return AppTest.from_file(str(ruta), default_timeout=120).run()


@pytest.mark.parametrize("pagina", [RAIZ / "app" / "Inicio.py", *PAGINAS], ids=lambda p: p.stem)
def test_pagina_carga_sin_errores(pagina):
    at = ejecutar(pagina)
    assert not at.exception, at.exception


def test_inicio_muestra_hallazgos_clave():
    at = ejecutar(RAIZ / "app" / "Inicio.py")
    texto = " ".join(m.value for m in at.markdown)
    assert "RFM no distingue a los ocasionales" in texto
    assert "Cuentas clave" in texto
    assert "Por qué k = 4 y no k = 2" in texto


def pagina(nombre: str) -> Path:
    return next(p for p in PAGINAS if nombre in p.stem)


@pytest.mark.parametrize("algoritmo", [0, 1, 2])
@pytest.mark.parametrize("vista", ["PCA 2D", "PCA 3D", "t-SNE"])
def test_clusters_todas_las_combinaciones(algoritmo, vista):
    at = ejecutar(pagina("Clusters"))
    at.selectbox[0].set_value(at.selectbox[0].options[algoritmo]).run()
    at.radio[0].set_value(vista).run()
    assert not at.exception
    if vista == "t-SNE":
        assert any("no para validar" in w.value for w in at.warning)


@pytest.mark.parametrize("entrada, mensaje", [("abc", "número entero"), ("99999999", "No existe ningún cliente")])
def test_busqueda_de_id_invalido_muestra_error(entrada, mensaje):
    at = ejecutar(pagina("Explorador"))
    at.text_input[0].input(entrada).run()
    assert not at.exception
    assert any(mensaje in e.value for e in at.error)


def test_busqueda_de_id_valido():
    at = ejecutar(pagina("Explorador"))
    at.text_input[0].input("12347").run()
    assert not at.exception
    assert not at.error


def test_filtros_sin_resultados_avisan():
    at = ejecutar(pagina("Explorador"))
    at.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("Ningún cliente" in w.value for w in at.warning)


def clasificar(valores: list[float]):
    at = ejecutar(pagina("Explorador"))
    for widget, valor in zip(at.number_input, valores):
        widget.set_value(valor)
    next(b for b in at.button if "Clasificar" in b.label).click().run()
    return at


def test_formulario_cliente_tipico():
    at = clasificar([51, 2, 286.86, 35])
    assert not at.exception
    assert any("Segmento asignado" in m.value for m in at.markdown)
    assert not any("acotó" in w.value for w in at.warning)


def test_formulario_valores_extremos_se_acotan():
    at = clasificar([5000, 100000, 1e8, 20000])
    assert not at.exception
    assert any("Segmento asignado" in m.value for m in at.markdown)
    assert any("acotó" in w.value for w in at.warning)


def test_formulario_cuenta_clave():
    # Perfil típico de cuenta clave: muy reciente, muchas facturas, ticket alto.
    at = clasificar([5, 60, 1500.0, 150])
    assert not at.exception
    assert any("Es VIP, DBSCAN lo considera atípico" in c.value for c in at.caption)
