"""Pruebas de modelado y evaluación con datos sintéticos de resultado conocido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from segmentacion.evaluacion import estabilidad_submuestreo, matriz_ari, metricas_internas
from segmentacion.modelado import (
    ajustar_gmm,
    ajustar_kmeans,
    barrido_kmeans,
    descuantizar,
    diagnostico_degeneracion,
    distancias_k_vecinos,
    indice_codo,
    ordenar_clusters_kmeans,
    resumen_densidad,
)


@pytest.fixture(scope="module")
def blobs():
    datos, verdad = make_blobs(n_samples=300, centers=[[0, 0], [10, 0], [0, 10]], cluster_std=0.5, random_state=0)
    return datos, verdad


def test_barrido_kmeans_columnas_y_mejor_k(blobs):
    datos, _ = blobs
    res = barrido_kmeans(datos, [2, 3, 4])
    assert list(res["k"]) == [2, 3, 4]
    assert {"inercia", "silhouette", "davies_bouldin", "calinski_harabasz", "pct_cluster_menor"} <= set(res.columns)
    assert res.loc[res["silhouette"].idxmax(), "k"] == 3       # tres grupos bien separados


def test_ordenar_clusters_mantiene_particion(blobs):
    datos, _ = blobs
    modelo = ajustar_kmeans(datos, 3)
    criterio = modelo.cluster_centers_[:, 0]                    # ordenar por la coordenada x del centro
    ordenado = ordenar_clusters_kmeans(modelo, datos, criterio)
    assert matriz_ari({"a": modelo.labels_, "b": ordenado.labels_}).loc["a", "b"] == pytest.approx(1.0)
    assert np.all(np.diff(ordenado.cluster_centers_[:, 0]) >= 0)  # etiqueta 0 = menor x


def test_distancias_k_vecinos_incluye_al_propio_punto():
    datos = np.array([[0.0], [1.0], [3.0], [6.0]])
    # min_samples=2: vecino más cercano distinto de sí mismo.
    assert distancias_k_vecinos(datos, 2).tolist() == [1.0, 1.0, 2.0, 3.0]


def test_indice_codo_en_curva_en_forma_de_l():
    curva = np.r_[np.linspace(0, 1, 90), np.linspace(1.5, 10, 10)]
    assert 88 <= indice_codo(curva) <= 91


def test_metricas_internas_excluyen_ruido(blobs):
    datos, verdad = blobs
    etiquetas = verdad.copy()
    etiquetas[:30] = -1
    m = metricas_internas(datos, etiquetas)
    assert m["n_clusters"] == 3
    assert m["pct_ruido"] == pytest.approx(10.0)
    assert m["silhouette"] > 0.8
    un_cluster = metricas_internas(datos, np.where(verdad == 0, -1, 0))
    assert np.isnan(un_cluster["silhouette"])                   # un solo cluster: no definido


def test_estabilidad_alta_con_grupos_separados(blobs):
    datos, _ = blobs
    aris = estabilidad_submuestreo(datos, lambda x, s: ajustar_kmeans(x, 3, s).labels_, n_repeticiones=5)
    assert len(aris) == 5
    assert aris.min() > 0.99


def test_matriz_ari_simetrica():
    a = np.array([0, 0, 1, 1])
    m = matriz_ari({"a": a, "b": 1 - a, "c": np.array([0, 1, 0, 1])})
    assert np.allclose(m.values, m.values.T)
    assert np.allclose(np.diag(m.values), 1)
    assert m.loc["a", "b"] == pytest.approx(1.0)                # mismas particiones con otras etiquetas


def datos_con_variable_discreta():
    """Columna 0 continua; columna 1 discreta con un 40 % de filas en el mismo valor."""
    rng = np.random.default_rng(0)
    n = 500
    niveles = np.where(rng.random(n) < 0.4, 1, rng.integers(2, 6, n))
    datos = np.column_stack([rng.normal(size=n), np.log1p(niveles)])
    return (datos - datos.mean(0)) / datos.std(0), niveles


def test_gmm_degenera_con_reg_covar_minimo_y_no_con_regularizacion():
    datos, niveles = datos_con_variable_discreta()
    degenerado = diagnostico_degeneracion(ajustar_gmm(datos, 4, "full", 1e-6), datos, niveles, 1)
    sano = diagnostico_degeneracion(ajustar_gmm(datos, 2, "full", 0.5), datos, niveles, 1)
    assert degenerado["degenerado"]
    assert not sano["degenerado"]


def test_resumen_densidad_detecta_cluster_de_un_solo_nivel():
    etiquetas = np.array([0, 0, 0, 1, 1, -1])
    niveles = np.array([1, 1, 1, 2, 3, 1])
    r = resumen_densidad(etiquetas, niveles)
    assert r["n_clusters"] == 2
    assert r["pct_ruido"] == pytest.approx(100 / 6)
    assert r["tamanos"] == [3, 2]
    assert r["pct_nivel_minimo"] == [100.0, 0.0]
    assert r["niveles_distintos"] == [1, 2]


def test_descuantizar_rompe_empates_sin_cambiar_el_redondeo():
    original = pd.Series([1, 1, 1, 2, 5])
    ruido = descuantizar(original)
    assert ruido.nunique() == 5
    assert (ruido.round() == original).all()
    assert (descuantizar(original) == ruido).all()              # reproducible con la semilla
