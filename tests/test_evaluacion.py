"""Pruebas de las medidas de exactitud (entrenamiento/prueba) con datos sintéticos de resultado conocido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from segmentacion.evaluacion import alinear_etiquetas, comparar_clasificadores, exactitud_reasignacion
from segmentacion.modelado import ajustar_kmeans


@pytest.fixture(scope="module")
def blobs():
    datos, verdad = make_blobs(n_samples=300, centers=[[0, 0], [10, 0], [0, 10]], cluster_std=0.5, random_state=0)
    return pd.DataFrame(datos, columns=["x", "y"]), verdad


def test_alinear_etiquetas_deshace_una_permutacion():
    referencia = np.array([0, 0, 1, 1, 2, 2])
    permutadas = np.array([2, 2, 0, 0, 1, 1])
    assert alinear_etiquetas(referencia, permutadas, 3).tolist() == referencia.tolist()


def test_alinear_etiquetas_conserva_los_errores():
    referencia = np.array([0, 0, 0, 1, 1, 1])
    predichas = np.array([1, 1, 0, 0, 0, 0])  # renumeradas: 0,0,1,1,1,1 -> un error
    alineadas = alinear_etiquetas(referencia, predichas, 2)
    assert (alineadas == referencia).sum() == 5


def test_exactitud_reasignacion_grupos_separados_es_total(blobs):
    datos, verdad = blobs

    def entrenar(entrenamiento: pd.DataFrame):
        modelo = ajustar_kmeans(entrenamiento.to_numpy(), 3)
        return lambda nuevos: modelo.predict(nuevos.to_numpy())

    resultado = exactitud_reasignacion(datos, verdad, entrenar, n_pliegues=5)
    assert len(resultado) == 5
    assert resultado["n_prueba"].sum() == len(datos)  # cada cliente se prueba una sola vez
    assert (resultado["exactitud"] == 1.0).all()


def test_exactitud_reasignacion_detecta_un_modelo_que_no_aprende(blobs):
    datos, verdad = blobs
    resultado = exactitud_reasignacion(datos, verdad, lambda _: (lambda nuevos: np.zeros(len(nuevos), dtype=int)))
    assert resultado["exactitud"].mean() == pytest.approx(1 / 3, abs=0.02)


def test_comparar_clasificadores(blobs):
    datos, verdad = blobs
    entrenamiento, prueba = datos.iloc[:200], datos.iloc[200:]
    tabla = comparar_clasificadores(
        {"árbol": DecisionTreeClassifier(random_state=0), "mayoritario": DummyClassifier(strategy="most_frequent")},
        entrenamiento, verdad[:200], prueba, verdad[200:])
    assert list(tabla.columns) == ["modelo", "exactitud_cv", "exactitud_cv_std", "exactitud_prueba"]
    fila = tabla.set_index("modelo")
    assert fila.loc["árbol", "exactitud_prueba"] == 1.0
    assert fila.loc["mayoritario", "exactitud_prueba"] < 0.5
