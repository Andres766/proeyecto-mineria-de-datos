"""Pruebas de la clasificación de clientes nuevos con los artefactos reales.

Confirman que el formulario de la app aplica exactamente el preprocesamiento del entrenamiento
(incluido el recorte p1–p99) y la misma regla de cuenta clave, y que las entradas extremas o
inválidas se manejan sin errores inesperados.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from segmentacion import config
from segmentacion.features import VARIABLES_CLUSTERING
from segmentacion.modelado import es_atipico_dbscan
from segmentacion.prediccion import cargar_artefactos, clasificar_cliente, validar_entrada

pytestmark = pytest.mark.skipif(not config.DBSCAN_NUCLEOS.exists(), reason="faltan los artefactos de models/")


@pytest.fixture(scope="module")
def artefactos():
    return cargar_artefactos()


@pytest.fixture(scope="module")
def clientes():
    return pd.read_parquet(config.PARQUET_SEGMENTADOS)


def test_pipeline_reproduce_todos_los_segmentos(artefactos, clientes):
    z = artefactos.preprocesador.transform(clientes[VARIABLES_CLUSTERING])
    assert np.array_equal(artefactos.kmeans.predict(z), clientes["segmento"].to_numpy())


def test_regla_completa_reproduce_cuentas_clave(artefactos, clientes):
    z = artefactos.preprocesador.transform(clientes[VARIABLES_CLUSTERING])
    zn = artefactos.preprocesador_sin_winsor.transform(clientes[VARIABLES_CLUSTERING])
    atipico = es_atipico_dbscan(artefactos.vecinos_nucleo, artefactos.eps, zn)
    assert np.array_equal(atipico, clientes["dbscan"].eq(-1).to_numpy())
    monetary = clientes["frequency"] * clientes["ticket_promedio"]
    assert np.allclose(monetary, clientes["monetary"])
    regla = (artefactos.kmeans.predict(z) == 3) & atipico & (monetary.to_numpy() >= artefactos.umbral_gasto)
    assert np.array_equal(regla, clientes["cuenta_clave"].to_numpy())


def test_formulario_coincide_con_el_entrenamiento(artefactos, clientes):
    muestra = pd.concat([clientes[clientes["cuenta_clave"]], clientes[~clientes["cuenta_clave"]].sample(150, random_state=42)])
    for _, c in muestra.iterrows():
        r = clasificar_cliente(artefactos, c["recency"], c["frequency"], c["ticket_promedio"], c["productos_distintos"])
        assert r["segmento"] == c["segmento"]
        assert r["cuenta_clave"] == c["cuenta_clave"]
        assert r["pca"] == pytest.approx([c["pc1"], c["pc2"], c["pc3"]], abs=1e-9)


def test_valores_extremos_se_recortan_sin_romper(artefactos):
    r = clasificar_cliente(artefactos, recency=10_000, frequency=1_000_000, ticket_promedio=1e9, productos_distintos=50_000)
    assert r["segmento"] in {0, 1, 2, 3}
    assert set(r["recortadas"]) == set(VARIABLES_CLUSTERING)
    assert all(math.isfinite(v) for v in r["z"].values())
    _, alto = artefactos.limites_log
    # Tras recortar, el valor estandarizado es el del límite p99 del entrenamiento.
    limite_z = artefactos.preprocesador.named_steps["escalado"].transform(pd.DataFrame([alto], columns=VARIABLES_CLUSTERING))[0]
    assert list(r["z"].values()) == pytest.approx(limite_z.tolist())
    minimo = clasificar_cliente(artefactos, 1, 1, 0.01, 1)
    assert minimo["recortadas"]["ticket_promedio"]["limite"] == "p1"


def test_cliente_tipico_no_se_recorta(artefactos):
    r = clasificar_cliente(artefactos, 51, 2, 286.86, 35)
    assert r["recortadas"] == {}
    assert r["monetary"] == pytest.approx(2 * 286.86)


@pytest.mark.parametrize(
    "entrada, texto",
    [
        ((0, 2, 100.0, 5), "al menos 1"),
        ((10, -3, 100.0, 5), "al menos 1"),
        ((10, 2.5, 100.0, 5), "entero"),
        ((10, 2, 0, 5), "mayor que 0"),
        ((10, 2, -50.0, 5), "mayor que 0"),
        ((10, 2, float("nan"), 5), "finito"),
        ((10, 2, float("inf"), 5), "finito"),
        (("abc", 2, 100.0, 5), "número"),
        ((None, 2, 100.0, 5), "número"),
    ],
)
def test_entradas_invalidas_dan_mensaje_claro(entrada, texto):
    with pytest.raises(ValueError, match=texto):
        validar_entrada(*entrada)


def test_varios_errores_a_la_vez():
    with pytest.raises(ValueError) as error:
        validar_entrada(0, 0, -1, 0)
    assert str(error.value).count("«") == 4
