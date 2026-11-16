"""Métricas y diagnósticos de clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from segmentacion import config


def diagnostico_kmeans(datos: np.ndarray, valores_k: list[int]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Ajusta K-Means para cada k y resume silhouette y tamaños de cluster.

    Se usa en la ingeniería de características para comparar alternativas de
    preprocesamiento con un clustering de prueba (no es el modelo final).

    Args:
        datos: matriz ya preprocesada (filas = clientes).
        valores_k: números de clusters a probar.

    Returns:
        Tupla ``(resumen, etiquetas)``: ``resumen`` con columnas ``k``, ``silhouette``,
        ``tamanos`` (ordenados de menor a mayor) y ``pct_cluster_menor``; ``etiquetas``
        mapea cada k a su vector de etiquetas.
    """
    filas, etiquetas = [], {}
    for k in valores_k:
        lab = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(datos)
        etiquetas[k] = lab
        tamanos = np.sort(np.bincount(lab))
        filas.append({
            "k": k,
            "silhouette": silhouette_score(datos, lab),
            "tamanos": tamanos.tolist(),
            "pct_cluster_menor": 100 * tamanos[0] / len(lab),
        })
    return pd.DataFrame(filas), etiquetas


def ari(etiquetas_a: np.ndarray, etiquetas_b: np.ndarray) -> float:
    """Índice de Rand ajustado entre dos particiones (1 = idénticas, ~0 = azar)."""
    return float(adjusted_rand_score(etiquetas_a, etiquetas_b))
