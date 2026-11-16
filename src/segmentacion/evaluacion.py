"""Métricas y diagnósticos de clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

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


def metricas_internas(datos: np.ndarray, etiquetas: np.ndarray, ruido: int = -1) -> dict[str, float]:
    """Silhouette, Davies-Bouldin y Calinski-Harabasz excluyendo los puntos de ruido.

    Para DBSCAN/HDBSCAN el ruido (−1) no es un cluster: se excluye del cálculo y se informa
    su porcentaje. Si quedan menos de 2 clusters las métricas no están definidas (NaN).

    Returns:
        ``n_clusters``, ``pct_ruido``, ``silhouette``, ``davies_bouldin``, ``calinski_harabasz``.
    """
    mascara = etiquetas != ruido
    validas = etiquetas[mascara]
    n_clusters = len(np.unique(validas))
    resultado = {"n_clusters": n_clusters, "pct_ruido": 100 * float((~mascara).mean()),
                 "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if n_clusters >= 2:
        sub = datos[mascara]
        resultado.update({
            "silhouette": float(silhouette_score(sub, validas)),
            "davies_bouldin": float(davies_bouldin_score(sub, validas)),
            "calinski_harabasz": float(calinski_harabasz_score(sub, validas)),
        })
    return resultado


def estabilidad_submuestreo(
    datos: np.ndarray,
    ajustar,
    n_repeticiones: int = 50,
    fraccion: float = 0.8,
    semilla: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Estabilidad por remuestreo: ARI entre la partición completa y la de submuestras.

    En cada repetición se toma una submuestra del ``fraccion`` de los clientes **sin
    reemplazo**, se reajusta el algoritmo y se compara (ARI) la etiqueta de esos clientes
    con la que tenían en la partición de referencia (ajustada con todos los datos). Se usa
    submuestreo y no bootstrap con reemplazo porque los duplicados que este introduce
    alteran la densidad local y sesgarían a DBSCAN/HDBSCAN; así todos los algoritmos se
    evalúan con el mismo procedimiento.

    Args:
        datos: matriz de entrada del algoritmo.
        ajustar: función ``ajustar(matriz, semilla) -> etiquetas``.
        n_repeticiones: número de submuestras.
        fraccion: proporción de clientes en cada submuestra.

    Returns:
        Vector con un ARI por repetición.
    """
    rng = np.random.default_rng(semilla)
    referencia = ajustar(datos, semilla)
    n = len(datos)
    valores = []
    for _ in range(n_repeticiones):
        idx = rng.choice(n, int(fraccion * n), replace=False)
        valores.append(adjusted_rand_score(referencia[idx], ajustar(datos[idx], int(rng.integers(1_000_000_000)))))
    return np.asarray(valores)


def matriz_ari(particiones: dict[str, np.ndarray]) -> pd.DataFrame:
    """ARI entre cada par de particiones (simétrica, con 1 en la diagonal)."""
    nombres = list(particiones)
    matriz = pd.DataFrame(np.eye(len(nombres)), index=nombres, columns=nombres)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            matriz.loc[a, b] = matriz.loc[b, a] = adjusted_rand_score(particiones[a], particiones[b])
    return matriz
