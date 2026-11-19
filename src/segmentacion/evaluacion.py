"""Métricas y diagnósticos de clustering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
    ajustar: Callable[[np.ndarray, int], np.ndarray],
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


# ---------------------------------------------------------------------------
# Exactitud: pruebas con datos de entrenamiento y de prueba
# ---------------------------------------------------------------------------
# En clustering no hay una etiqueta "verdadera" con la que medir la exactitud. Se usan dos
# pruebas que sí la tienen: (1) reasignar clientes que el modelo no vio al entrenar y comparar
# con el segmento del modelo final; (2) entrenar clasificadores supervisados que aprendan los
# segmentos y medir su exactitud en un conjunto de prueba separado.


def alinear_etiquetas(referencia: np.ndarray, predichas: np.ndarray, n_clusters: int) -> np.ndarray:
    """Renumera ``predichas`` para que coincidan al máximo con ``referencia``.

    Los números de cluster son arbitrarios (el cluster 2 de un ajuste puede ser el 0 de otro).
    Se busca la correspondencia uno a uno que maximiza las coincidencias con el algoritmo
    húngaro (``scipy.optimize.linear_sum_assignment``) sobre la matriz de confusión.

    Args:
        referencia: etiquetas de referencia (0..n_clusters-1).
        predichas: etiquetas a renumerar (0..n_clusters-1).
        n_clusters: número de clusters.

    Returns:
        ``predichas`` con la numeración de ``referencia``.
    """
    matriz = confusion_matrix(referencia, predichas, labels=list(range(n_clusters)))
    filas, columnas = linear_sum_assignment(-matriz)
    traduccion = np.empty(n_clusters, dtype=int)
    traduccion[columnas] = filas
    return traduccion[np.asarray(predichas)]


def exactitud_reasignacion(
    variables: pd.DataFrame,
    etiquetas: np.ndarray,
    entrenar: Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]],
    n_pliegues: int = 5,
    semilla: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Exactitud del modelo al asignar segmento a clientes que no vio al entrenar.

    Validación cruzada estratificada: en cada pliegue se entrena todo el proceso (preprocesado
    y clustering) solo con el resto de clientes, se asigna segmento a los clientes del pliegue
    y se compara, tras alinear la numeración, con su segmento en el modelo final. Es la misma
    operación que hace la app con un cliente nuevo.

    Args:
        variables: variables del modelo en escala original (una fila por cliente).
        etiquetas: segmento de cada cliente en el modelo final.
        entrenar: ``entrenar(datos_entrenamiento) -> predecir``, donde ``predecir(datos)``
            devuelve el segmento de cada fila.
        n_pliegues: número de pliegues (cada cliente está una vez en el conjunto de prueba).

    Returns:
        Una fila por pliegue con ``pliegue``, ``n_entrenamiento``, ``n_prueba`` y ``exactitud``.
    """
    etiquetas = np.asarray(etiquetas)
    n_clusters = len(np.unique(etiquetas))
    pliegues = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    filas = []
    for numero, (entrenamiento, prueba) in enumerate(pliegues.split(variables, etiquetas), start=1):
        predecir = entrenar(variables.iloc[entrenamiento])
        predichas = alinear_etiquetas(etiquetas[prueba], predecir(variables.iloc[prueba]), n_clusters)
        filas.append({"pliegue": numero, "n_entrenamiento": len(entrenamiento), "n_prueba": len(prueba),
                      "exactitud": float(accuracy_score(etiquetas[prueba], predichas))})
    return pd.DataFrame(filas)


def comparar_clasificadores(
    modelos: dict[str, ClassifierMixin],
    X_entrenamiento: pd.DataFrame,
    y_entrenamiento: np.ndarray,
    X_prueba: pd.DataFrame,
    y_prueba: np.ndarray,
    n_pliegues: int = 5,
    semilla: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Exactitud de varios clasificadores en validación cruzada y en el conjunto de prueba.

    La validación cruzada se hace solo con los datos de entrenamiento; el conjunto de prueba
    se usa una única vez, al final, con cada modelo reentrenado con todo el entrenamiento.

    Returns:
        Una fila por modelo con ``modelo``, ``exactitud_cv`` (media), ``exactitud_cv_std`` y
        ``exactitud_prueba``.
    """
    pliegues = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=semilla)
    filas = []
    for nombre, modelo in modelos.items():
        cv = cross_val_score(clone(modelo), X_entrenamiento, y_entrenamiento, cv=pliegues, scoring="accuracy")
        ajustado = clone(modelo).fit(X_entrenamiento, y_entrenamiento)
        filas.append({"modelo": nombre, "exactitud_cv": float(cv.mean()), "exactitud_cv_std": float(cv.std()),
                      "exactitud_prueba": float(accuracy_score(y_prueba, ajustado.predict(X_prueba)))})
    return pd.DataFrame(filas)
