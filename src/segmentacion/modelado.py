"""Reducción de dimensionalidad y algoritmos de clustering (K-Means, GMM, DBSCAN, HDBSCAN, t-SNE)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from segmentacion import config

N_INIT_KMEANS: int = 10


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def ajustar_pca(datos: np.ndarray, n_componentes: int | None = None) -> PCA:
    """Ajusta PCA sobre datos ya estandarizados (equivale a PCA sobre la matriz de correlaciones)."""
    return PCA(n_components=n_componentes, random_state=config.RANDOM_STATE).fit(datos)


def tabla_varianza_pca(pca: PCA) -> pd.DataFrame:
    """Varianza explicada y acumulada por componente."""
    return pd.DataFrame(
        {
            "varianza_explicada": pca.explained_variance_ratio_,
            "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_),
            "autovalor": pca.explained_variance_,
        },
        index=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )


def cargas_pca(pca: PCA, variables: list[str]) -> pd.DataFrame:
    """Cargas (loadings): peso de cada variable original en cada componente."""
    return pd.DataFrame(pca.components_.T, index=variables, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def ajustar_kmeans(datos: np.ndarray, k: int, semilla: int = config.RANDOM_STATE) -> KMeans:
    """K-Means con ``N_INIT_KMEANS`` inicializaciones k-means++."""
    return KMeans(n_clusters=k, n_init=N_INIT_KMEANS, random_state=semilla).fit(datos)


def barrido_kmeans(datos: np.ndarray, valores_k: list[int]) -> pd.DataFrame:
    """Inercia (codo), silhouette, Davies-Bouldin y Calinski-Harabasz para cada k."""
    filas = []
    for k in valores_k:
        modelo = ajustar_kmeans(datos, k)
        lab = modelo.labels_
        filas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(datos, lab),
            "davies_bouldin": davies_bouldin_score(datos, lab),
            "calinski_harabasz": calinski_harabasz_score(datos, lab),
            "pct_cluster_menor": 100 * np.bincount(lab).min() / len(lab),
        })
    return pd.DataFrame(filas)


def ordenar_clusters_kmeans(modelo: KMeans, criterio: np.ndarray) -> KMeans:
    """Renumera los clusters de un K-Means de menor a mayor ``criterio`` sin reajustarlo.

    Devuelve una copia del modelo con los centroides permutados y las etiquetas
    renumeradas: la partición es exactamente la misma (ningún cliente cambia de grupo) y
    la etiqueta 0 corresponde al cluster con menor ``criterio``. Sigue siendo un ``KMeans``
    estándar de scikit-learn, así que ``predict`` funciona igual al cargarlo con joblib.

    Args:
        modelo: K-Means ajustado.
        criterio: un valor por cluster (índice = etiqueta original), p. ej. la mediana de monetary.
    """
    import copy

    orden = np.argsort(criterio)                 # orden[nueva] = original
    nueva_de_original = np.empty_like(orden)
    nueva_de_original[orden] = np.arange(len(orden))
    ordenado = copy.deepcopy(modelo)
    ordenado.cluster_centers_ = modelo.cluster_centers_[orden]
    ordenado.labels_ = nueva_de_original[modelo.labels_]
    return ordenado


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------
def ajustar_gmm(
    datos: np.ndarray, k: int, tipo_covarianza: str, reg_covar: float, semilla: int = config.RANDOM_STATE
) -> GaussianMixture:
    """Mezcla gaussiana con 3 inicializaciones y hasta 500 iteraciones EM."""
    return GaussianMixture(
        n_components=k, covariance_type=tipo_covarianza, reg_covar=reg_covar,
        n_init=3, max_iter=500, random_state=semilla,
    ).fit(datos)


def varianza_componentes(gmm: GaussianMixture, indice_variable: int) -> np.ndarray:
    """Varianza de cada componente a lo largo de una variable (columna ``indice_variable``)."""
    cov = gmm.covariances_
    if gmm.covariance_type == "full":
        return cov[:, indice_variable, indice_variable]
    if gmm.covariance_type == "diag":
        return cov[:, indice_variable]
    if gmm.covariance_type == "spherical":
        return cov
    return np.repeat(cov[indice_variable, indice_variable], gmm.n_components)  # tied


VARIANZA_EMPIRICA_MINIMA: float = 0.01


def diagnostico_degeneracion(
    gmm: GaussianMixture,
    datos: np.ndarray,
    niveles: np.ndarray,
    indice_variable: int,
    tamano_minimo: float = 0.02,
) -> dict[str, float | bool]:
    """Detecta componentes que colapsan sobre un único valor de una variable discreta.

    EM estima la covarianza de cada componente y scikit-learn le suma ``reg_covar`` en la
    diagonal. La varianza **empírica** de un componente en la variable discreta es, por
    tanto, ``varianza - reg_covar``. Si es prácticamente nula, el componente está sentado
    sobre un único nivel (p. ej. todos los clientes con frequency = 1): la gaussiana modela
    la discreción del dato y no un grupo de clientes, y su densidad (y con ella la
    verosimilitud y el BIC) se infla sin límite salvo por ``reg_covar``.

    Criterio: degenerado si algún componente tiene varianza empírica menor que
    ``VARIANZA_EMPIRICA_MINIMA`` = 0,01 en unidades estandarizadas, es decir, una dispersión
    inferior a 0,1 desviaciones típicas de la variable.

    Args:
        gmm: modelo ajustado.
        datos: datos (estandarizados) con los que se ajustó.
        niveles: valor original de la variable discreta para cada fila (p. ej. frequency).
        indice_variable: columna de esa variable en ``datos``.
        tamano_minimo: tamaño mínimo (proporción) para describir la composición de un componente.

    Returns:
        ``varianza_empirica_min`` (mínimo entre componentes), ``max_un_nivel`` (descriptivo:
        mayor proporción de un único nivel entre los miembros de un componente de tamaño
        ≥ ``tamano_minimo``, por asignación dura) y ``degenerado``.
    """
    empirica = varianza_componentes(gmm, indice_variable) - gmm.reg_covar
    etiquetas = gmm.predict(datos)
    proporciones = []
    for comp in np.unique(etiquetas):
        miembros = etiquetas == comp
        if miembros.mean() >= tamano_minimo:
            proporciones.append(pd.Series(niveles[miembros]).value_counts(normalize=True).iloc[0])
    minimo = float(empirica.min())
    return {
        "varianza_empirica_min": minimo,
        "max_un_nivel": float(max(proporciones)) if proporciones else float("nan"),
        "degenerado": bool(minimo < VARIANZA_EMPIRICA_MINIMA),
    }


def barrido_gmm(
    datos: np.ndarray,
    valores_k: list[int],
    tipos_covarianza: list[str],
    reg_covar: float,
    niveles: np.ndarray | None = None,
    indice_variable: int | None = None,
) -> pd.DataFrame:
    """BIC y AIC por número de componentes y tipo de covarianza (y degeneración si se pide)."""
    filas = []
    for tipo in tipos_covarianza:
        for k in valores_k:
            gmm = ajustar_gmm(datos, k, tipo, reg_covar)
            fila = {"tipo": tipo, "k": k, "reg_covar": reg_covar, "bic": gmm.bic(datos), "aic": gmm.aic(datos),
                    "convergio": gmm.converged_}
            if niveles is not None and indice_variable is not None:
                fila.update(diagnostico_degeneracion(gmm, datos, niveles, indice_variable))
            filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# DBSCAN / HDBSCAN
# ---------------------------------------------------------------------------
def distancias_k_vecinos(datos: np.ndarray, min_samples: int) -> np.ndarray:
    """Distancia de cada punto a su vecino número ``min_samples`` contándose a sí mismo.

    Es exactamente la distancia que DBSCAN compara con ``eps`` para decidir si un punto es
    núcleo (``min_samples`` incluye al propio punto en scikit-learn).
    """
    distancias, _ = NearestNeighbors(n_neighbors=min_samples).fit(datos).kneighbors(datos)
    return distancias[:, -1]


def indice_codo(valores_ordenados: np.ndarray) -> int:
    """Codo de una curva creciente: punto de máxima distancia a la recta que une sus extremos."""
    y = np.asarray(valores_ordenados, dtype=float)
    x = np.arange(len(y), dtype=float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    distancia = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / np.hypot(dx, dy)
    return int(np.argmax(distancia))


def ajustar_dbscan(datos: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Etiquetas de DBSCAN (−1 = ruido)."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(datos)


def ajustar_hdbscan(datos: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """Etiquetas de HDBSCAN (−1 = ruido)."""
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(datos)


def resumen_densidad(etiquetas: np.ndarray, niveles: np.ndarray) -> dict[str, object]:
    """Resumen de una partición por densidad y de su relación con una variable discreta.

    Returns:
        ``n_clusters``, ``pct_ruido``, ``tamanos`` (desc.), ``pct_nivel_minimo`` (% de miembros
        con el nivel mínimo, p. ej. frequency = 1, en cada cluster) y ``niveles_distintos``
        (cuántos valores distintos de la variable hay en cada cluster).
    """
    clusters = [c for c in np.unique(etiquetas) if c != -1]
    clusters.sort(key=lambda c: -(etiquetas == c).sum())
    minimo = niveles.min()
    return {
        "n_clusters": len(clusters),
        "pct_ruido": 100 * float((etiquetas == -1).mean()),
        "tamanos": [int((etiquetas == c).sum()) for c in clusters],
        "pct_nivel_minimo": [round(100 * float((niveles[etiquetas == c] == minimo).mean()), 1) for c in clusters],
        "niveles_distintos": [int(len(np.unique(niveles[etiquetas == c]))) for c in clusters],
    }


def descuantizar(valores: pd.Series, semilla: int = config.RANDOM_STATE) -> pd.Series:
    """Suma ruido uniforme en (−0,5, 0,5) a una variable entera (diagnóstico de empates).

    Reparte cada valor entero por su intervalo de redondeo, de modo que los clientes que
    compartían exactamente el mismo valor dejan de coincidir. Solo se usa para comprobar si
    un resultado depende de los empates; los modelos finales usan los datos reales.
    """
    rng = np.random.default_rng(semilla)
    return valores.astype(float) + rng.uniform(-0.5, 0.5, len(valores))


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------
def calcular_tsne(datos: np.ndarray, perplexity: float) -> np.ndarray:
    """Proyección t-SNE 2D, SOLO para visualizar.

    t-SNE conserva vecindarios locales, no distancias globales: el tamaño de un grupo y la
    distancia entre grupos en el mapa no significan nada, por lo que no sirve para validar.
    """
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=config.RANDOM_STATE).fit_transform(datos)
