"""Clasificación de un cliente nuevo con los artefactos guardados (sin reentrenar).

Aplica exactamente la misma cadena que en el entrenamiento:

1. ``preprocesador.joblib``: log1p → recorte p1–p99 (límites fijos del entrenamiento) → StandardScaler.
2. ``kmeans.joblib``: segmento.
3. ``pca.joblib``: coordenadas para situarlo en el mapa PCA.
4. Marca de **cuenta clave** con la misma regla que en el notebook 04: segmento VIP, atípico para
   DBSCAN y gasto (= frequency × ticket_promedio) ≥ umbral. El atípico se decide con
   ``preprocesador_sin_winsor.joblib`` (DBSCAN se ajustó sin recorte) y los puntos núcleo de DBSCAN:
   un cliente sin ningún núcleo a distancia ≤ eps es ruido (``modelado.es_atipico_dbscan``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from segmentacion import config
from segmentacion.features import VARIABLES_CLUSTERING
from segmentacion.modelado import es_atipico_dbscan

SEGMENTO_VIP: int = 3
ETIQUETAS_VARIABLES: dict[str, str] = {
    "recency": "Días desde la última compra",
    "frequency": "Número de facturas",
    "ticket_promedio": "Ticket promedio (£)",
    "productos_distintos": "Productos distintos",
}


@dataclass
class Artefactos:
    """Modelos y parámetros necesarios para clasificar (todos generados por los notebooks)."""

    preprocesador: object
    preprocesador_sin_winsor: object
    kmeans: object
    pca: object
    vecinos_nucleo: object
    eps: float
    umbral_gasto: float

    @property
    def limites_log(self) -> tuple[np.ndarray, np.ndarray]:
        """Límites del recorte p1–p99 en escala log1p (los del entrenamiento)."""
        winsor = self.preprocesador.named_steps["winsor"].kw_args
        return np.asarray(winsor["a_min"]), np.asarray(winsor["a_max"])


def cargar_artefactos(directorio: Path = config.MODELS) -> Artefactos:
    """Carga los ``.joblib`` y los parámetros de DBSCAN y del umbral de cuentas clave."""
    metricas = json.loads((directorio / config.JSON_METRICAS.name).read_text(encoding="utf-8"))
    segmentos = json.loads((directorio / config.JSON_SEGMENTOS.name).read_text(encoding="utf-8"))
    return Artefactos(
        preprocesador=joblib.load(directorio / config.PREPROCESADOR.name),
        preprocesador_sin_winsor=joblib.load(directorio / config.PREPROCESADOR_SIN_WINSOR.name),
        kmeans=joblib.load(directorio / config.KMEANS.name),
        pca=joblib.load(directorio / config.PCA.name),
        vecinos_nucleo=joblib.load(directorio / config.DBSCAN_NUCLEOS.name),
        eps=float(metricas["dbscan"]["eps"]),
        umbral_gasto=float(segmentos["cuentas_clave"]["umbral_gasto"]),
    )


def validar_entrada(recency, frequency, ticket_promedio, productos_distintos) -> dict[str, float]:
    """Comprueba que los valores son válidos y los devuelve como números.

    Reglas: ``recency``, ``frequency`` y ``productos_distintos`` deben ser enteros ≥ 1;
    ``ticket_promedio`` un número finito > 0. No hay límite superior: los valores fuera del rango del
    entrenamiento se aceptan y el recorte p1–p99 del preprocesador los acota (igual que en el
    entrenamiento).

    Raises:
        ValueError: con la lista de problemas encontrados (en español).
    """
    valores = {"recency": recency, "frequency": frequency,
               "ticket_promedio": ticket_promedio, "productos_distintos": productos_distintos}
    errores = []
    limpios = {}
    for nombre, valor in valores.items():
        etiqueta = ETIQUETAS_VARIABLES[nombre]
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            errores.append(f"«{etiqueta}» debe ser un número (se recibió {valor!r}).")
            continue
        if not math.isfinite(numero):
            errores.append(f"«{etiqueta}» debe ser un número finito.")
            continue
        if nombre == "ticket_promedio":
            if numero <= 0:
                errores.append(f"«{etiqueta}» debe ser mayor que 0.")
                continue
        else:
            if numero != int(numero):
                errores.append(f"«{etiqueta}» debe ser un número entero.")
                continue
            if numero < 1:
                errores.append(f"«{etiqueta}» debe ser al menos 1.")
                continue
        limpios[nombre] = numero
    if errores:
        raise ValueError(" ".join(errores))
    return limpios


def clasificar_cliente(
    artefactos: Artefactos, recency, frequency, ticket_promedio, productos_distintos
) -> dict:
    """Clasifica un cliente con los modelos guardados.

    Returns:
        Diccionario con ``segmento``, ``monetary`` (frequency × ticket), ``z`` (valores
        estandarizados), ``pca`` (3 coordenadas), ``recortadas`` (variables que el recorte p1–p99
        modificó y hacia qué límite), ``atipico_dbscan``, ``distancia_nucleo`` y ``cuenta_clave``
        con ``motivo_cuenta_clave``.

    Raises:
        ValueError: si la entrada no es válida (ver :func:`validar_entrada`).
    """
    limpios = validar_entrada(recency, frequency, ticket_promedio, productos_distintos)
    fila = pd.DataFrame([limpios])[VARIABLES_CLUSTERING]

    z = artefactos.preprocesador.transform(fila)
    segmento = int(artefactos.kmeans.predict(z)[0])
    coordenadas = artefactos.pca.transform(z)[0]

    en_log = np.log1p(fila.to_numpy(dtype=float))[0]
    bajo, alto = artefactos.limites_log
    recortadas = {}
    for j, var in enumerate(VARIABLES_CLUSTERING):
        if en_log[j] > alto[j]:
            recortadas[var] = {"limite": "p99", "valor_usado": float(np.expm1(alto[j]))}
        elif en_log[j] < bajo[j]:
            recortadas[var] = {"limite": "p1", "valor_usado": float(np.expm1(bajo[j]))}

    z_sin_winsor = artefactos.preprocesador_sin_winsor.transform(fila)
    distancia = float(artefactos.vecinos_nucleo.kneighbors(z_sin_winsor, n_neighbors=1)[0][0, 0])
    atipico = bool(es_atipico_dbscan(artefactos.vecinos_nucleo, artefactos.eps, z_sin_winsor)[0])
    monetary = limpios["frequency"] * limpios["ticket_promedio"]

    es_vip = segmento == SEGMENTO_VIP
    supera_umbral = monetary >= artefactos.umbral_gasto
    cuenta_clave = es_vip and atipico and supera_umbral
    if cuenta_clave:
        motivo = "Es VIP, DBSCAN lo considera atípico y su gasto supera el umbral."
    else:
        faltan = [texto for cumple, texto in [(es_vip, "no es VIP"), (atipico, "DBSCAN no lo considera atípico"),
                                               (supera_umbral, "su gasto no llega al umbral")] if not cumple]
        motivo = "No es cuenta clave: " + ", ".join(faltan) + "."

    return {
        "segmento": segmento,
        "valores": limpios,
        "monetary": monetary,
        "z": dict(zip(VARIABLES_CLUSTERING, z[0].tolist())),
        "pca": coordenadas.tolist(),
        "recortadas": recortadas,
        "atipico_dbscan": atipico,
        "distancia_nucleo": distancia,
        "cuenta_clave": cuenta_clave,
        "motivo_cuenta_clave": motivo,
    }
