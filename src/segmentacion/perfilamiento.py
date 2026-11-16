"""Perfilamiento de segmentos de clientes."""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIABLES_RESUMEN: list[str] = ["recency", "frequency", "ticket_promedio", "productos_distintos", "monetary"]


def resumen_segmentos(clientes: pd.DataFrame, etiquetas: np.ndarray | pd.Series) -> pd.DataFrame:
    """Tamaño, peso en ingresos y medianas de las variables principales por segmento.

    Args:
        clientes: tabla de clientes con las columnas de ``VARIABLES_RESUMEN``.
        etiquetas: segmento de cada cliente (mismo orden que ``clientes``).

    Returns:
        DataFrame indexado por segmento con ``clientes``, ``pct_clientes``, ``pct_ingresos``,
        ``pct_una_compra`` (frequency = 1) y la mediana de cada variable, ordenado por la
        mediana de monetary.
    """
    datos = clientes.assign(segmento=np.asarray(etiquetas))
    grupos = datos.groupby("segmento")
    resumen = pd.DataFrame({
        "clientes": grupos.size(),
        "pct_clientes": 100 * grupos.size() / len(datos),
        "pct_ingresos": 100 * grupos["monetary"].sum() / datos["monetary"].sum(),
        "pct_una_compra": 100 * grupos["frequency"].apply(lambda s: (s == 1).mean()),
    })
    medianas = grupos[VARIABLES_RESUMEN].median().add_suffix("_mediana")
    return resumen.join(medianas).sort_values("monetary_mediana")
