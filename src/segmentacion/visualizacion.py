"""Utilidades de visualización compartidas por los notebooks (matplotlib/seaborn)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from segmentacion import config
# Reexportados para los notebooks; viven en colores.py porque la app no puede depender de matplotlib.
from segmentacion.colores import COLOR_BORDE_CUENTA_CLAVE, COLORES_SEGMENTO, MARCADOR_CUENTA_CLAVE  # noqa: F401

COLOR_PRINCIPAL: str = "#2E5A88"
COLOR_SECUNDARIO: str = "#E07B39"
COLOR_NEUTRO: str = "#9AA5B1"

DIAS_SEMANA: dict[int, str] = {
    0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"
}


def estilo() -> None:
    """Aplica un estilo homogéneo a todas las figuras de los notebooks."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 4.5),
            "figure.dpi": 100,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Guarda la figura como PNG en reports/figures para reutilizarla en el informe.

    Args:
        fig: figura de matplotlib.
        nombre: nombre de archivo sin extensión (p. ej. ``"01_ventas_mensuales"``).
    """
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{nombre}.png", dpi=120, bbox_inches="tight")
