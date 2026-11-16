"""Utilidades de visualización compartidas por los notebooks (matplotlib/seaborn)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from segmentacion import config

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


# Color fijo por segmento (mismo en notebooks y app). Paleta comprobada con un validador de
# accesibilidad: pasa la separación para daltonismo (ΔE ≥ 9,2) y para visión normal (ΔE ≥ 16,3)
# en TODOS los pares, requisito de los gráficos de dispersión (PCA, t-SNE). El aqua queda por
# debajo de 3:1 de contraste con el fondo claro: siempre se acompaña de leyenda, etiqueta o tabla.
COLORES_SEGMENTO: dict[int, str] = {
    0: "#eb6834",  # naranja
    1: "#4a3aa7",  # violeta
    2: "#1baf7a",  # aqua
    3: "#2a78d6",  # azul
}
# Las cuentas clave no llevan un quinto color (rompería la validación): se marcan con forma.
MARCADOR_CUENTA_CLAVE: str = "D"  # rombo en matplotlib ("diamond" en Plotly)
COLOR_BORDE_CUENTA_CLAVE: str = "#0b0b0b"
