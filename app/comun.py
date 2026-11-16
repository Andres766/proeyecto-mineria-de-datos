"""Utilidades compartidas por las páginas de la app: carga con caché, colores, formato y estilo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from segmentacion import config  # noqa: E402
from segmentacion.catalogo import CUENTAS_CLAVE, SEGMENTOS  # noqa: E402
from segmentacion.prediccion import Artefactos, cargar_artefactos  # noqa: E402
from segmentacion.visualizacion import COLOR_BORDE_CUENTA_CLAVE, COLORES_SEGMENTO  # noqa: E402

# Colores de texto y superficie (los de serie nunca se usan para texto).
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
NEUTRO = "#c3c2b7"
NOMBRES = {s: SEGMENTOS[s]["nombre"] for s in sorted(SEGMENTOS)}
COLOR_POR_NOMBRE = {NOMBRES[s]: COLORES_SEGMENTO[s] for s in NOMBRES}
ORDEN_NOMBRES = [NOMBRES[s] for s in sorted(NOMBRES)]
NOMBRE_CUENTAS_CLAVE = CUENTAS_CLAVE["nombre"]
MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]

ETIQUETAS = {
    "recency": "Días desde la última compra",
    "frequency": "Facturas",
    "monetary": "Gasto total (£)",
    "ticket_promedio": "Ticket promedio (£)",
    "productos_distintos": "Productos distintos",
    "antiguedad": "Antigüedad (días)",
    "dias_entre_compras": "Días entre compras",
    "unidades_por_factura": "Unidades por factura",
    "tasa_devoluciones": "Tasa de devoluciones",
    "pais": "País",
}


# ---------------------------------------------------------------------------
# Carga de datos y modelos (con caché: se lee una sola vez por sesión del servidor)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando clientes…")
def cargar_clientes() -> pd.DataFrame:
    """Clientes con variables, segmento, etiquetas de los algoritmos y coordenadas."""
    clientes = pd.read_parquet(config.PARQUET_SEGMENTADOS)
    clientes["nombre_segmento"] = pd.Categorical(clientes["nombre_segmento"], categories=ORDEN_NOMBRES, ordered=True)
    return clientes


@st.cache_data(show_spinner="Cargando transacciones…")
def cargar_ventas() -> pd.DataFrame:
    """Transacciones limpias (para la página de exploración)."""
    return pd.read_parquet(config.PARQUET_LIMPIO)


@st.cache_data
def cargar_bitacora() -> pd.DataFrame:
    return pd.read_csv(config.CSV_BITACORA)


@st.cache_data
def cargar_json(nombre: str) -> dict:
    """Lee un JSON de ``models/`` (metricas, segmentos, features)."""
    return json.loads((config.MODELS / f"{nombre}.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Cargando modelos…")
def artefactos() -> Artefactos:
    """Preprocesadores y modelos ajustados (no se reentrena nada en la app)."""
    return cargar_artefactos()


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------
def num(valor: float, decimales: int = 0) -> str:
    """Número con formato español: punto de miles y coma decimal."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(valor: float, decimales: int = 1) -> str:
    return f"{num(valor, decimales)} %"


def libras(valor: float, decimales: int = 0) -> str:
    return f"{num(valor, decimales)} £"


# ---------------------------------------------------------------------------
# Página y gráficos
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
.etiqueta-segmento {display:inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.85rem;
                    font-weight: 600; color: #ffffff;}
.barra-color {height: 6px; border-radius: 3px; margin-bottom: 0.5rem;}
.texto-secundario {color: #52514e; font-size: 0.92rem;}
.cifra-grande {font-size: 2rem; font-weight: 700; line-height: 1.1; color: #0b0b0b;}
</style>
"""


def configurar_pagina(titulo: str, icono: str = "📊") -> None:
    """Configura la página (título de pestaña, ancho) y aplica el estilo común."""
    st.set_page_config(page_title=f"{titulo} · Segmentación de clientes", page_icon=icono, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def etiqueta_segmento(segmento: int) -> str:
    """HTML de una etiqueta con el color del segmento (el texto va en blanco sobre el color)."""
    return (f'<span class="etiqueta-segmento" style="background:{COLORES_SEGMENTO[segmento]}">'
            f"{NOMBRES[segmento]}</span>")


def barra_color(segmento: int) -> str:
    return f'<div class="barra-color" style="background:{COLORES_SEGMENTO[segmento]}"></div>'


def estilo_figura(fig: go.Figure, altura: int = 420, leyenda: bool = True) -> go.Figure:
    """Estilo común de Plotly: fondo claro, rejilla discreta, textos en tinta neutra."""
    fig.update_layout(
        height=altura, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
        font=dict(color=TEXTO, size=13), title_font=dict(size=16),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if leyenda else None,
        showlegend=leyenda, hoverlabel=dict(bgcolor="white", font_color=TEXTO),
    )
    fig.update_xaxes(gridcolor="#ececea", zeroline=False)
    fig.update_yaxes(gridcolor="#ececea", zeroline=False)
    return fig


def nota_tsne() -> str:
    """Advertencia sobre t-SNE (texto guardado en metricas.json en el notebook 03)."""
    return cargar_json("metricas")["tsne"]["nota"]


MARCADOR_CUENTA_CLAVE_PLOTLY = dict(symbol="diamond", size=10, line=dict(color=COLOR_BORDE_CUENTA_CLAVE, width=1.2))
