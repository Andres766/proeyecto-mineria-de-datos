"""Exploración: calidad de datos, limpieza y patrones de venta (gráficos interactivos del EDA)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from comun import (COLOR_POR_NOMBRE, MESES, NEUTRO, ORDEN_NOMBRES, cargar_bitacora, cargar_clientes, cargar_ventas,
                   configurar_pagina, estilo_figura, libras, num, pct)

configurar_pagina("Exploración", "🔎")
st.title("Exploración de los datos")

ventas = cargar_ventas()
bitacora = cargar_bitacora()
clientes = cargar_clientes()
AZUL = "#2a78d6"
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


@st.cache_data
def agregados(ventas: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agregaciones del EDA calculadas una sola vez."""
    facturas = ventas.groupby("InvoiceNo").agg(fecha=("InvoiceDate", "min"), importe=("Importe", "sum"))
    mensual = (ventas.set_index("InvoiceDate").resample("MS")
               .agg({"Importe": "sum", "InvoiceNo": "nunique", "CustomerID": "nunique"})
               .rename(columns={"Importe": "ingresos", "InvoiceNo": "facturas", "CustomerID": "clientes"}))
    por_dia = facturas.groupby(facturas["fecha"].dt.day_of_week).size().reindex(range(7), fill_value=0)
    por_hora = facturas.groupby(facturas["fecha"].dt.hour).size()
    paises = (ventas.groupby("Country").agg(ingresos=("Importe", "sum"), clientes=("CustomerID", "nunique"))
              .sort_values("ingresos", ascending=False))
    desc = ventas.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    productos = ventas.groupby("StockCode").agg(ingresos=("Importe", "sum"), unidades=("Quantity", "sum")).join(desc)
    return {"mensual": mensual, "por_dia": por_dia, "por_hora": por_hora, "paises": paises, "productos": productos}


ag = agregados(ventas)

# ------------------------------------------------------------------ Limpieza
st.subheader("1. Limpieza de los datos")
inicial, final = int(bitacora["filas_antes"].iloc[0]), int(bitacora["filas_despues"].iloc[-1])
c1, c2, c3 = st.columns(3)
c1.metric("Filas originales", num(inicial), border=True)
c2.metric("Filas tras la limpieza", num(final), border=True)
c3.metric("Clientes", num(ventas["CustomerID"].nunique()), border=True)
fig = go.Figure(go.Bar(
    y=["0. Original"] + bitacora["paso"].tolist(), x=[inicial] + bitacora["filas_despues"].tolist(), orientation="h",
    marker_color=[NEUTRO] + [AZUL] * len(bitacora), text=[num(v) for v in [inicial] + bitacora["filas_despues"].tolist()],
    textposition="outside", hovertemplate="%{y}<br>%{x:,} filas<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[0, inicial * 1.18], title="filas", tickformat=",.0f")
fig.update_layout(title="Filas restantes tras cada paso")
st.plotly_chart(estilo_figura(fig, 380, leyenda=False), config={"displayModeBar": False})
with st.expander("Ver la bitácora completa con la justificación de cada paso"):
    st.dataframe(bitacora.style.format({"filas_antes": "{:,.0f}", "filas_despues": "{:,.0f}", "filas_eliminadas": "{:,.0f}",
                                        "pct_eliminado": "{:.2f} %"}, decimal=",", thousands="."),
                 hide_index=True, column_config={
        "paso": "Paso", "filas_antes": "Antes", "filas_despues": "Después", "filas_eliminadas": "Eliminadas",
        "pct_eliminado": "% eliminado", "justificacion": st.column_config.TextColumn("Justificación", width="large")})

# ------------------------------------------------------------------ Temporal
st.subheader("2. Ventas en el tiempo")
mensual = ag["mensual"].copy()
NOMBRE_MEDIDA = {"ingresos": "Ingresos (£)", "facturas": "Facturas", "clientes": "Clientes activos"}
medida = st.radio("Medida", list(NOMBRE_MEDIDA), horizontal=True, format_func=NOMBRE_MEDIDA.get)
incompleto = mensual.index == mensual.index.max()
etiquetas_mes = [f"{MESES[f.month][:3]} {f.year}" for f in mensual.index]
fig = go.Figure(go.Bar(
    x=etiquetas_mes, y=mensual[medida], marker_color=np.where(incompleto, NEUTRO, AZUL),
    hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
fig.add_annotation(x=etiquetas_mes[-1], y=mensual[medida].iloc[-1], text="incompleto<br>(hasta el 9)",
                   showarrow=True, arrowhead=0, ax=0, ay=-34, font=dict(size=10, color="#52514e"))
fig.update_xaxes(type="category")
fig.update_yaxes(tickformat=",.0f")
fig.update_layout(title=f"{NOMBRE_MEDIDA[medida]} por mes")
st.plotly_chart(estilo_figura(fig, 380, leyenda=False), config={"displayModeBar": False})
pico = mensual["ingresos"].idxmax()
st.caption(f"Pico de ingresos en {MESES[pico.month]} de {pico.year}: {libras(mensual.loc[pico, 'ingresos'])}. Diciembre de 2011 aparece en gris "
           "porque solo tiene datos hasta el día 9: su caída no es real.")

c1, c2 = st.columns(2)
por_dia = ag["por_dia"]
fig = go.Figure(go.Bar(x=DIAS, y=por_dia.values, marker_color=AZUL, hovertemplate="%{x}: %{y:,} facturas<extra></extra>"))
fig.update_layout(title="Facturas por día de la semana")
c1.plotly_chart(estilo_figura(fig, 340, leyenda=False), config={"displayModeBar": False})
por_hora = ag["por_hora"]
fig = go.Figure(go.Bar(x=por_hora.index, y=por_hora.values, marker_color=AZUL, hovertemplate="%{x} h: %{y:,} facturas<extra></extra>"))
fig.update_layout(title="Facturas por hora del día")
fig.update_xaxes(title="hora", dtick=1)
c2.plotly_chart(estilo_figura(fig, 340, leyenda=False), config={"displayModeBar": False})
st.caption(f"Sábados: {num(por_dia.iloc[5])} facturas. Hora con más facturas: {por_hora.idxmax()} h. "
           "Horario laboral y sin sábados: clientela mayoritariamente B2B.")

# ------------------------------------------------------------------ Países y productos
st.subheader("3. Países y productos")
c1, c2 = st.columns(2)
paises = ag["paises"]
sin_uk = c1.toggle("Excluir Reino Unido", value=False)
top = (paises.drop(index="United Kingdom") if sin_uk else paises).head(10)
fig = go.Figure(go.Bar(y=top.index, x=top["ingresos"], orientation="h", marker_color=AZUL,
                       customdata=top["clientes"], hovertemplate="%{y}<br>%{x:,.0f} £ · %{customdata} clientes<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_layout(title="Top 10 países por ingresos" + (" (sin Reino Unido)" if sin_uk else ""))
fig.update_xaxes(tickformat=",.0f", title="£")
c1.plotly_chart(estilo_figura(fig, 400, leyenda=False), config={"displayModeBar": False})
uk = paises.loc["United Kingdom"]
c1.caption(f"Reino Unido: {pct(100 * uk['clientes'] / paises['clientes'].sum())} de los clientes y "
           f"{pct(100 * uk['ingresos'] / paises['ingresos'].sum())} de los ingresos.")

criterio = c2.radio("Ordenar productos por", ["ingresos", "unidades"], horizontal=True,
                    format_func={"ingresos": "Ingresos", "unidades": "Unidades"}.get)
prods = ag["productos"].nlargest(10, criterio)
fig = go.Figure(go.Bar(y=prods["Description"].str.slice(0, 34), x=prods[criterio], orientation="h", marker_color=AZUL,
                       hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_layout(title=f"Top 10 productos por {criterio}")
fig.update_xaxes(tickformat=",.0f")
c2.plotly_chart(estilo_figura(fig, 400, leyenda=False), config={"displayModeBar": False})

# ------------------------------------------------------------------ Distribuciones
st.subheader("4. Distribución del gasto por cliente")
st.markdown("El gasto es muy asimétrico: unos pocos clientes concentran gran parte de los ingresos. En escala logarítmica "
            "la distribución se vuelve aproximadamente simétrica, por eso el modelo usa `log1p`.")
por_segmento = st.toggle("Colorear por segmento", value=True)
# Histograma sobre log10 del gasto (eje logarítmico) con las marcas del eje en libras.
datos_hist = clientes.assign(log_gasto=np.log10(clientes["monetary"]), segmento=clientes["nombre_segmento"].astype(str))
fig = px.histogram(datos_hist, x="log_gasto", color="segmento" if por_segmento else None, nbins=60,
                   color_discrete_map=COLOR_POR_NOMBRE, category_orders={"segmento": ORDEN_NOMBRES}, barmode="stack")
if not por_segmento:
    fig.update_traces(marker_color=AZUL)
marcas = [10, 100, 1_000, 10_000, 100_000]
fig.update_xaxes(tickvals=np.log10(marcas), ticktext=[libras(m) for m in marcas],
                 title="Gasto total por cliente (escala logarítmica)")
fig.update_traces(hovertemplate="%{y} clientes<extra></extra>")
fig.update_layout(title="Clientes por gasto total", yaxis_title="clientes", bargap=0.03)
st.plotly_chart(estilo_figura(fig, 400, leyenda=por_segmento), config={"displayModeBar": False})
orden = clientes["monetary"].sort_values(ascending=False)
uno = int(np.ceil(len(orden) * 0.01))
st.caption(f"El 1 % de los clientes ({num(uno)}) genera el {pct(100 * orden.iloc[:uno].sum() / orden.sum())} de los ingresos.")
