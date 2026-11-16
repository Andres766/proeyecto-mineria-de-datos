"""Perfiles y estrategias: ficha de cada segmento, radar, snake plot, tabla comparativa y estrategias con KPI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import (COLORES_SEGMENTO, ETIQUETAS, NOMBRE_CUENTAS_CLAVE, NOMBRES, barra_color, cargar_json,
                   configurar_pagina, estilo_figura, libras, num, pct)

configurar_pagina("Perfiles y estrategias", "🎯")
st.title("Perfiles y estrategias")

datos = cargar_json("segmentos")
segmentos = {g["id"]: g for g in datos["segmentos"]}
cuentas = datos["cuentas_clave"]
IDS = sorted(segmentos)
FORMATO_BASE = {"ticket_mediano": lambda v: libras(v, 2), "ingresos_temporada": lambda v: libras(v, 2)}


def linea_base(estrategia: dict) -> str:
    formato = FORMATO_BASE.get(estrategia["indicador"], pct)
    return formato(estrategia["linea_base"])


def mostrar_estrategias(estrategias: list[dict]) -> None:
    tabla = pd.DataFrame([{"Acción": e["accion"], "Momento": e["momento"], "KPI": e["kpi"],
                           "Línea base actual": linea_base(e)} for e in estrategias])
    st.dataframe(tabla, hide_index=True, column_config={
        "Acción": st.column_config.TextColumn(width="large"), "KPI": st.column_config.TextColumn(width="medium")})


# ------------------------------------------------------------------ Fichas
st.subheader("Ficha de cada segmento")
columnas = st.columns(len(IDS))
for col, s in zip(columnas, IDS):
    g, ind = segmentos[s], segmentos[s]["indicadores"]
    with col.container(border=True):
        st.markdown(barra_color(s) + f"**{g['nombre']}**", unsafe_allow_html=True)
        st.markdown(f"<span class='texto-secundario'>{g['descripcion']}</span>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Clientes", pct(ind["pct_clientes"]))
        b.metric("Ingresos", pct(ind["pct_ingresos"]))
        st.markdown(
            f"- Gasto mediano: **{libras(ind['monetary_mediano'])}**\n"
            f"- Ticket mediano: **{libras(ind['ticket_mediano'])}**\n"
            f"- Facturas (mediana): **{num(ind['frecuencia_mediana'])}**\n"
            f"- Compraron en los últimos 90 días: **{pct(ind['pct_compra_90d'])}**\n"
            f"- Con una sola compra: **{pct(ind['pct_una_compra'])}**\n"
            f"- País principal: {g['pais']['pais_principal']} ({pct(g['pais']['pct_pais_principal'])})"
        )

ic = cuentas["indicadores"]
with st.container(border=True):
    st.markdown(barra_color(3) + f"**{NOMBRE_CUENTAS_CLAVE}** · subgrupo de {NOMBRES[3]}", unsafe_allow_html=True)
    izq, der = st.columns([2, 3])
    with izq:
        a, b, c = st.columns(3)
        a.metric("Clientes", num(ic["clientes"]))
        b.metric("% de clientes", pct(ic["pct_clientes"]))
        c.metric("% de ingresos", pct(ic["pct_ingresos"]))
        st.caption(f"Criterio: {cuentas['criterio']} (umbral {libras(cuentas['umbral_gasto'], 2)}).")
    der.markdown(
        f"{cuentas['descripcion']}\n\nMediana: **{num(ic['frecuencia_mediana'])} facturas**, ticket de "
        f"**{libras(ic['ticket_mediano'])}** y gasto de **{libras(ic['monetary_mediano'])}**.")

# ------------------------------------------------------------------ Radar y snake plot
st.subheader("Comparación visual")
c1, c2 = st.columns(2)
ejes_radar = {"recency": "Recencia (reciente = alto)", "frequency": "Frecuencia", "ticket_promedio": "Ticket",
              "productos_distintos": "Variedad", "monetary": "Gasto", "antiguedad": "Antigüedad"}
fig = go.Figure()
for s in IDS:
    valores = [segmentos[s]["radar_percentil"][v] for v in ejes_radar]
    fig.add_trace(go.Scatterpolar(r=valores + valores[:1], theta=list(ejes_radar.values()) + [list(ejes_radar.values())[0]],
                                  name=NOMBRES[s], line=dict(color=COLORES_SEGMENTO[s], width=2.5), fill="toself",
                                  fillcolor=COLORES_SEGMENTO[s] + "1a",  # mismo color con ~10 % de opacidad
                                  hovertemplate=f"{NOMBRES[s]}<br>%{{theta}}: percentil %{{r:.0f}}<extra></extra>"))
fig.update_layout(title="Radar: percentil medio de cada segmento (0–100)", polar=dict(radialaxis=dict(range=[0, 100])))
c1.plotly_chart(estilo_figura(fig, 470), config={"displayModeBar": False})

vars_snake = ["recency", "frequency", "ticket_promedio", "productos_distintos"]
fig = go.Figure()
for s in IDS:
    fig.add_scatter(x=[ETIQUETAS[v] for v in vars_snake], y=[segmentos[s]["snake_z"][v] for v in vars_snake],
                    mode="lines+markers", name=NOMBRES[s], line=dict(color=COLORES_SEGMENTO[s], width=2.5), marker=dict(size=9),
                    hovertemplate=f"{NOMBRES[s]}<br>%{{x}}: %{{y:+.2f}} desv. típicas<extra></extra>")
fig.add_hline(y=0, line_color="#9AA5B1", line_width=1)
fig.update_layout(title="Snake plot: media estandarizada de las variables del modelo")
fig.update_yaxes(title="desviaciones típicas respecto a la media")
c2.plotly_chart(estilo_figura(fig, 470), config={"displayModeBar": False})
st.caption("En el radar, los ejes son percentiles (misma escala para todas las variables). En el snake plot, 0 es el "
           "cliente medio y ±1 una desviación típica.")

# ------------------------------------------------------------------ Tabla comparativa
st.subheader("Tabla comparativa (medianas)")
variables = ["recency", "frequency", "monetary", "ticket_promedio", "productos_distintos", "antiguedad",
             "dias_entre_compras", "unidades_por_factura"]
tabla = pd.DataFrame({NOMBRES[s]: {ETIQUETAS[v]: segmentos[s]["medianas"][v] for v in variables} for s in IDS})
tabla[NOMBRE_CUENTAS_CLAVE] = [cuentas["medianas"][v] for v in variables]
extra = pd.DataFrame({NOMBRES[s]: {"% con devoluciones": segmentos[s]["indicadores"]["pct_devuelve"],
                                   "% facturas en sep–nov": segmentos[s]["indicadores"]["pct_facturas_temporada"]} for s in IDS})
extra[NOMBRE_CUENTAS_CLAVE] = [ic["pct_devuelve"], ic["pct_facturas_temporada"]]
st.dataframe(pd.concat([tabla, extra]).style.format("{:,.1f}"))

# ------------------------------------------------------------------ Estrategias
st.subheader("Estrategias de marketing")
st.markdown("Cada estrategia tiene una **acción concreta**, el **momento** de aplicarla y el **KPI** que la mide. La "
            "**línea base** es el valor actual del indicador, calculado con los datos; sirve para medir el efecto (no se "
            "fijan metas inventadas).")
pestanas = st.tabs([NOMBRES[s] for s in IDS] + [NOMBRE_CUENTAS_CLAVE])
for pestana, s in zip(pestanas, IDS):
    with pestana:
        mostrar_estrategias(segmentos[s]["estrategias"])
with pestanas[-1]:
    mostrar_estrategias(cuentas["estrategias"])

# ------------------------------------------------------------------ Estacionalidad y RFM
st.subheader("Estacionalidad y coherencia con RFM")
c1, c2 = st.columns(2)
est = pd.DataFrame(datos["estacionalidad_pct_facturas"]).T
est.index = [NOMBRES[int(i)] if i != "Total" else "Total" for i in est.index]
fig = go.Figure(go.Heatmap(z=est.values, x=est.columns, y=est.index, colorscale="Blues", text=est.round(1).values,
                           texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z:.1f} % de sus facturas<extra></extra>",
                           colorbar=dict(title="%")))
fig.update_layout(title="% de las facturas de cada segmento por mes")
fig.update_yaxes(autorange="reversed")
c1.plotly_chart(estilo_figura(fig, 380, leyenda=False), config={"displayModeBar": False})

rfm = pd.DataFrame({NOMBRES[s]: segmentos[s]["composicion_rfm_pct"] for s in IDS}).T
fig = go.Figure(go.Heatmap(z=rfm.values, x=rfm.columns, y=rfm.index, colorscale="Blues", zmin=0, zmax=100,
                           text=rfm.round(0).values, texttemplate="%{text}", colorbar=dict(title="%"),
                           hovertemplate="%{y}: %{z:.1f} % son «%{x}» en RFM<extra></extra>"))
fig.update_layout(title="Composición de cada segmento según el baseline RFM")
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(tickangle=-35)
c2.plotly_chart(estilo_figura(fig, 380, leyenda=False), config={"displayModeBar": False})
st.caption("Los Grandes ocasionales hacen muchas de sus compras únicas en septiembre–octubre (aprovisionamiento para "
           "Navidad). En RFM, VIP equivale a Campeones + Leales, y los dos grupos de ocasionales se confunden.")

with st.expander("Productos más comprados por segmento"):
    cols = st.columns(len(IDS))
    for col, s in zip(cols, IDS):
        col.markdown(f"**{NOMBRES[s]}**")
        col.dataframe(pd.DataFrame(segmentos[s]["productos_top_penetracion"]).rename(
            columns={"descripcion": "Producto", "valor": "% clientes"}), hide_index=True,
            column_config={"% clientes": st.column_config.NumberColumn(format="%.1f %%")})
    st.caption("Los productos estrella son casi los mismos en todos los segmentos: lo que los diferencia es cuánto, cada "
               "cuánto y qué variedad compran.")
