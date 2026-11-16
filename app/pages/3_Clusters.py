"""Clusters: mapas PCA 2D/3D y t-SNE con selector de algoritmo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import (COLOR_POR_NOMBRE, MARCADOR_CUENTA_CLAVE_PLOTLY, NOMBRE_CUENTAS_CLAVE, ORDEN_NOMBRES, cargar_clientes,
                   cargar_json, configurar_pagina, estilo_figura, nota_tsne, num, pct)

configurar_pagina("Clusters", "🗺️")
st.title("Visualización de los clusters")

clientes = cargar_clientes().reset_index()
metricas = cargar_json("metricas")
varianza = metricas["pca"]["varianza_explicada"]

c1, c2, c3 = st.columns([2, 2, 1.4])
algoritmo = c1.selectbox("Algoritmo", ["K-Means (modelo final)", f"GMM ({metricas['gmm']['k']} componentes, BIC)",
                                       "DBSCAN (núcleo / atípicos)"])
vista = c2.radio("Proyección", ["PCA 2D", "PCA 3D", "t-SNE"], horizontal=True)
resaltar = c3.toggle(f"Resaltar {NOMBRE_CUENTAS_CLAVE.lower()}", value=algoritmo.startswith("K-Means"))

if vista == "t-SNE":
    st.warning(f"**t-SNE solo sirve para visualizar.** {nota_tsne()} En este proyecto se comprobó que el segmento más "
               "disperso en la realidad aparece como el más compacto en el mapa.", icon="⚠️")

# Grupo y color según el algoritmo elegido.
GMM_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
if algoritmo.startswith("K-Means"):
    clientes["grupo"] = clientes["nombre_segmento"].astype(str)
    orden = ORDEN_NOMBRES
    colores = COLOR_POR_NOMBRE
elif algoritmo.startswith("GMM"):
    clientes["grupo"] = "Componente " + clientes["gmm"].astype(str)
    orden = sorted(clientes["grupo"].unique(), key=lambda g: int(g.split()[-1]))
    colores = dict(zip(orden, GMM_COLORES))
else:
    clientes["grupo"] = clientes["dbscan"].map(lambda e: "Atípico (ruido)" if e == -1 else "Núcleo denso")
    orden = ["Núcleo denso", "Atípico (ruido)"]
    colores = {"Núcleo denso": "#c3c2b7", "Atípico (ruido)": "#0b0b0b"}

if vista == "PCA 2D":
    ejes = ("pc1", "pc2", None)
    titulos = (f"PC1 · actividad / valor ({pct(100 * varianza[0])})", f"PC2 · tamaño del pedido ({pct(100 * varianza[1])})", None)
elif vista == "PCA 3D":
    ejes = ("pc1", "pc2", "pc3")
    titulos = (f"PC1 ({pct(100 * varianza[0])})", f"PC2 ({pct(100 * varianza[1])})", f"PC3 ({pct(100 * varianza[2])})")
else:
    ejes = ("tsne_x", "tsne_y", None)
    titulos = ("t-SNE 1 (sin unidades)", "t-SNE 2 (sin unidades)", None)

datos_hover = clientes[["CustomerID", "nombre_segmento", "recency", "frequency", "ticket_promedio",
                        "productos_distintos", "monetary", "pais"]].astype({"nombre_segmento": str}).to_numpy()
plantilla = ("<b>Cliente %{customdata[0]}</b><br>Segmento: %{customdata[1]}<br>Días desde la última compra: %{customdata[2]}"
             "<br>Facturas: %{customdata[3]}<br>Ticket promedio: %{customdata[4]:,.2f} £<br>Productos distintos: %{customdata[5]}"
             "<br>Gasto total: %{customdata[6]:,.2f} £<br>País: %{customdata[7]}<extra></extra>")

fig = go.Figure()
for grupo in orden:
    m = (clientes["grupo"] == grupo).to_numpy()
    if resaltar:
        m = m & ~clientes["cuenta_clave"].to_numpy()
    comun_args = dict(name=grupo, mode="markers", customdata=datos_hover[m], hovertemplate=plantilla,
                      marker=dict(color=colores[grupo], size=4 if vista == "PCA 3D" else 6, opacity=0.65))
    if ejes[2]:
        fig.add_trace(go.Scatter3d(x=clientes.loc[m, ejes[0]], y=clientes.loc[m, ejes[1]], z=clientes.loc[m, ejes[2]], **comun_args))
    else:
        fig.add_trace(go.Scattergl(x=clientes.loc[m, ejes[0]], y=clientes.loc[m, ejes[1]], **comun_args))
if resaltar:
    m = clientes["cuenta_clave"].to_numpy()
    color_ck = colores.get(clientes.loc[m, "grupo"].iloc[0], "#2a78d6") if algoritmo.startswith("K-Means") else "#2a78d6"
    marcador = {**MARCADOR_CUENTA_CLAVE_PLOTLY, "color": color_ck}
    args = dict(name=NOMBRE_CUENTAS_CLAVE, mode="markers", customdata=datos_hover[m], hovertemplate=plantilla, marker=marcador)
    if ejes[2]:
        fig.add_trace(go.Scatter3d(x=clientes.loc[m, ejes[0]], y=clientes.loc[m, ejes[1]], z=clientes.loc[m, ejes[2]],
                                   **{**args, "marker": {**marcador, "size": 6}}))
    else:
        fig.add_trace(go.Scattergl(x=clientes.loc[m, ejes[0]], y=clientes.loc[m, ejes[1]], **args))

fig.update_layout(title=f"{algoritmo} · {vista}", legend=dict(itemsizing="constant"))
if ejes[2]:
    fig.update_layout(scene=dict(xaxis_title=titulos[0], yaxis_title=titulos[1], zaxis_title=titulos[2]))
else:
    fig.update_xaxes(title=titulos[0])
    fig.update_yaxes(title=titulos[1])
    if vista == "t-SNE":
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
st.plotly_chart(estilo_figura(fig, 640), config={"displayModeBar": True, "displaylogo": False})

# Resumen de tamaños del algoritmo elegido.
resumen = clientes.groupby("grupo").agg(clientes=("CustomerID", "size"), gasto_mediano=("monetary", "median"),
                                        ingresos=("monetary", "sum")).reindex(orden)
resumen["% clientes"] = 100 * resumen["clientes"] / resumen["clientes"].sum()
resumen["% ingresos"] = 100 * resumen["ingresos"] / resumen["ingresos"].sum()
st.dataframe(resumen.drop(columns="ingresos").rename_axis("Grupo"), column_config={
    "clientes": st.column_config.NumberColumn("Clientes", format="localized"),
    "gasto_mediano": st.column_config.NumberColumn("Gasto mediano (£)", format="%.2f"),
    "% clientes": st.column_config.NumberColumn(format="%.1f %%"), "% ingresos": st.column_config.NumberColumn(format="%.1f %%")})

if not algoritmo.startswith("K-Means"):
    st.info("Los colores de GMM y DBSCAN no son los segmentos del modelo final: se muestran para comparar algoritmos. "
            "GMM se descartó por su baja estabilidad, y DBSCAN solo separa un núcleo denso de los clientes atípicos.")

with st.expander("¿Qué representan los ejes de PCA?"):
    cargas = pd.DataFrame(metricas["pca"]["cargas"]).rename(index={
        "recency": "Días desde la última compra", "frequency": "Facturas", "ticket_promedio": "Ticket promedio",
        "productos_distintos": "Productos distintos"})
    st.markdown(f"PC1 y PC2 explican juntas el **{pct(100 * (varianza[0] + varianza[1]))}** de la varianza. Las cargas indican "
                "el peso de cada variable en cada componente:")
    st.dataframe(cargas.style.format("{:.2f}"))
    st.markdown("- **PC1 (actividad / valor):** más facturas y productos distintos, compras más recientes.\n"
                "- **PC2 (tamaño del pedido):** ticket alto y compras más espaciadas.")
st.caption(f"{num(len(clientes))} clientes. Pasa el cursor por un punto para ver sus datos.")
