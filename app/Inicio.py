"""Página de inicio: objetivo, KPIs y hallazgos clave. Todas las cifras se leen de los artefactos."""

import plotly.graph_objects as go
import streamlit as st

from comun import (COLORES_SEGMENTO, MESES, NEUTRO, NOMBRE_CUENTAS_CLAVE, NOMBRES, cargar_clientes, cargar_json,
                   cargar_ventas, configurar_pagina, estilo_figura, libras, num, pct)

configurar_pagina("Inicio", "🏠")

clientes = cargar_clientes()
fechas = cargar_ventas()["InvoiceDate"]
inicio, fin = fechas.min(), fechas.max()
metricas = cargar_json("metricas")
segmentos = {g["id"]: g for g in cargar_json("segmentos")["segmentos"]}
cuentas = cargar_json("segmentos")["cuentas_clave"]
final = metricas["modelo_final"]
comparativa = {fila["modelo"]: fila for fila in metricas["comparativa"]}

st.title("Segmentación avanzada de clientes")
st.markdown(
    "Proyecto final de **Inteligencia de Negocios y Minería de Datos**. A partir de las transacciones de un minorista "
    f"online británico de regalos (dataset *Online Retail*, UCI; {MESES[inicio.month]} {inicio.year} – "
    f"{MESES[fin.month]} {fin.year}) se identifican "
    "perfiles de clientes con **clustering (K-Means, GMM, DBSCAN)** y **reducción de dimensionalidad (PCA, t-SNE)** "
    "para orientar estrategias de marketing."
)

# --------------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes analizados", num(len(clientes)), border=True)
ingresos = clientes["monetary"].sum()
c2.metric("Ingresos analizados", f"{num(ingresos / 1e6, 2)} M£", border=True,
          help=f"Exactamente {libras(ingresos)}.")
c3.metric("Segmentos", str(final["k"]), border=True,
          help=f"Dentro de VIP hay además un subgrupo de {num(cuentas['indicadores']['clientes'])} "
               f"{NOMBRE_CUENTAS_CLAVE.lower()} (ver «Hallazgos clave»).")
c4.metric("Silhouette", num(final["silhouette"], 3),
          help=f"Del modelo final: K-Means con k = {final['k']}. Estabilidad (ARI mediano en submuestras): {num(final['estabilidad_mediana'], 3)}.",
          border=True)

# --------------------------------------------------------------------- Hallazgos clave
st.subheader("Hallazgos clave")
h1, h2, h3 = st.columns(3)

riesgo = ["Necesitan atención", "A punto de dormir", "En riesgo", "No se pueden perder", "Hibernando"]
peq, gra = segmentos[0], segmentos[1]
inactivos_peq = sum(peq["composicion_rfm_pct"][r] for r in riesgo)
inactivos_gra = sum(gra["composicion_rfm_pct"][r] for r in riesgo)
with h1.container(border=True):
    st.markdown("**RFM no distingue a los ocasionales; K-Means sí**")
    st.markdown(
        f"Para el baseline RFM, «{peq['nombre']}» y «{gra['nombre']}» son casi lo mismo: el "
        f"{pct(inactivos_peq, 0)} y el {pct(inactivos_gra, 0)} son clientes hibernando o en riesgo. "
        f"K-Means los separa por el **tamaño del pedido** (ticket mediano {libras(peq['indicadores']['ticket_mediano'])} "
        f"frente a {libras(gra['indicadores']['ticket_mediano'])}), y su valor es muy distinto: "
        f"**{pct(peq['indicadores']['pct_ingresos'])}** frente a **{pct(gra['indicadores']['pct_ingresos'])}** de los ingresos."
    )

ic = cuentas["indicadores"]
with h2.container(border=True):
    st.markdown(f"**{NOMBRE_CUENTAS_CLAVE}: pocos clientes, mucho negocio**")
    st.markdown(
        f"<div class='cifra-grande'>{pct(ic['pct_clientes'])} → {pct(ic['pct_ingresos'])}</div>"
        f"<div class='texto-secundario'>de los clientes · de los ingresos</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"{num(ic['clientes'])} clientes VIP que DBSCAN marca como atípicos y que superan el gasto mediano del resto de VIP. "
        f"Compran {num(ic['frecuencia_mediana'])} veces (mediana) con un ticket de {libras(ic['ticket_mediano'])}: "
        "requieren gestión de cuenta dedicada."
    )

k2, k5 = comparativa["K-Means k=2"], comparativa["K-Means k=5"]
with h3.container(border=True):
    st.markdown(f"**Por qué k = {final['k']} y no k = 2**")
    st.markdown(
        f"Las métricas internas premian k = 2 (silhouette {num(k2['silhouette'], 3)} frente a {num(final['silhouette'], 3)}), "
        "pero el silhouette baja con cada k: los clientes forman una nube continua, sin grupos naturales, y en ese caso "
        "siempre gana el k más pequeño. Con k = 2 solo se separa «activos» de «el resto», que no permite estrategias "
        f"distintas. k = {final['k']} da segmentos accionables y estables (ARI en el peor 10 % de submuestras "
        f"{num(final['estabilidad_p10'], 3)}, frente a {num(k5['estabilidad_p10'], 3)} con k = 5)."
    )

# --------------------------------------------------------------------- Segmentos de un vistazo
st.subheader("Los segmentos de un vistazo")
izq, der = st.columns([3, 2])
with izq:
    fig = go.Figure()
    ids = sorted(segmentos)
    fig.add_bar(y=[NOMBRES[s] for s in ids], x=[segmentos[s]["indicadores"]["pct_clientes"] for s in ids],
                orientation="h", name="% de clientes", marker_color=NEUTRO,
                hovertemplate="%{y}: %{x:.1f} % de los clientes<extra></extra>",
                text=[pct(segmentos[s]["indicadores"]["pct_clientes"]) for s in ids], textposition="outside",
                textfont=dict(color="#52514e"))
    fig.add_bar(y=[NOMBRES[s] for s in ids], x=[segmentos[s]["indicadores"]["pct_ingresos"] for s in ids],
                orientation="h", name="% de ingresos", marker_color=[COLORES_SEGMENTO[s] for s in ids],
                hovertemplate="%{y}: %{x:.1f} % de los ingresos<extra></extra>",
                text=[pct(segmentos[s]["indicadores"]["pct_ingresos"]) for s in ids], textposition="outside")
    fig.update_layout(barmode="group", bargap=0.25,
                      title="% de ingresos (color) frente a % de clientes (gris)")
    fig.update_xaxes(title="%", range=[0, 85])
    st.plotly_chart(estilo_figura(fig, 380, leyenda=False), config={"displayModeBar": False})
with der:
    for s in sorted(segmentos):
        g = segmentos[s]
        st.markdown(
            f"<span class='etiqueta-segmento' style='background:{g['color']}'>{g['nombre']}</span> "
            f"<span class='texto-secundario'>{g['descripcion'].split('.')[0]}.</span>",
            unsafe_allow_html=True,
        )

st.subheader("Cómo navegar")
st.markdown(
    "- **Exploración:** calidad de los datos, limpieza y patrones de venta.\n"
    "- **Modelos y validación:** comparación de K-Means, GMM y DBSCAN, y por qué se eligió el modelo final.\n"
    "- **Clusters:** mapas PCA 2D/3D y t-SNE con selector de algoritmo.\n"
    "- **Perfiles y estrategias:** ficha de cada segmento, radar y acciones de marketing con su KPI.\n"
    "- **Explorador:** filtrar y buscar clientes, descargar CSV y **clasificar un cliente nuevo**."
)
st.caption("Todas las cifras de esta app se calculan con el código del proyecto y se leen de los artefactos generados "
           "por los notebooks (`models/*.json`, `data/processed/*.parquet`).")
