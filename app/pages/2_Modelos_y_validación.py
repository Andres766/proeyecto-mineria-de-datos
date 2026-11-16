"""Modelos y validación: métricas de los algoritmos, codo/silhouette/BIC y justificación del modelo final."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from comun import cargar_json, configurar_pagina, estilo_figura, num, pct

configurar_pagina("Modelos y validación", "🧪")
st.title("Modelos y validación")

metricas = cargar_json("metricas")
final = metricas["modelo_final"]
comparativa = pd.DataFrame(metricas["comparativa"]).set_index("modelo")
barrido = pd.DataFrame(metricas["barrido_kmeans"])
gmm = metricas["gmm"]
rejilla = pd.DataFrame(gmm["rejilla_bic"])
AZUL, NARANJA = "#2a78d6", "#eb6834"
NOMBRE_FINAL = f"K-Means k={final['k']}"

st.markdown(
    "Se compararon **K-Means**, **GMM** (mezclas gaussianas) y **DBSCAN** (más **HDBSCAN**) sobre las mismas cuatro "
    "variables por cliente: días desde la última compra, número de facturas, ticket promedio y productos distintos "
    "(en escala log1p, recortadas a los percentiles 1–99 y estandarizadas)."
)

# ------------------------------------------------------------------ Tabla comparativa
st.subheader("Comparación de los modelos")
tabla = comparativa.rename(columns={
    "n_clusters": "Clusters", "pct_ruido": "% ruido", "silhouette": "Silhouette ↑", "davies_bouldin": "Davies-Bouldin ↓",
    "calinski_harabasz": "Calinski-Harabasz ↑", "estabilidad_mediana": "Estabilidad (mediana) ↑",
    "estabilidad_p10": "Estabilidad (peor 10 %) ↑", "ARI_descuantizado": "Robustez a empates ↑",
    "pct_segmento_menor": "% segmento menor"})
st.dataframe(
    tabla.style.format(precision=3, na_rep="—").apply(
        lambda fila: ["background-color: #e6f0fb; font-weight: 600" if fila.name == NOMBRE_FINAL else "" for _ in fila], axis=1),
    column_config={"modelo": "Modelo"})
st.caption("Métricas internas calculadas en el mismo espacio para todos; en DBSCAN y HDBSCAN el ruido se excluye. "
           "Estabilidad: ARI entre el modelo completo y 50 submuestras del 80 %. Robustez a empates: ARI al repartir "
           "aleatoriamente la frecuencia dentro de su intervalo de redondeo. En azul, el modelo final.")

tab_km, tab_gmm, tab_dens, tab_ari = st.tabs(["K-Means", "GMM", "DBSCAN y HDBSCAN", "Concordancia (ARI)"])

# ------------------------------------------------------------------ K-Means
with tab_km:
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Inercia (método del codo)", "Silhouette (↑ mejor)",
                                                        "Davies-Bouldin (↓ mejor)", "Calinski-Harabasz (↑ mejor)"),
                        vertical_spacing=0.16)
    for (fila, col), medida in zip([(1, 1), (1, 2), (2, 1), (2, 2)], ["inercia", "silhouette", "davies_bouldin", "calinski_harabasz"]):
        fig.add_scatter(x=barrido["k"], y=barrido[medida], mode="lines+markers", line=dict(color=AZUL, width=2),
                        marker=dict(size=8), row=fila, col=col, showlegend=False,
                        hovertemplate=f"k = %{{x}}<br>{medida}: %{{y:.3f}}<extra></extra>")
        elegido = barrido[barrido["k"] == final["k"]]
        fig.add_scatter(x=elegido["k"], y=elegido[medida], mode="markers", row=fila, col=col, showlegend=False,
                        marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=NARANJA, width=2.5)), hoverinfo="skip")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(estilo_figura(fig, 560, leyenda=False), config={"displayModeBar": False})
    st.caption(f"Círculo naranja: k = {final['k']}, el elegido.")

    fig = go.Figure()
    fig.add_scatter(x=barrido["k"], y=barrido["estabilidad_mediana"], mode="lines+markers", name="mediana",
                    line=dict(color=AZUL, width=2), marker=dict(size=8))
    fig.add_scatter(x=barrido["k"], y=barrido["estabilidad_p10"], mode="lines+markers", name="peor 10 % (percentil 10)",
                    line=dict(color=AZUL, width=2, dash="dot"), marker=dict(size=8, symbol="square"))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#9AA5B1", annotation_text="0,8", annotation_position="right")
    fig.update_layout(title="Estabilidad por k (ARI entre submuestras y modelo completo)")
    fig.update_xaxes(title="k", dtick=1)
    fig.update_yaxes(title="ARI", range=[0.3, 1.02])
    st.plotly_chart(estilo_figura(fig, 380), config={"displayModeBar": False})

    k2 = comparativa.loc["K-Means k=2"]
    k5 = comparativa.loc["K-Means k=5"]
    st.markdown(
        f"- Las métricas internas son mejores con **k = 2** (silhouette {num(k2['silhouette'], 3)}) y empeoran poco a poco al "
        "subir k. Es lo esperable cuando los clientes forman una **nube continua** sin grupos naturales separados.\n"
        f"- **k = 5** tiene la mejor estabilidad mediana ({num(k5['estabilidad_mediana'], 3)}), pero su peor caso cae a "
        f"{num(k5['estabilidad_p10'], 3)}: hay dos soluciones casi igual de buenas y en parte de las submuestras sus "
        "segmentos se reorganizan.\n"
        f"- **k = {final['k']}** mantiene una estabilidad alta también en el peor caso ({num(final['estabilidad_p10'], 3)})."
    )

# ------------------------------------------------------------------ GMM
with tab_gmm:
    st.markdown(
        "**Problema detectado:** un tercio de los clientes tiene exactamente 1 factura. Con la regularización por defecto "
        "(`reg_covar` = 1e-6), GMM coloca componentes justo sobre ese valor, con varianza casi cero; la verosimilitud se "
        f"dispara y el BIC baja artificialmente hasta {num(gmm['bic_mejor_reg_defecto'])}. Se buscó el `reg_covar` mínimo "
        f"sin componentes degenerados: **{gmm['reg_covar']}**."
    )
    colores_tipo = {"full": "#008300", "tied": "#e34948", "diag": "#eda100", "spherical": "#e87ba4"}
    simbolos = {"full": "circle", "tied": "square", "diag": "diamond", "spherical": "triangle-up"}
    fig = go.Figure()
    for tipo, sub in rejilla.groupby("tipo"):
        fig.add_scatter(x=sub["k"], y=sub["bic"], mode="lines+markers", name=tipo, line=dict(color=colores_tipo[tipo], width=2),
                        marker=dict(size=8, symbol=simbolos[tipo]), hovertemplate=f"{tipo}, k = %{{x}}<br>BIC: %{{y:,.0f}}<extra></extra>")
    fig.update_layout(title=f"BIC por número de componentes y tipo de covarianza (reg_covar = {gmm['reg_covar']})")
    fig.update_xaxes(title="componentes", dtick=1)
    fig.update_yaxes(title="BIC (↓ mejor)")
    st.plotly_chart(estilo_figura(fig, 420), config={"displayModeBar": False})
    elegido = comparativa.filter(like="(BIC)", axis=0).iloc[0]
    st.markdown(
        f"El menor BIC corresponde a covarianza **{gmm['tipo']}** con **{gmm['k']} componentes**, pero ese modelo tiene un "
        f"silhouette de {num(elegido['silhouette'], 3)}, una estabilidad mediana de {num(elegido['estabilidad_mediana'], 3)} y "
        f"un componente de solo el {pct(elegido['pct_segmento_menor'])} de los clientes. Se usa para comparar, no como modelo final."
    )

# ------------------------------------------------------------------ DBSCAN / HDBSCAN
with tab_dens:
    db = metricas["dbscan"]
    hd = comparativa.loc["HDBSCAN"]
    st.markdown(
        f"- **DBSCAN** (`eps` = {num(db['eps'], 3)} tomado del codo del gráfico de k-distancias, `min_samples` = "
        f"{db['min_samples']} = 2 × número de variables) encuentra **un único grupo denso** y marca como ruido al "
        f"**{pct(db['pct_ruido'])}** de los clientes. Esos clientes atípicos son, sobre todo, compradores de volumen "
        "muy alto. Por eso DBSCAN no se usa para segmentar, sino para **marcar atípicos**; cruzado con los segmentos, "
        "da el subgrupo de cuentas clave.\n"
        "- Con `eps` pequeños DBSCAN parecía encontrar grupos, pero eran **falsas zonas densas**: rebanadas de clientes "
        "con el mismo número de facturas (los empates hacen que esos puntos parezcan más densos).\n"
        f"- **HDBSCAN** separa exactamente a los clientes de 1 factura del resto. Al romper los empates esa estructura "
        f"desaparece (ARI {num(hd['ARI_descuantizado'], 3)}): es un artefacto y no se usa."
    )

# ------------------------------------------------------------------ Concordancia
with tab_ari:
    matriz = pd.DataFrame(metricas["matriz_ari"])
    fig = go.Figure(go.Heatmap(z=matriz.values, x=matriz.columns, y=matriz.index, colorscale="Blues", zmin=0, zmax=1,
                               text=matriz.round(2).values, texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="ARI entre modelos y con el baseline RFM (1 = particiones idénticas)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo_figura(fig, 560, leyenda=False), config={"displayModeBar": False})
    st.caption(f"ARI del modelo final con el baseline RFM: {num(final['ARI_baseline_rfm'], 3)}. Es bajo en parte porque se "
               "comparan 4 segmentos con 10, pero el segmento VIP coincide casi por completo con «Campeones» + «Leales» "
               "(ver Perfiles y estrategias).")

# ------------------------------------------------------------------ Decisión
st.subheader("Modelo elegido y por qué")
with st.container(border=True):
    st.markdown(
        f"**K-Means con k = {final['k']}** sobre las 4 variables escaladas.\n\n"
        f"- **Métricas:** silhouette {num(final['silhouette'], 3)}, Davies-Bouldin {num(final['davies_bouldin'], 3)}, "
        f"Calinski-Harabasz {num(final['calinski_harabasz'])}.\n"
        f"- **Estabilidad:** ARI mediano {num(final['estabilidad_mediana'], 3)} y {num(final['estabilidad_p10'], 3)} en el peor 10 % "
        "de las submuestras.\n"
        f"- **Robustez a los empates de frecuencia:** ARI {num(final['ARI_descuantizado'], 3)}.\n"
        "- **Negocio:** 4 segmentos de tamaño útil (20–28 % de los clientes cada uno), claramente distintos y accionables.\n\n"
        "No se eligió k = 2 aunque gana en métricas internas: en una nube continua esas métricas siempre favorecen el k "
        "mínimo, y dos segmentos no permiten estrategias diferenciadas. Tampoco k = 5, por su inestabilidad. GMM quedó por "
        "detrás en todas las métricas, y DBSCAN y HDBSCAN no encuentran grupos separados por densidad."
    )
