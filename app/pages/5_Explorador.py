"""Explorador: filtros, búsqueda por ID, descarga CSV y clasificación de un cliente nuevo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from comun import (COLOR_POR_NOMBRE, ETIQUETAS, NOMBRE_CUENTAS_CLAVE,
                   NOMBRES, ORDEN_NOMBRES, artefactos, cargar_clientes, cargar_json, configurar_pagina,
                   etiqueta_segmento, estilo_figura, libras, mostrar_estrategias, num, pct)
from segmentacion.prediccion import clasificar_cliente

configurar_pagina("Explorador", "🧭")
st.title("Explorador de clientes")

clientes = cargar_clientes()
segmentos = {g["id"]: g for g in cargar_json("segmentos")["segmentos"]}
cuentas = cargar_json("segmentos")["cuentas_clave"]
medianas = cargar_json("features")["medianas_variables_clustering"]
VARS = ["recency", "frequency", "ticket_promedio", "productos_distintos", "monetary"]


def comparar_con_segmento(valores: dict, segmento: int) -> pd.DataFrame:
    """Tabla cliente frente a la mediana de su segmento, ya formateada en español."""
    enteras = {"recency", "frequency", "productos_distintos"}
    filas = []
    for v in VARS:
        mediana = segmentos[segmento]["medianas"][v]
        decimales = 0 if v in enteras else 2
        filas.append({"Variable": ETIQUETAS[v], "Cliente": num(valores[v], decimales),
                      "Mediana del segmento": num(mediana, decimales),
                      "vs. mediana": "▲ mayor" if valores[v] > mediana else ("▼ menor" if valores[v] < mediana else "= igual")})
    return pd.DataFrame(filas)


def mapa_pca(punto: tuple[float, float] | None = None, titulo: str = "") -> go.Figure:
    """Clientes en el plano PCA coloreados por segmento; opcionalmente, un punto destacado."""
    fig = go.Figure()
    for nombre in ORDEN_NOMBRES:
        sub = clientes[clientes["nombre_segmento"] == nombre]
        fig.add_trace(go.Scattergl(x=sub["pc1"], y=sub["pc2"], mode="markers", name=nombre, hoverinfo="skip",
                                   marker=dict(color=COLOR_POR_NOMBRE[nombre], size=5, opacity=0.35)))
    if punto is not None:
        fig.add_trace(go.Scatter(x=[punto[0]], y=[punto[1]], mode="markers+text", name="Cliente nuevo", text=["Cliente nuevo"],
                                 textposition="top center", textfont=dict(color="#0b0b0b", size=13),
                                 marker=dict(symbol="star", size=24, color="#ffffff", line=dict(color="#0b0b0b", width=2.5)),
                                 hovertemplate="Cliente nuevo<br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>"))
    fig.update_layout(title=titulo, legend=dict(itemsizing="constant"))
    fig.update_xaxes(title="PC1 · actividad / valor")
    fig.update_yaxes(title="PC2 · tamaño del pedido")
    return estilo_figura(fig, 480)


tab_buscar, tab_nuevo = st.tabs(["🔍 Buscar y filtrar clientes", "➕ Clasificar un cliente nuevo"])

# =================================================================== Buscar y filtrar
with tab_buscar:
    c1, c2, c3 = st.columns([2, 2, 1])
    elegidos = c1.multiselect("Segmento", ORDEN_NOMBRES, default=ORDEN_NOMBRES)
    paises = sorted(clientes["pais"].unique())
    elegidos_pais = c2.multiselect("País", paises, placeholder="Todos los países")
    solo_ck = c3.toggle(f"Solo {NOMBRE_CUENTAS_CLAVE.lower()}")

    filtro = clientes["nombre_segmento"].isin(elegidos)
    if elegidos_pais:
        filtro &= clientes["pais"].isin(elegidos_pais)
    if solo_ck:
        filtro &= clientes["cuenta_clave"]
    vista = clientes.loc[filtro, ["nombre_segmento", "cuenta_clave", "pais", "recency", "frequency", "ticket_promedio",
                                  "productos_distintos", "monetary", "tasa_devoluciones"]]

    if vista.empty:
        st.warning("Ningún cliente cumple los filtros elegidos. Prueba a añadir segmentos o países.")
    else:
        a, b, c = st.columns(3)
        a.metric("Clientes filtrados", num(len(vista)))
        b.metric("Ingresos de la selección", libras(vista["monetary"].sum()))
        c.metric("% de los ingresos totales", pct(100 * vista["monetary"].sum() / clientes["monetary"].sum()))
        st.dataframe(vista.reset_index(), hide_index=True, height=360, column_config={
            "CustomerID": st.column_config.NumberColumn("Cliente", format="%d"),
            "nombre_segmento": "Segmento", "cuenta_clave": st.column_config.CheckboxColumn(NOMBRE_CUENTAS_CLAVE),
            "pais": "País", "recency": "Días desde la última compra", "frequency": "Facturas",
            "ticket_promedio": st.column_config.NumberColumn("Ticket promedio (£)", format="%.2f"),
            "productos_distintos": "Productos distintos",
            "monetary": st.column_config.NumberColumn("Gasto total (£)", format="%.2f"),
            "tasa_devoluciones": st.column_config.NumberColumn("Tasa de devoluciones", format="%.3f")})
        st.download_button("⬇️ Descargar selección en CSV", vista.reset_index().to_csv(index=False).encode("utf-8"),
                           file_name="clientes_segmentados.csv", mime="text/csv")

    st.divider()
    st.markdown("#### Buscar un cliente por su ID")
    texto_id = st.text_input("CustomerID", placeholder="Por ejemplo, 12347", max_chars=10)
    if texto_id.strip():
        if not texto_id.strip().isdigit():
            st.error("El ID de cliente debe ser un número entero (por ejemplo, 12347).")
        elif int(texto_id) not in clientes.index:
            st.error(f"No existe ningún cliente con ID {texto_id.strip()} en los datos analizados. Recuerda que se "
                     "excluyeron las compras sin cliente identificado y los clientes cuyas únicas líneas eran "
                     "cancelaciones o cargos.")
        else:
            c = clientes.loc[int(texto_id)]
            seg = int(c["segmento"])
            st.markdown(f"Cliente **{texto_id.strip()}** · {etiqueta_segmento(seg)}"
                        + (f" · **{NOMBRE_CUENTAS_CLAVE}**" if c["cuenta_clave"] else "") + f" · {c['pais']}",
                        unsafe_allow_html=True)
            izq, der = st.columns([3, 2])
            izq.dataframe(comparar_con_segmento(c.to_dict(), seg), hide_index=True)
            der.markdown(f"<span class='texto-secundario'>{segmentos[seg]['descripcion']}</span>", unsafe_allow_html=True)
            if c["dbscan"] == -1:
                der.info("DBSCAN marca a este cliente como **atípico** (alejado de la masa principal de clientes).")

# =================================================================== Cliente nuevo
with tab_nuevo:
    st.markdown(
        "Introduce los datos de un cliente para saber a qué segmento pertenecería. El modelo **no se reentrena**: se aplica "
        "el mismo preprocesamiento del entrenamiento (log1p, recorte a los percentiles 1–99 y estandarización) y el "
        "K-Means guardado. Los valores iniciales son la **mediana** de todos los clientes."
    )
    with st.form("cliente_nuevo"):
        c1, c2, c3, c4 = st.columns(4)
        recency = c1.number_input(ETIQUETAS["recency"], min_value=1, value=int(medianas["recency"]), step=1,
                                  help="Días entre la última compra y la fecha de referencia (1 = ayer).")
        frequency = c2.number_input(ETIQUETAS["frequency"], min_value=1, value=int(medianas["frequency"]), step=1,
                                    help="Número de facturas (pedidos) distintas.")
        ticket = c3.number_input(ETIQUETAS["ticket_promedio"], min_value=0.01, value=round(float(medianas["ticket_promedio"]), 2),
                                 step=10.0, format="%.2f", help="Gasto medio por factura.")
        productos = c4.number_input(ETIQUETAS["productos_distintos"], min_value=1, value=int(medianas["productos_distintos"]),
                                    step=1, help="Número de productos (StockCode) distintos comprados.")
        enviado = st.form_submit_button("Clasificar cliente", type="primary")

    if enviado:
        try:
            r = clasificar_cliente(artefactos(), recency, frequency, ticket, productos)
        except ValueError as error:
            st.error(f"No se puede clasificar: {error}")
            r = None
        except Exception:  # noqa: BLE001 - cualquier fallo inesperado se informa sin romper la página
            st.error("Ocurrió un error inesperado al clasificar. Revisa los valores e inténtalo de nuevo.")
            r = None

        if r is not None:
            seg = r["segmento"]
            st.markdown(f"### Segmento asignado: {etiqueta_segmento(seg)}", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            a.metric("Gasto total estimado", libras(r["monetary"], 2), help="Facturas × ticket promedio.")
            b.metric("Atípico para DBSCAN", "Sí" if r["atipico_dbscan"] else "No",
                     help=f"Distancia al punto núcleo más cercano: {num(r['distancia_nucleo'], 3)} (atípico si supera eps).")
            c.metric(NOMBRE_CUENTAS_CLAVE, "Sí" if r["cuenta_clave"] else "No")
            st.caption(r["motivo_cuenta_clave"] + f" Umbral de gasto: {libras(cuentas['umbral_gasto'], 2)}.")

            if r["recortadas"]:
                detalle = "; ".join(f"«{ETIQUETAS[v]}» se acotó al {d['limite']} del entrenamiento ({num(d['valor_usado'], 2)})"
                                    for v, d in r["recortadas"].items())
                st.warning(f"Algunos valores están fuera del rango de los clientes de entrenamiento y el modelo los acota, "
                           f"igual que hizo al entrenar: {detalle}.", icon="✂️")

            izq, der = st.columns([1.25, 1])
            izq.plotly_chart(mapa_pca((r["pca"][0], r["pca"][1]), "El cliente nuevo sobre el mapa PCA de los segmentos"),
                             config={"displayModeBar": False})
            with der:
                st.markdown("**Comparación con la mediana del segmento**")
                st.dataframe(comparar_con_segmento({**r["valores"], "monetary": r["monetary"]}, seg), hide_index=True)
                st.markdown(f"<span class='texto-secundario'>{segmentos[seg]['descripcion']}</span>", unsafe_allow_html=True)

            st.markdown(f"#### Estrategias recomendadas para {NOMBRES[seg]}")
            mostrar_estrategias(segmentos[seg]["estrategias"])
            if r["cuenta_clave"]:
                st.markdown(f"#### Además, como {NOMBRE_CUENTAS_CLAVE.lower()}")
                mostrar_estrategias(cuentas["estrategias"])

            with st.expander("Ver el preprocesamiento aplicado (idéntico al del entrenamiento)"):
                a_min, a_max = artefactos().limites_log
                pasos = pd.DataFrame({
                    "Valor introducido": [r["valores"][v] for v in r["z"]],
                    "log1p": [np.log1p(r["valores"][v]) for v in r["z"]],
                    "Límite p1 (log1p)": a_min, "Límite p99 (log1p)": a_max,
                    "Tras el recorte": [float(np.clip(np.log1p(r["valores"][v]), lo, hi)) for v, lo, hi in zip(r["z"], a_min, a_max)],
                    "Estandarizado (z)": list(r["z"].values()),
                }, index=[ETIQUETAS[v] for v in r["z"]])
                st.dataframe(pasos.style.format("{:,.3f}", decimal=",", thousands="."))
                st.caption("Los límites del recorte son los percentiles 1 y 99 calculados con los clientes de entrenamiento "
                           "y guardados en el preprocesador; no se recalculan con el cliente nuevo.")
    else:
        st.plotly_chart(mapa_pca(titulo="Mapa PCA de los segmentos (pulsa «Clasificar cliente» para situar uno nuevo)"),
                        config={"displayModeBar": False})
