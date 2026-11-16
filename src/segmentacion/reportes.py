"""Generación de los documentos del proyecto (informe, guía de sustentación y guion de demo).

Los documentos se escriben como plantillas Markdown en ``reports/plantillas/`` con marcadores
``{clave}``. Todas las cifras se calculan aquí a partir de los artefactos (``models/*.json``), los datos
procesados y, para algunas cifras del EDA, el dataset crudo. Si una plantilla usa una clave que no
existe, el renderizado falla: así ninguna cifra puede escribirse a mano por error.

Uso (desde la raíz, con src/ en PYTHONPATH):
    python -m segmentacion.reportes
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from segmentacion import config
from segmentacion.features import VARIABLES_CLUSTERING, calcular_vif, construir_preprocesador

PLANTILLAS = config.RAIZ / "reports" / "plantillas"
# Plantilla -> destino (el README va en la raíz del repositorio).
DOCUMENTOS = {
    "INFORME.md": config.RAIZ / "reports" / "INFORME.md",
    "GUIA_SUSTENTACION.md": config.RAIZ / "reports" / "GUIA_SUSTENTACION.md",
    "GUION_DEMO.md": config.RAIZ / "reports" / "GUION_DEMO.md",
    "README.md": config.RAIZ / "README.md",
}


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------
def num(valor: float, decimales: int = 0) -> str:
    """Número con formato español (punto de miles, coma decimal)."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(valor: float, decimales: int = 1) -> str:
    """Porcentaje con formato español (p. ej. «24,1 %»)."""
    return f"{num(valor, decimales)} %"


def libras(valor: float, decimales: int = 0) -> str:
    """Importe en libras con formato español (p. ej. «3.098,15 £»)."""
    return f"{num(valor, decimales)} £"


MESES = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


def fecha_larga(fecha: pd.Timestamp) -> str:
    """Fecha en texto (p. ej. «10 de diciembre de 2011»)."""
    return f"{fecha.day} de {MESES[fecha.month]} de {fecha.year}"


def tabla_markdown(df: pd.DataFrame) -> str:
    """DataFrame (ya formateado como texto) a tabla Markdown."""
    columnas = [str(c) for c in df.columns]
    filas = ["| " + " | ".join(columnas) + " |", "|" + "|".join("---" for _ in columnas) + "|"]
    filas += ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(filas)


# ---------------------------------------------------------------------------
# Cifras
# ---------------------------------------------------------------------------
def _cifras_eda(c: dict) -> None:
    bitacora = pd.read_csv(config.CSV_BITACORA)
    ventas = pd.read_parquet(config.PARQUET_LIMPIO)
    devoluciones = pd.read_parquet(config.PARQUET_DEVOLUCIONES)
    paso = {int(p.split(".")[0]): fila for p, fila in zip(bitacora["paso"], bitacora.itertuples())}

    c["filas_originales"] = num(bitacora["filas_antes"].iloc[0])
    c["filas_limpias"] = num(bitacora["filas_despues"].iloc[-1])
    c["duplicados"] = num(paso[1].filas_eliminadas)
    c["filas_sin_cliente"] = num(paso[2].filas_eliminadas)
    c["pct_paso_sin_cliente"] = pct(paso[2].pct_eliminado, 2)
    c["filas_no_producto"] = num(paso[3].filas_eliminadas)
    c["cancelaciones"] = num(paso[4].filas_eliminadas)
    c["filas_no_positivas"] = num(paso[5].filas_eliminadas)
    c["compras_anuladas"] = num(paso[6].filas_eliminadas)
    exactas = int(devoluciones["EmparejaCompra"].sum())
    c["anuladas_exactas"] = num(exactas)
    c["anuladas_manuales"] = num(paso[6].filas_eliminadas - exactas)
    tabla = bitacora[["paso", "filas_antes", "filas_despues", "filas_eliminadas", "pct_eliminado"]].copy()
    tabla.columns = ["Paso", "Filas antes", "Filas después", "Eliminadas", "% eliminado"]
    for col in ["Filas antes", "Filas después", "Eliminadas"]:
        tabla[col] = tabla[col].map(num)
    tabla["% eliminado"] = tabla["% eliminado"].map(lambda v: pct(v, 2))
    tabla["Paso"] = tabla["Paso"].str.replace("<", "&lt;")
    c["tabla_bitacora"] = tabla_markdown(tabla)

    c["facturas"] = num(ventas["InvoiceNo"].nunique())
    c["clientes"] = num(ventas["CustomerID"].nunique())
    c["productos"] = num(ventas["StockCode"].nunique())
    c["paises"] = num(ventas["Country"].nunique())
    c["ingresos"] = libras(ventas["Importe"].sum(), 2)
    c["fecha_inicio"] = fecha_larga(ventas["InvoiceDate"].min())
    c["fecha_fin"] = fecha_larga(ventas["InvoiceDate"].max())

    mensual = ventas.set_index("InvoiceDate").resample("MS")["Importe"].sum()
    pico = mensual.idxmax()
    c["mes_pico"] = f"{MESES[pico.month]} de {pico.year}"
    c["ingresos_mes_pico"] = libras(mensual.max())
    facturas = ventas.drop_duplicates("InvoiceNo")
    c["facturas_sabado"] = num((facturas["InvoiceDate"].dt.day_of_week == 5).sum())
    c["hora_pico"] = str(facturas["InvoiceDate"].dt.hour.value_counts().idxmax())
    por_pais = ventas.groupby("Country").agg(ingresos=("Importe", "sum"), clientes=("CustomerID", "nunique"))
    c["pct_clientes_uk"] = pct(100 * por_pais.loc["United Kingdom", "clientes"] / ventas["CustomerID"].nunique())
    c["pct_ingresos_uk"] = pct(100 * por_pais.loc["United Kingdom", "ingresos"] / por_pais["ingresos"].sum())
    gasto = ventas.groupby("CustomerID")["Importe"].sum().sort_values(ascending=False)
    uno = int(np.ceil(len(gasto) * 0.01))
    c["pct_ingresos_top1"] = pct(100 * gasto.iloc[:uno].sum() / gasto.sum())

    # Cifras sobre el dataset original (antes de limpiar).
    if config.PARQUET_RAW.exists():
        crudo = pd.read_parquet(config.PARQUET_RAW)
        importe = crudo["Quantity"] * crudo["UnitPrice"]
        cancel = crudo["InvoiceNo"].str.startswith("C")
        positivas = ~cancel & (importe > 0)
        sin_cliente = crudo["CustomerID"].isna()
        c["pct_lineas_sin_cliente"] = pct(100 * sin_cliente.mean(), 2)
        c["pct_importe_sin_cliente"] = pct(100 * importe[positivas & sin_cliente].sum() / importe[positivas].sum(), 2)
        c["pct_importe_cancelado"] = pct(100 * -importe[cancel].sum() / importe[positivas].sum())
        c["max_cantidad_anulada"] = num(crudo["Quantity"].max())
    else:
        raise FileNotFoundError("Falta data/raw/online_retail.parquet: ejecuta python -m segmentacion.carga")


def _cifras_features(c: dict) -> None:
    clientes = pd.read_parquet(config.PARQUET_SEGMENTADOS)
    features = json.loads(config.JSON_FEATURES.read_text(encoding="utf-8"))
    c["fecha_referencia"] = fecha_larga(pd.Timestamp(features["fecha_referencia"]))
    una = clientes["frequency"].eq(1)
    c["clientes_una_compra"] = num(una.sum())
    c["pct_una_compra"] = pct(100 * una.mean())
    c["mediana_dias_entre_compras"] = num(clientes["dias_entre_compras"].median(), 2)
    c["corr_dias_frequency"] = num(clientes.loc[~una, ["dias_entre_compras", "frequency"]].corr("spearman").iloc[0, 1], 3)

    numericas = clientes[["recency", "frequency", "monetary", "ticket_promedio", "productos_distintos", "antiguedad",
                          "unidades_por_factura", "tasa_devoluciones"]]
    vif_todas = calcular_vif(np.log1p(numericas))
    c["vif_monetary"] = num(vif_todas["monetary"])
    c["vif_frequency"] = num(vif_todas["frequency"])
    c["vif_ticket"] = num(vif_todas["ticket_promedio"])
    c["vif_max_final"] = num(calcular_vif(np.log1p(clientes[VARIABLES_CLUSTERING])).max(), 3)
    corr = numericas.corr("spearman")
    c["corr_frequency_monetary"] = num(corr.loc["frequency", "monetary"], 3)
    c["corr_ticket_unidades"] = num(corr.loc["ticket_promedio", "unidades_por_factura"], 3)
    c["corr_max_final"] = num(clientes[VARIABLES_CLUSTERING].corr("spearman").where(
        np.triu(np.ones((4, 4), dtype=bool), k=1)).abs().max().max(), 3)

    tasa = clientes["tasa_devoluciones"]
    c["clientes_con_devolucion"] = num((tasa > 0).sum())
    c["pct_tasa_cero"] = pct(100 * (tasa == 0).mean())
    c["tasa_maxima"] = num(tasa.max(), 4)

    zn = construir_preprocesador(clientes[VARIABLES_CLUSTERING], winsorizar=False).transform(clientes[VARIABLES_CLUSTERING])
    extremo = (np.abs(zn) > 3).any(axis=1)
    c["clientes_extremos"] = num(extremo.sum())
    c["pct_ingresos_extremos"] = pct(100 * clientes.loc[extremo, "monetary"].sum() / clientes["monetary"].sum())
    pre = joblib.load(config.PREPROCESADOR)
    bajo, alto = pre.named_steps["winsor"].kw_args["a_min"], pre.named_steps["winsor"].kw_args["a_max"]
    en_log = np.log1p(clientes[VARIABLES_CLUSTERING].to_numpy(dtype=float))
    recortados = ((en_log > alto) | (en_log < bajo)).any(axis=1)
    c["clientes_recortados"] = num(recortados.sum())
    c["pct_recortados"] = pct(100 * recortados.mean())

    # Comparaciones de preprocesamiento (mismo K-Means de prueba que el notebook 02).
    from sklearn.preprocessing import RobustScaler, StandardScaler

    from segmentacion.evaluacion import diagnostico_kmeans

    datos_log = np.log1p(clientes[VARIABLES_CLUSTERING])
    estandar, _ = diagnostico_kmeans(StandardScaler().fit_transform(datos_log), [3, 4, 5, 6])
    robusto, _ = diagnostico_kmeans(RobustScaler().fit_transform(datos_log), [3, 4, 5, 6])
    c["dif_max_silhouette_escaladores"] = num((robusto["silhouette"] - estandar["silhouette"]).max(), 3)
    sin_tasa, et_sin = diagnostico_kmeans(pre.transform(clientes[VARIABLES_CLUSTERING]), [3, 4])
    columnas_tasa = VARIABLES_CLUSTERING + ["tasa_devoluciones"]
    con_tasa, et_con = diagnostico_kmeans(construir_preprocesador(clientes[columnas_tasa]).transform(clientes[columnas_tasa]), [3, 4])
    c["silhouette_sin_tasa_k3"] = num(sin_tasa.loc[0, "silhouette"], 3)
    c["silhouette_con_tasa_k3"] = num(con_tasa.loc[0, "silhouette"], 3)
    devuelve = (tasa > 0).to_numpy()
    tam_100 = [int((et_con[4] == g).sum()) for g in np.unique(et_con[4]) if devuelve[et_con[4] == g].mean() == 1]
    c["cluster_devolvedores_k4"] = num(tam_100[0]) if tam_100 else "0"
    c["pct_cluster_devolvedores_k4"] = pct(100 * tam_100[0] / len(clientes)) if tam_100 else "0 %"


def _cifras_modelos(c: dict) -> None:
    metricas = json.loads(config.JSON_METRICAS.read_text(encoding="utf-8"))
    final = metricas["modelo_final"]
    comp = {f["modelo"]: f for f in metricas["comparativa"]}
    clientes = pd.read_parquet(config.PARQUET_SEGMENTADOS)

    c["k_final"] = str(final["k"])
    for clave in ["silhouette", "davies_bouldin", "estabilidad_mediana", "estabilidad_p10", "ARI_descuantizado", "ARI_baseline_rfm"]:
        c[f"final_{clave.lower()}"] = num(final[clave], 3)
    c["final_calinski_harabasz"] = num(final["calinski_harabasz"])
    c["final_pct_menor"] = pct(comp[f"K-Means k={final['k']}"]["pct_segmento_menor"])
    for nombre, prefijo in [("K-Means k=2", "k2"), ("K-Means k=5", "k5"), ("K-Means k=6", "k6")]:
        for clave in ["silhouette", "davies_bouldin", "estabilidad_mediana", "estabilidad_p10"]:
            c[f"{prefijo}_{clave}"] = num(comp[nombre][clave], 3)
        c[f"{prefijo}_pct_menor"] = pct(comp[nombre]["pct_segmento_menor"])
    gmm_bic = next(v for k, v in comp.items() if "(BIC)" in k)
    c["gmm_silhouette"] = num(gmm_bic["silhouette"], 3)
    c["gmm_estabilidad_mediana"] = num(gmm_bic["estabilidad_mediana"], 3)
    c["gmm_estabilidad_p10"] = num(gmm_bic["estabilidad_p10"], 3)
    c["gmm_pct_menor"] = pct(gmm_bic["pct_segmento_menor"])
    c["hdbscan_ari_descuantizado"] = num(comp["HDBSCAN"]["ARI_descuantizado"], 3)
    c["hdbscan_pct_ruido"] = pct(comp["HDBSCAN"]["pct_ruido"])

    tabla = pd.DataFrame(metricas["comparativa"]).set_index("modelo")
    tabla = tabla[["n_clusters", "silhouette", "davies_bouldin", "estabilidad_mediana", "estabilidad_p10", "ARI_descuantizado", "pct_ruido"]]
    tabla.columns = ["Clusters", "Silhouette ↑", "Davies-Bouldin ↓", "Estabilidad mediana ↑", "Estabilidad p10 ↑",
                     "Robustez a empates ↑", "% ruido"]
    formateada = tabla.apply(lambda col: col.map(lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v))
                                                 else (num(v) if col.name == "Clusters" else
                                                       (pct(v) if col.name == "% ruido" else num(v, 3)))))
    c["tabla_modelos"] = tabla_markdown(formateada.reset_index().rename(columns={"modelo": "Modelo"}))

    barrido = pd.DataFrame(metricas["barrido_kmeans"])
    c["estabilidad_mediana_min_k2_6"] = num(barrido.loc[barrido["k"] <= 6, "estabilidad_mediana"].min(), 3)

    gmm = metricas["gmm"]
    c["reg_covar"] = num(gmm["reg_covar"], 2)
    c["gmm_tipo"] = gmm["tipo"]
    c["gmm_k"] = str(gmm["k"])
    c["bic_degenerado"] = num(gmm["bic_mejor_reg_defecto"])
    c["bic_elegido"] = num(min(f["bic"] for f in gmm["rejilla_bic"]))

    db = metricas["dbscan"]
    c["eps"] = num(db["eps"], 3)
    c["min_samples"] = str(db["min_samples"])
    ruido = clientes["dbscan"].eq(-1)
    c["atipicos"] = num(ruido.sum())
    c["pct_atipicos"] = pct(100 * ruido.mean())
    c["pct_ingresos_atipicos"] = pct(100 * clientes.loc[ruido, "monetary"].sum() / clientes["monetary"].sum())

    zn = joblib.load(config.PREPROCESADOR_SIN_WINSOR).transform(clientes[VARIABLES_CLUSTERING])
    niveles = np.unique(zn[:, VARIABLES_CLUSTERING.index("frequency")])
    c["hueco_frequency"] = num(niveles[1] - niveles[0], 3)
    from segmentacion.modelado import distancias_k_vecinos

    kdist = distancias_k_vecinos(zn, db["min_samples"])
    una = clientes["frequency"].eq(1).to_numpy()
    c["kdist_f1"] = num(np.median(kdist[una]), 3)
    c["kdist_resto"] = num(np.median(kdist[~una]), 3)

    pca = metricas["pca"]
    c["pca_pc1"] = pct(100 * pca["varianza_explicada"][0])
    c["pca_pc2"] = pct(100 * pca["varianza_explicada"][1])
    c["pca_2"] = pct(100 * sum(pca["varianza_explicada"][:2]))

    # Dos óptimos casi empatados de K-Means con k = 5 (60 inicializaciones únicas, como en el notebook 03).
    from sklearn.cluster import KMeans

    z = joblib.load(config.PREPROCESADOR).transform(clientes[VARIABLES_CLUSTERING])
    # Se agrupan por inercia entera: diferencias decimales son la misma solución con otro orden de cálculo.
    inercias = np.unique(np.round([KMeans(5, n_init=1, random_state=s).fit(z).inertia_ for s in range(60)]))
    alternativas = 100 * (inercias[1:] / inercias[0] - 1)
    c["k5_inercia_mejor"] = num(inercias[0])
    c["k5_dif_min"] = pct(alternativas.min(), 2)
    c["k5_dif_max"] = pct(alternativas.max(), 2)

    # t-SNE: dispersión real frente a la del mapa (perplexity guardada).
    seg = clientes["segmento"].to_numpy()
    tsne = clientes[["tsne_x", "tsne_y"]].to_numpy()
    disp_real = [np.linalg.norm(z[seg == s] - z[seg == s].mean(0), axis=1).mean() for s in range(final["k"])]
    disp_tsne = [np.linalg.norm(tsne[seg == s] - tsne[seg == s].mean(0), axis=1).mean() for s in range(final["k"])]
    c["tsne_mas_disperso_real"] = str(int(np.argmax(disp_real)))
    c["tsne_mas_compacto_mapa"] = str(int(np.argmin(disp_tsne)))


def _cifras_segmentos(c: dict) -> None:
    datos = json.loads(config.JSON_SEGMENTOS.read_text(encoding="utf-8"))
    riesgo = ["Necesitan atención", "A punto de dormir", "En riesgo", "No se pueden perder", "Hibernando"]
    filas = []
    for g in datos["segmentos"]:
        s, ind, med = g["id"], g["indicadores"], g["medianas"]
        c[f"s{s}_nombre"] = g["nombre"]
        c[f"s{s}_pct_clientes"] = pct(ind["pct_clientes"])
        c[f"s{s}_pct_ingresos"] = pct(ind["pct_ingresos"])
        c[f"s{s}_clientes"] = num(ind["clientes"])
        c[f"s{s}_ticket"] = libras(ind["ticket_mediano"], 2)
        c[f"s{s}_gasto"] = libras(ind["monetary_mediano"], 2)
        c[f"s{s}_recency"] = num(med["recency"])
        c[f"s{s}_frequency"] = num(med["frequency"])
        c[f"s{s}_productos"] = num(med["productos_distintos"])
        c[f"s{s}_pct_una_compra"] = pct(ind["pct_una_compra"])
        c[f"s{s}_pct_compra_90d"] = pct(ind["pct_compra_90d"])
        c[f"s{s}_pct_devuelve"] = pct(ind["pct_devuelve"])
        c[f"s{s}_pct_temporada"] = pct(ind["pct_facturas_temporada"])
        c[f"s{s}_rfm_riesgo"] = pct(sum(g["composicion_rfm_pct"][r] for r in riesgo), 0)
        c[f"s{s}_rfm_valor"] = pct(g["composicion_rfm_pct"]["Campeones"] + g["composicion_rfm_pct"]["Leales"])
        valor = ind["pct_una_compra_en_temporada"]
        c[f"s{s}_una_compra_temporada"] = "—" if valor is None or np.isnan(valor) else pct(valor)
        c[f"s{s}_captados_temporada"] = pct(ind["pct_captados_temporada"])
        filas.append([g["nombre"], num(ind["clientes"]), pct(ind["pct_clientes"]), pct(ind["pct_ingresos"]),
                      num(med["recency"]), num(med["frequency"]), libras(ind["ticket_mediano"], 2),
                      num(med["productos_distintos"]), libras(ind["monetary_mediano"], 2)])
    ck = datos["cuentas_clave"]
    ic = ck["indicadores"]
    c["ck_nombre_subgrupo"] = ck["nombre"].lower()
    c["ck_clientes"] = num(ic["clientes"])
    c["ck_pct_clientes"] = pct(ic["pct_clientes"])
    c["ck_pct_ingresos"] = pct(ic["pct_ingresos"])
    c["ck_frequency"] = num(ic["frecuencia_mediana"])
    c["ck_ticket"] = libras(ic["ticket_mediano"], 2)
    c["ck_gasto"] = libras(ic["monetary_mediano"], 2)
    c["ck_umbral"] = libras(ck["umbral_gasto"], 2)
    c["ck_pct_devuelve"] = pct(ic["pct_devuelve"])
    filas.append([f"↳ {ck['nombre']} (subgrupo de VIP)", num(ic["clientes"]), pct(ic["pct_clientes"]), pct(ic["pct_ingresos"]),
                  num(ck["medianas"]["recency"]), num(ck["medianas"]["frequency"]), libras(ic["ticket_mediano"], 2),
                  num(ck["medianas"]["productos_distintos"]), libras(ic["monetary_mediano"], 2)])
    c["tabla_segmentos"] = tabla_markdown(pd.DataFrame(filas, columns=[
        "Segmento", "Clientes", "% clientes", "% ingresos", "Recency (días)", "Facturas", "Ticket mediano",
        "Productos", "Gasto mediano"]))

    formato_base = {"ticket_mediano": lambda v: libras(v, 2), "ingresos_temporada": lambda v: libras(v, 2)}
    lineas = []
    for g in [*datos["segmentos"], {**ck, "nombre": f"{ck['nombre']} (VIP)"}]:
        lineas.append(f"**{g['nombre']}**\n")
        for i, e in enumerate(g["estrategias"], start=1):
            base = formato_base.get(e["indicador"], pct)(e["linea_base"])
            lineas.append(f"{i}. **Acción:** {e['accion']} **Momento:** {e['momento']} **KPI:** {e['kpi']} "
                          f"**Línea base actual:** {base}.")
        lineas.append("")
    c["lista_estrategias"] = "\n".join(lineas).strip()


def _cifras_pruebas(c: dict) -> None:
    salida = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True, text=True,
                            cwd=config.RAIZ)
    recogidas = re.search(r"(\d+) tests? collected", salida.stdout)
    if not recogidas:
        raise RuntimeError(f"No se pudo contar las pruebas:\n{salida.stdout}\n{salida.stderr}")
    c["n_pruebas"] = recogidas.group(1)


def calcular_cifras() -> dict[str, str]:
    """Calcula todas las cifras que usan las plantillas (como texto ya formateado)."""
    cifras: dict[str, str] = {}
    for paso in (_cifras_eda, _cifras_features, _cifras_modelos, _cifras_segmentos, _cifras_pruebas):
        paso(cifras)
    return cifras


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------
class _CifrasEstrictas(dict):
    def __missing__(self, clave: str) -> str:
        raise KeyError(f"La plantilla usa la cifra «{clave}», que no está calculada en reportes.py")


def numerar_figuras_y_tablas(texto: str) -> str:
    """Numera figuras (``![pie](ruta)``) y tablas (línea ``Tabla: pie`` antes de la tabla)."""
    salida, figura, tabla = [], 0, 0
    for linea in texto.splitlines():
        imagen = re.fullmatch(r"!\[(.+?)\]\((.+?)\)", linea.strip())
        rotulo = re.fullmatch(r"Tabla:\s*(.+)", linea.strip())
        if imagen:
            figura += 1
            salida += [f"![Figura {figura}. {imagen.group(1)}]({imagen.group(2)})", "",
                       f"*Figura {figura}. {imagen.group(1)}*"]
        elif rotulo:
            tabla += 1
            salida.append(f"*Tabla {tabla}. {rotulo.group(1)}*")
        else:
            salida.append(linea)
    return "\n".join(salida) + "\n"


def renderizar(plantilla: str, cifras: dict[str, str]) -> str:
    """Sustituye los marcadores ``{clave}`` y numera figuras y tablas."""
    return numerar_figuras_y_tablas(plantilla.format_map(_CifrasEstrictas(cifras)))


def generar() -> list[str]:
    """Renderiza todas las plantillas en ``reports/``. Devuelve los nombres generados."""
    cifras = calcular_cifras()
    generados = []
    for nombre, destino in DOCUMENTOS.items():
        texto = (PLANTILLAS / nombre).read_text(encoding="utf-8")
        destino.write_text(renderizar(texto, cifras), encoding="utf-8")
        generados.append(str(destino.relative_to(config.RAIZ)))
    return generados


if __name__ == "__main__":
    for nombre in generar():
        print(f"Generado {nombre}")
