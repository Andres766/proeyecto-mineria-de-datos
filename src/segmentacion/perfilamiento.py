"""Perfilamiento de segmentos de clientes."""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIABLES_RESUMEN: list[str] = ["recency", "frequency", "ticket_promedio", "productos_distintos", "monetary"]

# Meses de la temporada de pedidos de Navidad: septiembre–noviembre concentran el pico de
# ventas del año (notebook 01). Diciembre de 2011 está incompleto y se excluye.
MESES_TEMPORADA: tuple[int, ...] = (9, 10, 11)


def resumen_segmentos(clientes: pd.DataFrame, etiquetas: np.ndarray | pd.Series) -> pd.DataFrame:
    """Tamaño, peso en ingresos y medianas de las variables principales por segmento.

    Args:
        clientes: tabla de clientes con las columnas de ``VARIABLES_RESUMEN``.
        etiquetas: segmento de cada cliente (mismo orden que ``clientes``).

    Returns:
        DataFrame indexado por segmento con ``clientes``, ``pct_clientes``, ``pct_ingresos``,
        ``pct_una_compra`` (frequency = 1) y la mediana de cada variable, ordenado por la
        mediana de monetary.
    """
    datos = clientes.assign(segmento=np.asarray(etiquetas))
    grupos = datos.groupby("segmento")
    resumen = pd.DataFrame({
        "clientes": grupos.size(),
        "pct_clientes": 100 * grupos.size() / len(datos),
        "pct_ingresos": 100 * grupos["monetary"].sum() / datos["monetary"].sum(),
        "pct_una_compra": 100 * grupos["frequency"].apply(lambda s: (s == 1).mean()),
    })
    medianas = grupos[VARIABLES_RESUMEN].median().add_suffix("_mediana")
    return resumen.join(medianas).sort_values("monetary_mediana")


def perfil_variables(clientes: pd.DataFrame, etiquetas: np.ndarray, variables: list[str]) -> pd.DataFrame:
    """Media y mediana de cada variable por segmento (columnas MultiIndex: estadístico, variable)."""
    grupos = clientes[variables].assign(segmento=np.asarray(etiquetas)).groupby("segmento")
    return pd.concat({"media": grupos.mean(), "mediana": grupos.median()}, axis=1)


def medias_estandarizadas(datos_z: np.ndarray, etiquetas: np.ndarray, variables: list[str]) -> pd.DataFrame:
    """Media de cada variable estandarizada por segmento (datos del *snake plot*).

    0 = media de todos los clientes; +1 = una desviación típica por encima.
    """
    return pd.DataFrame(datos_z, columns=variables).assign(segmento=np.asarray(etiquetas)).groupby("segmento").mean()


def percentiles_medios(
    clientes: pd.DataFrame, etiquetas: np.ndarray, variables: list[str], invertir: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Percentil medio (0–100) de los clientes de cada segmento en cada variable (datos del radar).

    Usar percentiles pone todas las variables en la misma escala sin que los extremos dominen.
    Las variables de ``invertir`` se dan la vuelta (p. ej. recency: más alto = compró hace menos).
    """
    rangos = clientes[variables].rank(pct=True) * 100
    for var in invertir:
        rangos[var] = 100 - rangos[var]
    return rangos.assign(segmento=np.asarray(etiquetas)).groupby("segmento").mean()


def _facturas(ventas: pd.DataFrame, segmento_por_cliente: pd.Series) -> pd.DataFrame:
    """Una fila por factura con cliente, fecha, importe y segmento."""
    facturas = (ventas.groupby("InvoiceNo")
                .agg(CustomerID=("CustomerID", "first"), fecha=("InvoiceDate", "min"), importe=("Importe", "sum"))
                .reset_index())
    facturas["segmento"] = facturas["CustomerID"].map(segmento_por_cliente)
    facturas["mes"] = facturas["fecha"].dt.to_period("M")
    return facturas


def facturas_por_mes(
    ventas: pd.DataFrame, segmento_por_cliente: pd.Series, clientes_incluidos: pd.Index | None = None
) -> pd.DataFrame:
    """% de las facturas de cada segmento que cae en cada mes (filas = segmentos + "Total").

    Args:
        ventas: ventas limpias.
        segmento_por_cliente: Serie CustomerID -> segmento.
        clientes_incluidos: si se indica, solo se cuentan las facturas de esos clientes
            (p. ej. los de una sola compra).
    """
    facturas = _facturas(ventas, segmento_por_cliente)
    if clientes_incluidos is not None:
        facturas = facturas[facturas["CustomerID"].isin(clientes_incluidos)]
    tabla = pd.crosstab(facturas["segmento"], facturas["mes"], normalize="index") * 100
    tabla.loc["Total"] = facturas["mes"].value_counts(normalize=True).reindex(tabla.columns, fill_value=0) * 100
    return tabla


def pais_principal(clientes: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """País más frecuente de cada segmento, su peso y el país principal fuera de Reino Unido."""
    datos = clientes.assign(segmento=np.asarray(etiquetas))
    filas = {}
    for seg, grupo in datos.groupby("segmento"):
        cuota = grupo["pais"].value_counts(normalize=True) * 100
        fuera = cuota.drop("United Kingdom", errors="ignore")
        filas[seg] = {"pais_principal": cuota.index[0], "pct_pais_principal": cuota.iloc[0],
                      "segundo_pais": fuera.index[0] if len(fuera) else None,
                      "pct_segundo_pais": fuera.iloc[0] if len(fuera) else 0.0,
                      "paises_distintos": int(grupo["pais"].nunique())}
    return pd.DataFrame(filas).T


def productos_top(
    ventas: pd.DataFrame, segmento_por_cliente: pd.Series, n: int = 5, criterio: str = "importe"
) -> pd.DataFrame:
    """Productos principales de cada segmento.

    Args:
        criterio: ``"importe"`` (lo que más facturan) o ``"penetracion"`` (% de clientes del
            segmento que lo compraron al menos una vez).

    Returns:
        DataFrame largo con ``segmento``, ``posicion``, ``StockCode``, ``descripcion`` y ``valor``.
    """
    datos = ventas.assign(segmento=ventas["CustomerID"].map(segmento_por_cliente))
    descripcion = datos.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
    clientes_seg = segmento_por_cliente.value_counts()
    filas = []
    for seg, grupo in datos.groupby("segmento"):
        if criterio == "importe":
            valores = grupo.groupby("StockCode")["Importe"].sum()
        elif criterio == "penetracion":
            valores = 100 * grupo.groupby("StockCode")["CustomerID"].nunique() / clientes_seg[seg]
        else:
            raise ValueError(f"criterio desconocido: {criterio}")
        for posicion, (codigo, valor) in enumerate(valores.nlargest(n).items(), start=1):
            filas.append({"segmento": seg, "posicion": posicion, "StockCode": codigo,
                          "descripcion": descripcion[codigo], "valor": valor})
    return pd.DataFrame(filas)


def indicadores_segmento(
    clientes: pd.DataFrame, ventas: pd.DataFrame, grupo_por_cliente: pd.Series
) -> pd.DataFrame:
    """Indicadores por grupo que sirven de línea base de los KPI de las estrategias.

    Args:
        clientes: tabla de clientes (índice = CustomerID).
        ventas: ventas limpias.
        grupo_por_cliente: Serie CustomerID -> grupo (segmento o subgrupo). Los clientes que no
            aparecen en ella no se cuentan.

    Returns:
        DataFrame indexado por grupo. Porcentajes en 0–100; ``pct_ingresos`` es sobre el total
        de todos los clientes de ``clientes``.
    """
    datos = clientes.loc[grupo_por_cliente.index].assign(grupo=grupo_por_cliente)
    facturas = _facturas(ventas, grupo_por_cliente).dropna(subset=["segmento"])
    en_temporada = facturas["fecha"].dt.month.isin(MESES_TEMPORADA) & (facturas["fecha"].dt.year == 2011)
    clientes_temporada = set(facturas.loc[en_temporada, "CustomerID"])
    primera = facturas.groupby("CustomerID")["fecha"].min()
    captado_temporada = primera >= pd.Timestamp("2011-10-01")

    filas = {}
    for grupo, sub in datos.groupby("grupo"):
        ids = sub.index
        fac = facturas[facturas["CustomerID"].isin(ids)]
        una = sub["frequency"] == 1
        dias = sub["dias_entre_compras"].dropna()
        filas[grupo] = {
            "clientes": int(len(sub)),
            "pct_clientes": 100 * len(sub) / len(clientes),
            "ingresos": float(sub["monetary"].sum()),
            "pct_ingresos": 100 * sub["monetary"].sum() / clientes["monetary"].sum(),
            "monetary_mediano": float(sub["monetary"].median()),
            "ticket_mediano": float(sub["ticket_promedio"].median()),
            "frecuencia_mediana": float(sub["frequency"].median()),
            "pct_compra_30d": 100 * float((sub["recency"] <= 30).mean()),
            "pct_compra_90d": 100 * float((sub["recency"] <= 90).mean()),
            "pct_sin_compra_90d": 100 * float((sub["recency"] > 90).mean()),
            "pct_una_compra": 100 * float(una.mean()),
            "pct_intervalo_60d": 100 * float((dias <= 60).mean()) if len(dias) else float("nan"),
            "pct_devuelve": 100 * float((sub["tasa_devoluciones"] > 0).mean()),
            "tasa_devoluciones_media": float(sub["tasa_devoluciones"].mean()),
            "pct_clientes_compran_temporada": 100 * float(ids.isin(list(clientes_temporada)).mean()),
            "pct_facturas_temporada": 100 * float(en_temporada[fac.index].mean()) if len(fac) else float("nan"),
            "pct_una_compra_en_temporada": (100 * float(ids[una].isin(list(clientes_temporada)).mean())
                                            if una.any() else float("nan")),
            "pct_captados_temporada": 100 * float(captado_temporada.reindex(ids, fill_value=False).mean()),
            "ingresos_temporada": float(fac.loc[en_temporada[fac.index], "importe"].sum()),
        }
    return pd.DataFrame(filas).T
