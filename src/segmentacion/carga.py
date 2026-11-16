"""Descarga del dataset Online Retail y conversión a parquet.

Uso desde la raíz del proyecto:
    python -m segmentacion.carga        (con src/ en PYTHONPATH)

Estrategia de descarga:
    1. Zip oficial de UCI.
    2. Si falla, paquete ``ucimlrepo`` (id=352).
    3. Si ambos fallan, se indica dónde colocar el archivo manualmente.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from segmentacion import config

# Tipos esperados. InvoiceNo y StockCode mezclan números y texto
# (p. ej. "C536379", "85123A"), por eso se fuerzan a texto.
COLUMNAS_TEXTO: list[str] = ["InvoiceNo", "StockCode", "Description", "Country"]


def _descargar_zip_uci() -> pd.DataFrame:
    """Descarga el zip oficial de UCI, lo guarda en data/raw y lee el .xlsx."""
    import requests

    respuesta = requests.get(config.URL_DATASET, timeout=120)
    respuesta.raise_for_status()
    config.ZIP_RAW.write_bytes(respuesta.content)
    with zipfile.ZipFile(io.BytesIO(respuesta.content)) as zf:
        nombre_xlsx = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        zf.extract(nombre_xlsx, config.DATA_RAW)
        extraido = config.DATA_RAW / nombre_xlsx
        if extraido != config.XLSX_RAW:
            extraido.replace(config.XLSX_RAW)
    return _leer_xlsx()


def _leer_xlsx() -> pd.DataFrame:
    """Lee el .xlsx original forzando los tipos de las columnas de texto."""
    return pd.read_excel(
        config.XLSX_RAW,
        dtype={c: str for c in COLUMNAS_TEXTO},
        engine="openpyxl",
    )


def _descargar_ucimlrepo() -> pd.DataFrame:
    """Alternativa: descarga mediante el paquete oficial ``ucimlrepo``."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=config.UCI_ID)
    df = dataset.data.original
    if df is None or df.empty:
        raise RuntimeError("ucimlrepo no devolvió datos para id=352")
    return df


def normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica tipos consistentes a las 8 columnas del dataset.

    Args:
        df: DataFrame crudo con las columnas originales de UCI.

    Returns:
        Copia con texto limpio (sin espacios laterales), fechas como datetime,
        CustomerID como entero nullable y cantidades/precios numéricos.
    """
    df = df.copy()
    for col in COLUMNAS_TEXTO:
        df[col] = df[col].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = df["Quantity"].astype("int64")
    df["UnitPrice"] = df["UnitPrice"].astype("float64")
    df["CustomerID"] = df["CustomerID"].astype("Int64")
    return df


def obtener_datos_crudos(forzar: bool = False) -> pd.DataFrame:
    """Devuelve el dataset crudo, descargándolo y cacheándolo en parquet si hace falta.

    Args:
        forzar: si es True, vuelve a descargar aunque exista el parquet.

    Returns:
        DataFrame con las 541.909 filas originales y tipos normalizados.

    Raises:
        FileNotFoundError: si no se pudo obtener el dataset por ninguna vía.
    """
    if config.PARQUET_RAW.exists() and not forzar:
        return pd.read_parquet(config.PARQUET_RAW)

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if config.XLSX_RAW.exists() and not forzar:
        df = _leer_xlsx()
    else:
        errores: list[str] = []
        df = None
        for nombre, funcion in [("UCI zip", _descargar_zip_uci), ("ucimlrepo", _descargar_ucimlrepo)]:
            try:
                print(f"Descargando con {nombre}...")
                df = funcion()
                break
            except Exception as exc:  # noqa: BLE001 - se informa y se prueba la siguiente vía
                errores.append(f"{nombre}: {exc}")
        if df is None:
            raise FileNotFoundError(
                "No se pudo descargar el dataset.\n"
                + "\n".join(errores)
                + f"\nDescárgalo de {config.URL_DATASET} y coloca "
                f"'Online Retail.xlsx' en {config.DATA_RAW}"
            )

    df = normalizar_tipos(df)
    df.to_parquet(config.PARQUET_RAW, index=False)
    return df


if __name__ == "__main__":
    datos = obtener_datos_crudos()
    print(f"Filas: {len(datos):,} | Columnas: {list(datos.columns)}")
    print(f"Parquet guardado en: {config.PARQUET_RAW}")
