"""Configuración central del proyecto: rutas, semilla y constantes compartidas."""

from pathlib import Path

# Raíz del repositorio (src/segmentacion/config.py -> subir dos niveles).
RAIZ: Path = Path(__file__).resolve().parents[2]

DATA_RAW: Path = RAIZ / "data" / "raw"
DATA_PROCESSED: Path = RAIZ / "data" / "processed"
MODELS: Path = RAIZ / "models"
FIGURES: Path = RAIZ / "reports" / "figures"

# Archivos del dataset.
URL_DATASET: str = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_ID: int = 352
ZIP_RAW: Path = DATA_RAW / "online_retail.zip"
XLSX_RAW: Path = DATA_RAW / "Online Retail.xlsx"
PARQUET_RAW: Path = DATA_RAW / "online_retail.parquet"
PARQUET_LIMPIO: Path = DATA_PROCESSED / "transacciones_limpias.parquet"
PARQUET_DEVOLUCIONES: Path = DATA_PROCESSED / "devoluciones.parquet"
CSV_BITACORA: Path = DATA_PROCESSED / "bitacora_limpieza.csv"

# Semilla única para todo proceso aleatorio.
RANDOM_STATE: int = 42
