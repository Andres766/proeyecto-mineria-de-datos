# Imagen de la app de segmentación (alternativa a Streamlit Community Cloud).
#   docker build -t segmentacion-clientes .
#   docker run --rm -p 8501:8501 segmentacion-clientes   ->  http://localhost:8501
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Solo las dependencias de ejecución (las mismas que usa Streamlit Cloud).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código, configuración y artefactos ya entrenados: la app no reentrena nada.
COPY .streamlit/ .streamlit/
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/processed/ data/processed/

# Usuario sin privilegios.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/Inicio.py", "--server.port=8501", "--server.address=0.0.0.0"]
