# Documentación técnica

**Proyecto:** Segmentación avanzada de clientes con clustering y reducción de dimensionalidad · Inteligencia de Negocios
y Minería de Datos · Universidad Cooperativa de Colombia.
**Equipo:** {integrantes}. **Docente:** {docente}.
**App publicada:** https://segmentacion-retail-ucc.streamlit.app

Este documento describe cómo está construido el sistema: arquitectura, flujo de datos, módulos, artefactos, aplicación,
pruebas, despliegue y cómo mantenerlo. La metodología y los resultados se explican en el
[informe](reports/INFORME.md). Todas las cifras de este documento se generan desde el código
(`python -m segmentacion.reportes`).

## 1. Visión general

El sistema toma las transacciones del dataset *Online Retail* (UCI), las limpia, construye variables por cliente,
entrena y compara modelos de clustering, perfila los segmentos y publica los resultados en una aplicación web que
además clasifica clientes nuevos con el modelo entrenado.

```
UCI (Online Retail)
   │  segmentacion.carga           descarga y convierte a parquet
   ▼
data/raw/online_retail.parquet
   │  segmentacion.limpieza        6 pasos con bitácora
   ▼
data/processed/transacciones_limpias.parquet  (+ devoluciones, bitácora)
   │  notebooks 02 → 05            variables, modelado, perfiles y validación
   ▼
models/*.joblib, models/*.json, data/processed/clientes_segmentados.parquet
   │                                         │
   ▼                                         ▼
app/ (Streamlit: lee artefactos,      segmentacion.reportes + scripts/exportar_informe.py
      no reentrena)                   (informe, guía, resumen, README y este documento)
```

**Principios de diseño:**

- **Separación entre lógica y narración:** toda la lógica está en el paquete `src/segmentacion` (funciones con
  docstrings y anotaciones de tipo); los notebooks solo la invocan y explican los resultados.
- **Entrenar una vez, usar muchas:** los notebooks guardan los modelos ya entrenados en `models/`; la app y los
  documentos solo los leen.
- **Nada escrito a mano:** las cifras de la app y de los documentos se leen de los artefactos (`models/*.json`).
- **Reproducibilidad:** `random_state = 42` en todo proceso aleatorio y versiones fijas de las dependencias.

## 2. Tecnologías

| Componente | Uso |
|---|---|
| Python 3.11 | Lenguaje de todo el proyecto |
| pandas, NumPy, PyArrow | Manipulación de datos y almacenamiento en parquet |
| scikit-learn, SciPy | Preprocesamiento, clustering, PCA, t-SNE, métricas, clasificadores y algoritmo húngaro |
| joblib | Serialización de los modelos entrenados |
| Streamlit, Plotly | Aplicación web y gráficos interactivos |
| Jupyter, matplotlib, seaborn | Notebooks y figuras del informe (solo desarrollo) |
| pytest | Pruebas automáticas |
| python-docx, Microsoft Word | Exportación de los documentos a Word y PDF |
| Docker | Despliegue alternativo |

Las versiones exactas están fijadas en `requirements.txt` (ejecución: lo que instala la app publicada) y
`requirements-dev.txt` (desarrollo: notebooks, pruebas y documentos).

## 3. Estructura del repositorio

```
app/                  Inicio.py + pages/ (6 páginas de Streamlit) y comun.py (utilidades compartidas)
data/processed/       transacciones limpias, devoluciones, bitácora y clientes con variables y segmentos
data/raw/             dataset original (no se versiona; lo descarga segmentacion.carga)
models/               preprocesadores, modelos entrenados y resultados en JSON (los lee la app)
notebooks/            01_eda · 02_features · 03_modelado · 04_perfiles · 05_validacion (con salidas)
reports/              informe, guía de sustentación, guion de demo, resumen, figuras y plantillas
scripts/              exportación de documentos a Word y PDF
src/segmentacion/     paquete con toda la lógica del proyecto
tests/                pruebas automáticas (pytest)
.streamlit/           configuración de la app (tema)
Dockerfile            imagen de la app
```

## 4. Flujo de ejecución

Para reproducir el análisis completo desde cero (con `PYTHONPATH=src`):

| Paso | Comando | Genera |
|---|---|---|
| 1. Descarga | `python -m segmentacion.carga` | `data/raw/online_retail.parquet` |
| 2. Limpieza | `python -m segmentacion.limpieza` | `transacciones_limpias.parquet`, `devoluciones.parquet`, `bitacora_limpieza.csv` |
| 3. Exploración | notebook `01_eda` | Figuras de la sección de datos |
| 4. Variables | notebook `02_features` | `clientes_features.parquet`, `preprocesador.joblib`, `features.json` |
| 5. Modelado | notebook `03_modelado` | `kmeans.joblib`, `gmm.joblib`, `pca.joblib`, `dbscan_nucleos.joblib`, `preprocesador_sin_winsor.joblib`, `metricas.json`, `clientes_segmentados.parquet` |
| 6. Perfiles | notebook `04_perfiles` | `segmentos.json` y nombres de segmento en `clientes_segmentados.parquet` |
| 7. Validación | notebook `05_validacion` | `validacion.json` |
| 8. Documentos | `python -m segmentacion.reportes` y `python scripts/exportar_informe.py` | Markdown, Word y PDF |
| 9. Pruebas | `pytest` | {n_pruebas} pruebas |

Los notebooks se ejecutan en orden con `jupyter nbconvert --to notebook --execute --inplace`: cada uno usa lo que
genera el anterior.

## 5. Módulos del paquete `src/segmentacion`

| Módulo | Responsabilidad | Funciones principales |
|---|---|---|
| `config` | Rutas, nombres de artefactos y semilla (`RANDOM_STATE = 42`) | — |
| `carga` | Descargar el dataset de UCI y normalizar tipos | `obtener_datos_crudos`, `normalizar_tipos` |
| `limpieza` | Limpieza en 6 pasos con bitácora | `limpiar`, `emparejar_compras_anuladas`, `emparejar_anulaciones_manuales`, `guardar` |
| `features` | Variables por cliente, tasa de devoluciones, VIF, preprocesador y baseline RFM | `calcular_variables_cliente`, `tasa_devoluciones`, `construir_preprocesador`, `segmentar_rfm_quintiles` |
| `modelado` | PCA, K-Means, GMM, DBSCAN, HDBSCAN y t-SNE | `ajustar_kmeans`, `barrido_kmeans`, `ajustar_gmm`, `diagnostico_degeneracion`, `ajustar_dbscan`, `es_atipico_dbscan`, `calcular_tsne` |
| `evaluacion` | Métricas internas, estabilidad, ARI y exactitud con datos de prueba | `metricas_internas`, `estabilidad_submuestreo`, `matriz_ari`, `exactitud_reasignacion`, `comparar_clasificadores` |
| `perfilamiento` | Perfil, indicadores y estacionalidad de cada segmento | `resumen_segmentos`, `indicadores_segmento`, `facturas_por_mes`, `productos_top` |
| `catalogo` | Nombres, descripciones y estrategias de cada segmento | — (datos) |
| `prediccion` | Clasificar un cliente nuevo con los artefactos guardados | `cargar_artefactos`, `validar_entrada`, `clasificar_cliente` |
| `colores` | Colores fijos por segmento (sin dependencias gráficas) | — (constantes) |
| `visualizacion` | Estilo y guardado de figuras de los notebooks | `estilo`, `guardar_figura` |
| `reportes` | Calcular cifras y renderizar los documentos desde plantillas | `calcular_cifras`, `renderizar`, `generar` |

## 6. Artefactos

**Modelos y resultados (`models/`):**

| Archivo | Contenido | Lo usa |
|---|---|---|
| `preprocesador.joblib` | Pipeline log1p → recorte p1–p99 (límites del entrenamiento) → StandardScaler | App, predicción |
| `preprocesador_sin_winsor.joblib` | Mismo pipeline sin recorte (espacio de DBSCAN) | Predicción |
| `kmeans.joblib` | Modelo final: K-Means con k = {k_final} | App, predicción |
| `gmm.joblib` | Mezcla gaussiana de comparación | Notebooks |
| `pca.joblib` | PCA para los mapas 2D y 3D | App, predicción |
| `dbscan_nucleos.joblib` | Vecinos más cercanos sobre los puntos núcleo de DBSCAN | Predicción (marca de atípico) |
| `features.json` | Variables, límites del recorte y medianas | App (formulario) |
| `metricas.json` | Comparación de modelos, barridos, parámetros de GMM y DBSCAN, PCA y t-SNE | App, documentos |
| `segmentos.json` | Perfil, indicadores y estrategias de cada segmento y de las cuentas clave | App, documentos |
| `validacion.json` | Exactitud con datos de prueba, reglas del árbol y tabla de ajustes | App, documentos |

Los `.joblib` contienen solo objetos de scikit-learn y NumPy (sin clases propias), de modo que se cargan con
`requirements.txt` sin depender del paquete del proyecto.

**Datos (`data/processed/`):** `transacciones_limpias.parquet` ({filas_limpias} líneas), `devoluciones.parquet`,
`bitacora_limpieza.csv`, `clientes_features.parquet` y `clientes_segmentados.parquet` ({clientes} clientes con sus
variables, segmento, coordenadas PCA y t-SNE, marca de DBSCAN y de cuenta clave).

## 7. Modelo y clasificación de un cliente nuevo

**Variables del modelo:** días desde la última compra, número de facturas, ticket promedio y productos distintos.

**Preprocesamiento** (`features.construir_preprocesador`): `log1p` → recorte a los percentiles 1 y 99 calculados en el
entrenamiento (los límites quedan fijos dentro del pipeline) → `StandardScaler`.

**Modelo final:** K-Means con k = {k_final}, 10 inicializaciones k-means++ y `random_state = 42`. Los segmentos se
numeran de 0 a 3 de menor a mayor gasto mediano.

**Clasificación de un cliente nuevo** (`prediccion.clasificar_cliente`), sin reentrenar:

1. Valida la entrada: días, facturas y productos deben ser enteros ≥ 1 y el ticket un número mayor que 0.
2. Aplica el preprocesador guardado y asigna el segmento con `kmeans.predict`.
3. Proyecta al cliente con el PCA guardado para dibujarlo sobre el mapa.
4. Informa qué variables recortó el límite p1–p99.
5. Marca de atípico: con el preprocesador sin recorte, el cliente es atípico si no tiene ningún punto núcleo de DBSCAN a
   distancia ≤ `eps` = {eps} (la misma regla con la que DBSCAN etiqueta los puntos frontera).
6. **Cuenta clave** = segmento VIP **y** atípico **y** gasto (facturas × ticket) ≥ {ck_umbral}.

Las pruebas comprueban que este proceso reproduce el segmento de los {clientes} clientes y exactamente las
{ck_clientes} cuentas clave.

## 8. Aplicación web

- **Punto de entrada:** `app/Inicio.py`; las demás páginas están en `app/pages/` (Exploración, Modelos y validación,
  Clusters, Perfiles y estrategias, Explorador).
- **`app/comun.py`:** carga de datos y modelos con caché (`st.cache_data` y `st.cache_resource`), formato numérico
  español, colores de segmento, estilo común de los gráficos y tarjetas de estrategias.
- **Solo lectura:** la app no entrena nada; lee `models/` y `data/processed/`. Por eso arranca rápido y siempre muestra
  los mismos resultados que los notebooks.
- **Dependencias mínimas:** la app no puede importar matplotlib ni seaborn, que no están en `requirements.txt`; una
  prueba automática lo vigila.
- **Configuración:** tema claro en `.streamlit/config.toml`.

## 9. Pruebas

El proyecto tiene **{n_pruebas} pruebas** (`pytest`, configuración en `pyproject.toml`):

| Archivo | Qué comprueba |
|---|---|
| `test_limpieza.py` | Emparejamiento 1 a 1 de compras anuladas y regla de las anulaciones manuales «M» |
| `test_features.py` | Tasa de devoluciones, RFM, variables por cliente, VIF, preprocesador y baseline RFM por quintiles |
| `test_modelado.py` | K-Means, GMM, DBSCAN y métricas con datos sintéticos de resultado conocido |
| `test_evaluacion.py` | Exactitud de reasignación, alineación de etiquetas y comparación de clasificadores |
| `test_perfilamiento.py` | Indicadores y perfiles de segmento |
| `test_prediccion.py` | Que la clasificación reproduce los segmentos y las cuentas clave, recortes y errores de entrada |
| `test_app.py` | Que cada página de la app carga sin errores (AppTest) y maneja los casos inválidos |
| `test_dependencias_app.py` | Que la app solo usa librerías de `requirements.txt` |
| `test_reportes.py` | Formato numérico, numeración de figuras y que una cifra inexistente haga fallar el documento |

## 10. Documentos

- `python -m segmentacion.reportes` renderiza las plantillas de `reports/plantillas/` (informe, guía de sustentación,
  guion de demo, resumen para la exposición, README y este documento). Si una plantilla usa una cifra que no está
  calculada, el proceso falla: así no puede quedar ningún número escrito a mano.
- `python scripts/exportar_informe.py` exporta el informe y el resumen a Word (formato APA 7) y a PDF mediante Microsoft
  Word. Los datos de la portada están en `reports/portada.json`.

## 11. Despliegue

- **Streamlit Community Cloud** (despliegue actual): repositorio `Andres766/proeyecto-mineria-de-datos`, rama `main`,
  archivo principal `app/Inicio.py`, Python 3.11. Cada `git push` a `main` vuelve a publicar la app.
- **Docker:** `docker build -t segmentacion-clientes .` y `docker run --rm -p 8501:8501 segmentacion-clientes`. La
  imagen instala solo `requirements.txt` e incluye un chequeo de salud.
- La app se probó en un entorno virtual nuevo con solo `requirements.txt` antes de publicarla. Pasos detallados en
  [DESPLIEGUE.md](DESPLIEGUE.md).

## 12. Mantenimiento

- **Actualizar con datos nuevos:** volver a ejecutar la limpieza y los notebooks 02 a 05 en orden, regenerar los
  documentos y ejecutar las pruebas. La segmentación es una foto fija del periodo analizado.
- **Cambiar textos o estrategias de un segmento:** editar `src/segmentacion/catalogo.py` y volver a ejecutar el notebook
  04 (las líneas base de cada KPI se recalculan).
- **Añadir una cifra a un documento:** calcularla en `src/segmentacion/reportes.py` y usarla en la plantilla como
  `{{clave}}`.
- **Antes de subir cambios:** ejecutar `pytest` y comprobar que todas las pruebas pasan.
