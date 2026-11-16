# Guía de despliegue

## Antes de desplegar (ya comprobado)

- La app se probó en un **entorno virtual nuevo con solo `requirements.txt`**. Las 6 páginas cargan, los modelos
  (`.joblib`) se cargan sin depender del paquete del proyecto y el formulario de cliente nuevo funciona.
- Una prueba automática (`tests/test_dependencias_app.py`) falla si la app llega a importar una librería que no esté en
  `requirements.txt`, como matplotlib.
- El repositorio contiene todo lo que la app necesita: código (`app/`, `src/`), modelos (`models/`), datos procesados
  (`data/processed/`) y configuración (`.streamlit/config.toml`). El dataset crudo no se sube.

## Opción A · Streamlit Community Cloud (recomendada, gratuita)

1. **Sube los últimos cambios a GitHub.** El repositorio es `https://github.com/Andres766/proeyecto-mineria-de-datos`,
   rama `main`.
2. Entra en **https://share.streamlit.io** y pulsa **Continue with GitHub**. Autoriza a Streamlit a leer tus
   repositorios; si el repositorio es privado, concede también acceso a repositorios privados.
3. Pulsa **Create app** y elige **Deploy a public app from GitHub** (o la opción equivalente, "Yup, I have an app").
4. Rellena el formulario:
   - **Repository:** `Andres766/proeyecto-mineria-de-datos`
   - **Branch:** `main`
   - **Main file path:** `app/Inicio.py`
   - **App URL** (opcional): un subdominio fácil de recordar, p. ej. `segmentacion-clientes-retail`.
5. Abre **Advanced settings** y selecciona **Python 3.11**, la versión con la que se desarrolló y probó el proyecto. Deja
   vacío el apartado de *secrets*: la app no usa ninguno. Guarda.
6. Pulsa **Deploy**. Streamlit instala `requirements.txt`; la primera vez tarda unos minutos y puedes seguir el progreso
   en el registro de la derecha.
7. Cuando cargue, recorre las 6 páginas del menú lateral. Copia la URL y **pásamela** para revisar la app publicada
   página por página.

**Después del despliegue:**

- Cada `git push` a `main` vuelve a desplegar la app automáticamente.
- Si nadie la usa durante un tiempo, Streamlit la "duerme": al abrirla aparece un botón para despertarla, que tarda un
  momento. Antes de presentarla, ábrela unos minutos antes.
- La versión de Python solo se elige al crear la app. Para cambiarla hay que borrarla (**Settings → Delete app**) y
  desplegarla de nuevo.

**Si algo falla:**

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError` en el registro | Falta una librería en `requirements.txt` | Añadirla con versión fija, ejecutar `pytest` y hacer push |
| Error al instalar dependencias | Versión de Python distinta de la 3.11 | Borrar la app y desplegarla de nuevo eligiendo 3.11 |
| `FileNotFoundError` en `models/` o `data/processed/` | Archivo sin subir | `git status`, añadir el archivo, commit y push |
| La página no carga tras un push | Despliegue en curso | Esperar a que termine o pulsar *Reboot app* en el menú de la app |

## Opción B · Docker

Imagen basada en `python:3.11-slim` con solo `requirements.txt`. Se probó en local: el contenedor queda en estado
*healthy* y las 6 páginas cargan sin errores.

```bash
docker build -t segmentacion-clientes .
docker run --rm -p 8501:8501 segmentacion-clientes
# abrir http://localhost:8501
```

La imagen sirve para cualquier servicio que ejecute contenedores (Render, Railway, Google Cloud Run, Azure Container
Apps…). En todos ellos hay que exponer el puerto `8501`; el `HEALTHCHECK` usa `/_stcore/health`.

## Opción C · Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Inicio.py
```
