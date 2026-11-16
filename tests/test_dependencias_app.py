"""La app solo puede usar librerías de requirements.txt.

En el despliegue (Streamlit Community Cloud) se instala solo requirements.txt: si la app importara una
librería de desarrollo (matplotlib, seaborn…) fallaría al arrancar. Esta prueba importa los módulos de la
app en un proceso aparte y comprueba que ninguna dependencia de desarrollo queda cargada.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SOLO_DESARROLLO = ["matplotlib", "seaborn", "pytest", "openpyxl", "ucimlrepo", "requests_cache", "IPython"]

CODIGO = f"""
import sys
sys.path[:0] = [{str(RAIZ / 'app')!r}, {str(RAIZ / 'src')!r}]
import comun  # noqa: F401  (importa todos los módulos de src que usa la app)
cargados = [m for m in {SOLO_DESARROLLO!r} if m in sys.modules]
print(",".join(cargados))
"""


def test_la_app_no_importa_dependencias_de_desarrollo():
    resultado = subprocess.run([sys.executable, "-c", CODIGO], capture_output=True, text=True, cwd=RAIZ, timeout=120)
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stdout.strip() == "", f"La app importa dependencias de desarrollo: {resultado.stdout.strip()}"


def test_requirements_cubre_las_importaciones_de_la_app():
    requisitos = {linea.split("==")[0].strip().lower() for linea in (RAIZ / "requirements.txt").read_text().splitlines()
                  if linea.strip() and not linea.startswith("#")}
    assert {"streamlit", "plotly", "pandas", "numpy", "scikit-learn", "joblib", "pyarrow"} <= requisitos
