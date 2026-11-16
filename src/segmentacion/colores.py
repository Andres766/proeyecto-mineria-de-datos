"""Colores fijos por segmento, compartidos por notebooks y app.

Módulo sin dependencias: la app lo importa y en el despliegue solo están instaladas las librerías de
requirements.txt (sin matplotlib).
"""

# Color fijo por segmento (mismo en notebooks y app). Paleta comprobada con un validador de
# accesibilidad: pasa la separación para daltonismo (ΔE ≥ 9,2) y para visión normal (ΔE ≥ 16,3)
# en TODOS los pares, requisito de los gráficos de dispersión (PCA, t-SNE). El aqua queda por
# debajo de 3:1 de contraste con el fondo claro: siempre se acompaña de leyenda, etiqueta o tabla.
COLORES_SEGMENTO: dict[int, str] = {
    0: "#eb6834",  # naranja
    1: "#4a3aa7",  # violeta
    2: "#1baf7a",  # aqua
    3: "#2a78d6",  # azul
}
# Las cuentas clave no llevan un quinto color (rompería la validación): se marcan con forma.
MARCADOR_CUENTA_CLAVE: str = "D"  # rombo en matplotlib ("diamond" en Plotly)
COLOR_BORDE_CUENTA_CLAVE: str = "#0b0b0b"
