"""Exporta a Word (.docx) y PDF el informe y el resumen para la exposición, con formato APA (7.ª edición).

Formato APA: portada de estudiante, tamaño carta con márgenes de 2,54 cm, Times New Roman 12 a doble espacio, sangría
de 1,27 cm, número de página arriba a la derecha, títulos sin numerar (nivel 1 centrado y nivel 2 a la izquierda, en
negrita), tablas y figuras con «Tabla N»/«Figura N» en negrita y el título en cursiva encima, tablas solo con líneas
horizontales y referencias con sangría francesa.

Uso (desde la raíz del proyecto, en Windows con Microsoft Word instalado para el PDF):
    python -m segmentacion.reportes                         # 1) regenerar los .md con las cifras actuales
    python scripts/exportar_informe.py                      # 2) INFORME y RESUMEN_EXPOSICION (.docx + .pdf)
    python scripts/exportar_informe.py RESUMEN_EXPOSICION   #    o solo uno de ellos

Los datos de la portada (autores, docente, fecha…) están en reports/portada.json.
El índice y el índice de figuras son campos de Word: los actualiza scripts/docx_a_pdf.ps1 antes de exportar
el PDF (o se actualizan a mano en Word con F9).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
REPORTES = RAIZ / "reports"
NEGRO = RGBColor(0, 0, 0)
FUENTE, TAMANO = "Times New Roman", 12
SANGRIA = Cm(1.27)
ANCHO_MAX_CM, ALTO_MAX_CM = 16.4, 18.0   # área útil de una hoja carta con márgenes de 2,54 cm: 16,51 cm de ancho
FILAS_MAX_TABLA_ENTERA = 10   # las tablas más largas pueden partirse entre páginas
# Documento -> subtítulo de portada (None = el de portada.json), índices de tablas y figuras, y si es un artículo APA
# (resumen sin sangría y el título del trabajo en lugar del encabezado «Introducción»).
DOCUMENTOS = {
    "INFORME": {"subtitulo": None, "indice_figuras": True, "articulo": True},
    "RESUMEN_EXPOSICION": {"subtitulo": "Resumen para la exposición y preguntas probables", "indice_figuras": False,
                           "articulo": False},
}


# ---------------------------------------------------------------------------
# Utilidades de Word
# ---------------------------------------------------------------------------
def campo(parrafo, instruccion: str, texto_previo: str = "", negrita: bool = False) -> None:
    """Inserta un campo de Word (TOC, PAGE, SEQ…) en un párrafo."""
    def run_con(elemento):
        r = OxmlElement("w:r")
        if negrita:
            propiedades = OxmlElement("w:rPr")
            propiedades.append(OxmlElement("w:b"))
            r.append(propiedades)
        r.append(elemento)
        parrafo._p.append(r)

    inicio = OxmlElement("w:fldChar")
    inicio.set(qn("w:fldCharType"), "begin")
    run_con(inicio)
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruccion} "
    run_con(instr)
    separador = OxmlElement("w:fldChar")
    separador.set(qn("w:fldCharType"), "separate")
    run_con(separador)
    texto = OxmlElement("w:t")
    texto.text = texto_previo
    run_con(texto)
    fin = OxmlElement("w:fldChar")
    fin.set(qn("w:fldCharType"), "end")
    run_con(fin)


def bordes(propiedades, **lados: bool) -> None:
    """Añade bordes (top, bottom, left, right, insideH, insideV) a unas propiedades de tabla o de celda."""
    etiqueta = "w:tblBorders" if propiedades.tag == qn("w:tblPr") else "w:tcBorders"
    contenedor = OxmlElement(etiqueta)
    for lado, visible in lados.items():
        borde = OxmlElement(f"w:{lado}")
        borde.set(qn("w:val"), "single" if visible else "nil")
        if visible:
            borde.set(qn("w:sz"), "8")
            borde.set(qn("w:space"), "0")
            borde.set(qn("w:color"), "000000")
        contenedor.append(borde)
    propiedades.append(contenedor)


def interlineado_sencillo(parrafo, sangria: bool = False) -> None:
    formato = parrafo.paragraph_format
    formato.line_spacing = 1.0
    formato.first_line_indent = SANGRIA if sangria else Cm(0)


def texto_con_formato(parrafo, texto: str, tamano: float | None = None) -> None:
    """Añade texto interpretando **negrita**, *cursiva* y `código` de Markdown."""
    texto = re.sub(r"(\d) ([%£])", "\\1\u00a0\\2", texto)   # «20,5 %» no se parte al final de una línea
    for trozo in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", texto):
        if not trozo:
            continue
        if trozo.startswith("**") and trozo.endswith("**"):
            run = parrafo.add_run(trozo[2:-2])
            run.bold = True
        elif trozo.startswith("*") and trozo.endswith("*") and len(trozo) > 2:
            run = parrafo.add_run(trozo[1:-1])
            run.italic = True
        elif trozo.startswith("`") and trozo.endswith("`"):
            run = parrafo.add_run(trozo[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        else:
            run = parrafo.add_run(trozo.replace("&lt;", "<"))
        if tamano:
            run.font.size = Pt(tamano)


def _fuente(estilo) -> None:
    estilo.font.name = FUENTE
    estilo.font.size = Pt(TAMANO)
    estilo.font.color.rgb = NEGRO
    fuentes = estilo.element.get_or_add_rPr().get_or_add_rFonts()
    for atributo in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fuentes.attrib.pop(qn(atributo), None)
    for atributo in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fuentes.set(qn(atributo), FUENTE)


def configurar_estilos(doc: Document) -> None:
    """Estilos APA: Times New Roman 12, doble espacio sin espacio extra entre párrafos y títulos en negrita."""
    normal = doc.styles["Normal"]
    _fuente(normal)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 2.0
    normal.paragraph_format.widow_control = True
    for nombre, alineacion, cursiva in [("Heading 1", WD_ALIGN_PARAGRAPH.CENTER, False),
                                        ("Heading 2", WD_ALIGN_PARAGRAPH.LEFT, False),
                                        ("Heading 3", WD_ALIGN_PARAGRAPH.LEFT, True)]:
        estilo = doc.styles[nombre]
        _fuente(estilo)
        estilo.font.bold = True
        estilo.font.italic = cursiva
        estilo.paragraph_format.alignment = alineacion
        estilo.paragraph_format.space_before = Pt(0)
        estilo.paragraph_format.space_after = Pt(0)
        estilo.paragraph_format.line_spacing = 2.0
        estilo.paragraph_format.first_line_indent = Cm(0)
        estilo.paragraph_format.keep_with_next = True
    for nombre in ("Caption", "List Bullet", "List Bullet 2", "TOC Heading", "toc 1", "toc 2", "table of figures"):
        try:
            estilo = doc.styles[nombre]
        except KeyError:
            continue
        _fuente(estilo)
    pie = doc.styles["Caption"]
    pie.font.italic = False
    pie.font.bold = False
    pie.paragraph_format.space_before = Pt(0)
    pie.paragraph_format.space_after = Pt(0)
    pie.paragraph_format.line_spacing = 2.0
    pie.paragraph_format.keep_with_next = True


def pie_numerado(doc: Document, tipo: str, numero: int, texto: str) -> None:
    """Rótulo APA encima de la tabla o figura: «Tabla N» en negrita y, en la línea siguiente, el título en cursiva.

    El número es un campo SEQ para que Word construya los índices de tablas y de figuras.
    """
    parrafo = doc.add_paragraph(style="Caption")
    parrafo.alignment = WD_ALIGN_PARAGRAPH.LEFT
    parrafo.add_run(f"{tipo} ").bold = True
    campo(parrafo, f"SEQ {tipo} \\* ARABIC", str(numero), negrita=True)
    titulo = parrafo.add_run()
    titulo.add_break()
    titulo.add_text(texto.rstrip("."))
    titulo.italic = True


# ---------------------------------------------------------------------------
# Portada e índices
# ---------------------------------------------------------------------------
def lista_de_nombres(nombres: list[str]) -> str:
    """«A, B, C y D», como pide APA para varios autores en la portada."""
    return nombres[0] if len(nombres) == 1 else f"{', '.join(nombres[:-1])} y {nombres[-1]}"


def portada(doc: Document, datos: dict) -> None:
    """Portada de estudiante APA 7: título en negrita a 3–4 líneas del margen superior y, tras una línea en blanco,
    autores, programa y universidad, asignatura, docente y fecha, todo centrado y sin etiquetas."""
    def linea(texto: str = "", negrita: bool = False) -> None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if texto:
            p.add_run(texto).bold = negrita

    for _ in range(3):
        linea()
    linea(datos["titulo"], negrita=True)
    linea(datos["subtitulo"], negrita=True)
    linea()
    linea(lista_de_nombres(datos["autores"]))
    linea(f"{datos['programa']}, {datos['universidad']}")
    linea(datos["asignatura"])
    linea(datos["docente"])
    linea(datos["fecha"])


def indices(doc: Document, indice_figuras: bool = True) -> None:
    """Índice general y, si se pide, índices de tablas y de figuras, cada uno en su página."""
    titulos = [("Índice", 'TOC \\o "1-2" \\h \\z \\u'), ("Índice de tablas", 'TOC \\h \\z \\c "Tabla"'),
               ("Índice de figuras", 'TOC \\h \\z \\c "Figura"')]
    titulos = titulos if indice_figuras else titulos[:1]
    for titulo, instruccion in titulos:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.page_break_before = True
        p.add_run(titulo).bold = True
        campo(doc.add_paragraph(), instruccion, "Haga clic derecho y elija «Actualizar campo» para generar el índice.")


def numeros_de_pagina(seccion) -> None:
    """Número de página arriba a la derecha en todas las páginas, también en la portada (APA 7)."""
    encabezado = seccion.header.paragraphs[0]
    encabezado.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    encabezado.paragraph_format.line_spacing = 1.0
    campo(encabezado, "PAGE", "1")


# ---------------------------------------------------------------------------
# Cuerpo
# ---------------------------------------------------------------------------
def anchos_de_columna(cabecera: list[str], cuerpo: list[list[str]], tamano: float) -> list:
    """Anchos de columna que evitan partir el texto de las celdas.

    Ideal de una columna: el ancho de su celda más larga (la cabecera puede ir en varias líneas). Mínimo: el de su
    palabra más larga, para no partir palabras. Si los ideales caben, se usan y el sobrante se reparte por igual; si no,
    cada columna recibe su mínimo y el espacio restante va a las columnas en proporción a lo que les falta para su ideal.
    """
    cm_por_caracter = 0.016 * tamano   # ancho medio aproximado de un carácter de Times New Roman
    NEGRITA = 1.15                     # la cabecera va en negrita, más ancha

    def palabras(texto: str) -> list[str]:   # «1.078,19 £» no se parte (espacio de no separación)
        return re.sub(r"(\d) ([%£])", r"\1~\2", texto).split(" ") or [""]

    columnas = range(len(cabecera))
    palabra_cabecera = [NEGRITA * max(len(p) for p in palabras(cabecera[j])) for j in columnas]
    palabra_cuerpo = [max(len(p) for f in cuerpo for p in palabras(f[j])) for j in columnas]
    ideales = [cm_por_caracter * max(max(len(f[j]) for f in cuerpo), palabra_cabecera[j]) + 0.4 for j in columnas]
    if sum(ideales) <= ANCHO_MAX_CM:
        extra = (ANCHO_MAX_CM - sum(ideales)) / len(ideales)
        return [Cm(ideal + extra) for ideal in ideales]
    minimos = [cm_por_caracter * max(palabra_cabecera[j], palabra_cuerpo[j]) + 0.4 for j in columnas]
    faltas = [max(ideal - minimo, 0) for ideal, minimo in zip(ideales, minimos)]
    sobrante = max(ANCHO_MAX_CM - sum(minimos), 0)
    return [Cm(minimo + sobrante * falta / (sum(faltas) or 1)) for minimo, falta in zip(minimos, faltas)]


def agregar_tabla(doc: Document, filas: list[str]) -> None:
    celdas = [[c.strip() for c in f.strip().strip("|").split("|")] for f in filas]
    cabecera, cuerpo = celdas[0], [f for f in celdas[2:]]
    tabla = doc.add_table(rows=1 + len(cuerpo), cols=len(cabecera))
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    # APA: sin líneas verticales; solo una línea encima y debajo de la cabecera y otra al final de la tabla.
    bordes(tabla._tbl.tblPr, top=True, bottom=True, left=False, right=False, insideH=False, insideV=False)
    tamano = 9 if len(cabecera) > 6 or len(cuerpo) > FILAS_MAX_TABLA_ENTERA else 10
    for j, texto in enumerate(cabecera):
        celda = tabla.rows[0].cells[j]
        celda.paragraphs[0].text = ""
        interlineado_sencillo(celda.paragraphs[0])
        texto_con_formato(celda.paragraphs[0], f"**{texto}**", tamano)
        bordes(celda._tc.get_or_add_tcPr(), bottom=True)
    for i, fila in enumerate(cuerpo, start=1):
        for j, texto in enumerate(fila):
            parrafo = tabla.rows[i].cells[j].paragraphs[0]
            interlineado_sencillo(parrafo)
            texto_con_formato(parrafo, texto, tamano)
            if j > 0 and re.fullmatch(r"[\d.,%£ —-]+", texto):
                parrafo.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if len(cuerpo) <= FILAS_MAX_TABLA_ENTERA:
        # Mantener la tabla en una sola página: cada fila "sigue" a la siguiente.
        for fila in tabla.rows[:-1]:
            for celda in fila.cells:
                for parrafo in celda.paragraphs:
                    parrafo.paragraph_format.keep_with_next = True
    else:
        # Tabla larga: puede partirse entre páginas, repitiendo la cabecera (que no queda sola al pie de una página).
        tabla.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for celda in tabla.rows[0].cells:
            celda.paragraphs[0].paragraph_format.keep_with_next = True
    tabla.autofit = False
    for j, ancho in enumerate(anchos_de_columna(cabecera, cuerpo, tamano)):
        tabla.columns[j].width = ancho   # rejilla de la tabla (la que respeta Word)
        for fila in tabla.rows:
            fila.cells[j].width = ancho
    for fila in tabla.rows:  # ninguna fila se parte
        fila._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def separador(doc: Document) -> None:
    """Línea en blanco tras una tabla o figura (o tras su nota)."""
    interlineado_sencillo(doc.add_paragraph())


def agregar_imagen(doc: Document, ruta: Path) -> None:
    ancho_px, alto_px = Image.open(ruta).size
    ancho_cm = min(ANCHO_MAX_CM, ALTO_MAX_CM * ancho_px / alto_px)
    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    interlineado_sencillo(parrafo)   # con doble espacio Word duplicaría la altura de la línea de la imagen
    parrafo.add_run().add_picture(str(ruta), width=Cm(ancho_cm))
    separador(doc)


def convertir(markdown: str, doc: Document, titulo: str, articulo: bool) -> None:
    """Convierte el Markdown del documento en párrafos de Word con formato APA.

    Con ``articulo`` (el informe), el resumen va sin sangría y el encabezado «Introducción» se sustituye por el título
    del trabajo en una página nueva, como pide APA para la primera página del texto.
    """
    lineas = markdown.splitlines()
    i, titulo_saltado, seccion_actual, parrafo_pendiente = 0, False, "", []

    def volcar_parrafo():
        if parrafo_pendiente:
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            texto = " ".join(parrafo_pendiente)
            nota = texto.startswith("*Nota.*")
            resumen = articulo and seccion_actual == "Resumen" and not texto.startswith("*Palabras clave:*")
            p.paragraph_format.first_line_indent = Cm(0) if nota or resumen else SANGRIA
            texto_con_formato(p, texto)
            parrafo_pendiente.clear()
            if nota:
                separador(doc)

    def siguiente_no_vacia(desde: int) -> str:
        for linea in lineas[desde:]:
            if linea.strip():
                return linea.strip()
        return ""

    while i < len(lineas):
        linea = lineas[i].rstrip()
        sin_espacio = linea.strip()
        if sin_espacio.startswith("# ") and not titulo_saltado:
            titulo_saltado = True
            while i < len(lineas) and not lineas[i].startswith("## "):   # título y subtítulo van en la portada
                i += 1
            continue
        if not sin_espacio:
            volcar_parrafo()
        elif sin_espacio.startswith("## "):
            volcar_parrafo()
            primera = not seccion_actual
            seccion_actual = sin_espacio[3:]
            texto = titulo if articulo and seccion_actual == "Introducción" else seccion_actual
            encabezado = doc.add_heading(texto, level=1)
            # Página nueva para el comienzo del cuerpo, el inicio del texto (título) y las referencias.
            encabezado.paragraph_format.page_break_before = primera or texto == titulo or seccion_actual == "Referencias"
        elif sin_espacio.startswith("### "):
            volcar_parrafo()
            doc.add_heading(sin_espacio[4:], level=2)
        elif (imagen := re.fullmatch(r"!\[Figura (\d+)\. (.+?)\]\((.+?)\)", sin_espacio)):
            volcar_parrafo()
            pie_numerado(doc, "Figura", int(imagen.group(1)), imagen.group(2))
            agregar_imagen(doc, REPORTES / imagen.group(3))
        elif re.fullmatch(r"\*\*\d+\. .+\*\*", sin_espacio):   # pregunta numerada en negrita: párrafo propio
            volcar_parrafo()
            p = doc.add_paragraph()
            p.paragraph_format.keep_with_next = True
            texto_con_formato(p, sin_espacio)
        elif re.fullmatch(r"\*Figura \d+\..+\*", sin_espacio):
            pass  # el rótulo ya se añadió encima de la imagen
        elif (rotulo := re.fullmatch(r"\*Tabla (\d+)\. (.+)\*", sin_espacio)):
            volcar_parrafo()
            pie_numerado(doc, "Tabla", int(rotulo.group(1)), rotulo.group(2))
        elif sin_espacio.startswith("|"):
            volcar_parrafo()
            filas = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                filas.append(lineas[i])
                i += 1
            agregar_tabla(doc, filas)
            if not siguiente_no_vacia(i).startswith("*Nota.*"):   # la nota va justo debajo de la tabla
                separador(doc)
            continue
        elif re.match(r"^\s*- ", linea):
            volcar_parrafo()
            nivel = 2 if linea.startswith("  ") else 1
            texto = re.sub(r"^\s*- ", "", linea)
            while (i + 1 < len(lineas) and lineas[i + 1].startswith("    " if nivel == 2 else "  ")
                   and not re.match(r"^\s*(- |\d+\. )", lineas[i + 1])):
                i += 1
                texto += " " + lineas[i].strip()
            if seccion_actual == "Referencias":   # APA: sin viñeta y con sangría francesa de 1,27 cm
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = SANGRIA
                p.paragraph_format.first_line_indent = -SANGRIA
            else:
                p = doc.add_paragraph(style="List Bullet" if nivel == 1 else "List Bullet 2")
            texto_con_formato(p, texto)
        elif (numerado := re.match(r"^(\d+)\. (.+)", sin_espacio)):
            volcar_parrafo()
            texto = numerado.group(2)
            while i + 1 < len(lineas) and lineas[i + 1].startswith("   ") and lineas[i + 1].strip():
                i += 1
                texto += " " + lineas[i].strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = SANGRIA
            p.paragraph_format.first_line_indent = -SANGRIA
            texto_con_formato(p, f"{numerado.group(1)}.\t{texto}")
        else:
            parrafo_pendiente.append(sin_espacio)
        i += 1
    volcar_parrafo()


def construir_docx(nombre: str, destino: Path) -> None:
    opciones = DOCUMENTOS[nombre]
    datos = json.loads((REPORTES / "portada.json").read_text(encoding="utf-8"))
    datos["subtitulo"] = opciones["subtitulo"] or datos["subtitulo"]
    doc = Document()
    configurar_estilos(doc)
    seccion = doc.sections[0]
    # APA: tamaño carta y márgenes de 2,54 cm (1 pulgada).
    seccion.page_height, seccion.page_width = Cm(27.94), Cm(21.59)
    for margen in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(seccion, margen, Cm(2.54))
    seccion.header_distance = seccion.footer_distance = Cm(1.27)
    doc.core_properties.title = datos["titulo"]
    doc.core_properties.author = lista_de_nombres(datos["autores"])
    doc.core_properties.language = "es-CO"

    numeros_de_pagina(seccion)
    portada(doc, datos)
    indices(doc, opciones["indice_figuras"])
    markdown = (REPORTES / f"{nombre}.md").read_text(encoding="utf-8")
    convertir(markdown, doc, datos["titulo"], opciones["articulo"])
    doc.save(destino)


def exportar(nombre: str) -> None:
    docx = REPORTES / f"{nombre}.docx"
    pdf = REPORTES / f"{nombre}.pdf"
    construir_docx(nombre, docx)
    print(f"Generado {docx.relative_to(RAIZ)}")
    if sys.platform == "win32":
        resultado = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                    str(RAIZ / "scripts" / "docx_a_pdf.ps1"), str(docx), str(pdf)],
                                   capture_output=True, text=True)
        if resultado.returncode != 0:
            sys.exit(f"No se pudo generar el PDF con Word:\n{resultado.stdout}\n{resultado.stderr}")
        print(f"Generado {pdf.relative_to(RAIZ)} (índices actualizados por Word)")
    else:
        print("PDF: abre el .docx en Word, actualiza los índices (F9) y exporta a PDF.")


if __name__ == "__main__":
    pedidos = sys.argv[1:] or list(DOCUMENTOS)
    desconocidos = [n for n in pedidos if n not in DOCUMENTOS]
    if desconocidos:
        sys.exit(f"Documento desconocido: {desconocidos}. Opciones: {list(DOCUMENTOS)}")
    for nombre in pedidos:
        exportar(nombre)
