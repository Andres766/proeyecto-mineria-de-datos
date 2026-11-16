"""Exporta reports/INFORME.md a Word (.docx) y PDF con portada, índices, figuras numeradas y referencias.

Uso (desde la raíz del proyecto, en Windows con Microsoft Word instalado para el PDF):
    python -m segmentacion.reportes          # 1) regenerar INFORME.md con las cifras actuales
    python scripts/exportar_informe.py       # 2) INFORME.docx + INFORME.pdf

Los datos de la portada (autor, docente, fecha…) están en reports/portada.json.
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
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
REPORTES = RAIZ / "reports"
AZUL_OSCURO = RGBColor(0x1F, 0x38, 0x64)
GRIS_TEXTO = RGBColor(0x52, 0x51, 0x4E)
ANCHO_MAX_CM, ALTO_MAX_CM = 16.0, 19.0


# ---------------------------------------------------------------------------
# Utilidades de Word
# ---------------------------------------------------------------------------
def campo(parrafo, instruccion: str, texto_previo: str = "") -> None:
    """Inserta un campo de Word (TOC, PAGE, SEQ…) en un párrafo."""
    def run_con(elemento):
        r = OxmlElement("w:r")
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


def sombrear_celda(celda, color_hex: str) -> None:
    propiedades = celda._tc.get_or_add_tcPr()
    sombra = OxmlElement("w:shd")
    sombra.set(qn("w:val"), "clear")
    sombra.set(qn("w:color"), "auto")
    sombra.set(qn("w:fill"), color_hex)
    propiedades.append(sombra)


def texto_con_formato(parrafo, texto: str, tamano: float | None = None) -> None:
    """Añade texto interpretando **negrita**, *cursiva* y `código` de Markdown."""
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


def configurar_estilos(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for nombre, tamano, antes in [("Heading 1", 16, 18), ("Heading 2", 13, 12)]:
        estilo = doc.styles[nombre]
        estilo.font.name = "Calibri"
        estilo.font.size = Pt(tamano)
        estilo.font.bold = True
        estilo.font.color.rgb = AZUL_OSCURO
        estilo.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
        estilo.paragraph_format.space_before = Pt(antes)
        estilo.paragraph_format.space_after = Pt(6)
        estilo.paragraph_format.keep_with_next = True
    pie = doc.styles["Caption"]
    pie.font.name = "Calibri"
    pie.font.size = Pt(9.5)
    pie.font.italic = True
    pie.font.bold = False
    pie.font.color.rgb = GRIS_TEXTO
    pie.paragraph_format.space_after = Pt(10)


def pie_numerado(doc: Document, tipo: str, numero: int, texto: str, alineacion=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    """Pie de figura/tabla con campo SEQ (para que Word construya el índice de figuras)."""
    parrafo = doc.add_paragraph(style="Caption")
    parrafo.alignment = alineacion
    parrafo.paragraph_format.keep_with_next = tipo == "Tabla"  # el pie de una tabla va encima y no se separa de ella
    parrafo.add_run(f"{tipo} ")
    campo(parrafo, f"SEQ {tipo} \\* ARABIC", str(numero))
    parrafo.add_run(f". {texto}")


# ---------------------------------------------------------------------------
# Portada e índices
# ---------------------------------------------------------------------------
def portada(doc: Document, datos: dict) -> None:
    for _ in range(4):
        doc.add_paragraph()
    for texto, tamano, negrita in [(datos["universidad"], 14, True), (datos["programa"], 12, False),
                                   (datos["asignatura"], 12, False)]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(texto)
        r.font.size = Pt(tamano)
        r.bold = negrita
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(datos["titulo"])
    r.font.size = Pt(24)
    r.bold = True
    r.font.color.rgb = AZUL_OSCURO
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(datos["subtitulo"])
    r.font.size = Pt(14)
    r.font.color.rgb = GRIS_TEXTO
    for _ in range(6):
        doc.add_paragraph()
    for etiqueta, valor in [("Autor", datos["autor"]), ("Docente", datos["docente"] or "_______________________________")]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(f"{etiqueta}: ").bold = True
        p.add_run(valor)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(datos["fecha"])


def indices(doc: Document) -> None:
    """Índice general e índice de figuras, cada uno en su página (salto antes del título, no párrafos de salto)."""
    titulos = [("Índice", 'TOC \\o "1-2" \\h \\z \\u'), ("Índice de figuras", 'TOC \\h \\z \\c "Figura"')]
    for n, (titulo, instruccion) in enumerate(titulos):
        p = doc.add_paragraph()
        p.paragraph_format.page_break_before = n > 0
        r = p.add_run(titulo)
        r.bold = True
        r.font.size = Pt(16)
        r.font.color.rgb = AZUL_OSCURO
        campo(doc.add_paragraph(), instruccion, "Haga clic derecho y elija «Actualizar campo» para generar el índice.")


def numeros_de_pagina(seccion) -> None:
    pie = seccion.footer.paragraphs[0]
    pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
    campo(pie, "PAGE", "1")


# ---------------------------------------------------------------------------
# Cuerpo
# ---------------------------------------------------------------------------
def agregar_tabla(doc: Document, filas: list[str]) -> None:
    celdas = [[c.strip() for c in f.strip().strip("|").split("|")] for f in filas]
    cabecera, cuerpo = celdas[0], [f for f in celdas[2:]]
    tabla = doc.add_table(rows=1 + len(cuerpo), cols=len(cabecera))
    tabla.style = "Table Grid"
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    tamano = 8.5 if len(cabecera) > 6 else 9.5
    for j, texto in enumerate(cabecera):
        celda = tabla.rows[0].cells[j]
        celda.paragraphs[0].text = ""
        texto_con_formato(celda.paragraphs[0], f"**{texto}**", tamano)
        sombrear_celda(celda, "E6F0FB")
    for i, fila in enumerate(cuerpo, start=1):
        for j, texto in enumerate(fila):
            parrafo = tabla.rows[i].cells[j].paragraphs[0]
            texto_con_formato(parrafo, texto, tamano)
            if j > 0 and re.fullmatch(r"[\d.,%£ —-]+", texto):
                parrafo.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    # Mantener la tabla en una sola página: cada fila "sigue" a la siguiente y ninguna fila se parte.
    for fila in tabla.rows[:-1]:
        for celda in fila.cells:
            for parrafo in celda.paragraphs:
                parrafo.paragraph_format.keep_with_next = True
    for fila in tabla.rows:
        fila._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    doc.add_paragraph()


def agregar_imagen(doc: Document, ruta: Path) -> None:
    ancho_px, alto_px = Image.open(ruta).size
    ancho_cm = min(ANCHO_MAX_CM, ALTO_MAX_CM * ancho_px / alto_px)
    parrafo = doc.add_paragraph()
    parrafo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    parrafo.paragraph_format.keep_with_next = True
    parrafo.add_run().add_picture(str(ruta), width=Cm(ancho_cm))


def convertir(markdown: str, doc: Document) -> None:
    lineas = markdown.splitlines()
    i, titulo_saltado, seccion_actual, parrafo_pendiente = 0, False, "", []

    def volcar_parrafo():
        if parrafo_pendiente:
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            texto_con_formato(p, " ".join(parrafo_pendiente))
            parrafo_pendiente.clear()

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
            titulo = doc.add_heading(seccion_actual, level=1)
            titulo.paragraph_format.page_break_before = primera  # el cuerpo empieza en una página nueva
        elif sin_espacio.startswith("### "):
            volcar_parrafo()
            doc.add_heading(sin_espacio[4:], level=2)
        elif (imagen := re.fullmatch(r"!\[Figura (\d+)\. (.+?)\]\((.+?)\)", sin_espacio)):
            volcar_parrafo()
            agregar_imagen(doc, REPORTES / imagen.group(3))
            pie_numerado(doc, "Figura", int(imagen.group(1)), imagen.group(2))
        elif re.fullmatch(r"\*Figura \d+\..+\*", sin_espacio):
            pass  # el pie ya se añadió junto a la imagen
        elif (rotulo := re.fullmatch(r"\*Tabla (\d+)\. (.+)\*", sin_espacio)):
            volcar_parrafo()
            pie_numerado(doc, "Tabla", int(rotulo.group(1)), rotulo.group(2), WD_ALIGN_PARAGRAPH.LEFT)
        elif sin_espacio.startswith("|"):
            volcar_parrafo()
            filas = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                filas.append(lineas[i])
                i += 1
            agregar_tabla(doc, filas)
            continue
        elif re.match(r"^\s*- ", linea):
            volcar_parrafo()
            nivel = 2 if linea.startswith("  ") else 1
            texto = re.sub(r"^\s*- ", "", linea)
            while i + 1 < len(lineas) and lineas[i + 1].startswith("    " if nivel == 2 else "  ") and \
                    not re.match(r"^\s*(- |\d+\. )", lineas[i + 1]):
                i += 1
                texto += " " + lineas[i].strip()
            if seccion_actual == "Referencias":
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(1.0)
                p.paragraph_format.first_line_indent = Cm(-1.0)
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
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            texto_con_formato(p, f"{numerado.group(1)}.\t{texto}")
        else:
            parrafo_pendiente.append(sin_espacio)
        i += 1
    volcar_parrafo()


def construir_docx(destino: Path) -> None:
    datos = json.loads((REPORTES / "portada.json").read_text(encoding="utf-8"))
    doc = Document()
    configurar_estilos(doc)
    seccion = doc.sections[0]
    seccion.page_height, seccion.page_width = Cm(29.7), Cm(21.0)
    for margen in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(seccion, margen, Cm(2.5))
    doc.core_properties.title = datos["titulo"]
    doc.core_properties.author = datos["autor"]
    doc.core_properties.language = "es-CO"

    portada(doc, datos)
    cuerpo = doc.add_section(WD_SECTION.NEW_PAGE)   # portada sin número de página
    cuerpo.footer.is_linked_to_previous = False
    numeros_de_pagina(cuerpo)
    indices(doc)
    convertir((REPORTES / "INFORME.md").read_text(encoding="utf-8"), doc)
    doc.save(destino)


if __name__ == "__main__":
    docx = REPORTES / "INFORME.docx"
    pdf = REPORTES / "INFORME.pdf"
    construir_docx(docx)
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
