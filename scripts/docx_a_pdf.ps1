# Abre un .docx con Microsoft Word, actualiza el índice, el índice de figuras y los campos,
# guarda el documento y lo exporta a PDF.
#   powershell -ExecutionPolicy Bypass -File scripts/docx_a_pdf.ps1 reports/INFORME.docx reports/INFORME.pdf
param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [Parameter(Mandatory = $true)][string]$Pdf
)
$ErrorActionPreference = "Stop"
$rutaDocx = (Resolve-Path $Docx).Path
$rutaPdf = [System.IO.Path]::GetFullPath($Pdf)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($rutaDocx)
    $doc.Fields.Update() | Out-Null
    foreach ($indice in $doc.TablesOfContents) { $indice.Update() | Out-Null }
    foreach ($indice in $doc.TablesOfFigures) { $indice.Update() | Out-Null }
    $doc.Save()
    $doc.ExportAsFixedFormat($rutaPdf, 17)   # 17 = wdExportFormatPDF
    $doc.Close([ref]0)
    Write-Output "PDF generado: $rutaPdf"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
