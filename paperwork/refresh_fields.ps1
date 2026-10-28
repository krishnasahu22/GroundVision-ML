# Opens the generated GroundVision ML thesis, updates its tables of contents, and exports a PDF.
$dir = $PSScriptRoot
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
$d = $w.Documents.Open("$dir\GroundVision_ML_Thesis_raw.docx")
foreach ($t in $d.TablesOfContents) { $t.Update() }
$d.SaveAs2("$dir\GroundVision_ML_Thesis.docx", 16)
$d.ExportAsFixedFormat("$dir\GroundVision_ML_Thesis.pdf", 17)
$d.Close($false)
$w.Quit()
"done"
