# -*- coding: utf-8 -*-
"""
Builds the GroundVision ML thesis from the college template.

  template : paperwork/Final Thesis Format New final-2.docx   (front matter + logo kept as-is)
  content  : paperwork/thesis_content.py
    output   : paperwork/GroundVision_ML_Thesis_raw.docx  (then refresh_fields.ps1 -> GroundVision_ML_Thesis.docx)

Names of people (students, guide, HOD, principal) are LEFT AS THE TEMPLATE'S PLACEHOLDERS.
Run: python paperwork/build_thesis.py
"""

import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_content as C  # noqa: E402

TEMPLATE = HERE / "Final Thesis Format New final-2.docx"
OUT = HERE / "GroundVision_ML_Thesis_raw.docx"
FIG = HERE / "figures"

TITLE = "GroundVision ML: Machine Learning-Based Groundwater Level Forecasting and Risk Analysis"
DEPT = "Artificial Intelligence and Data Science"
DEPT_UP = DEPT.upper()

doc = Document(str(TEMPLATE))
P = list(doc.paragraphs)  # snapshot of original paragraphs (indices from template inspection)


# ------------------------------------------------------------------ run helpers --
def para_text(p):
    return "".join(r.text for r in p.runs)


def replace_in_par(p, old, new):
    """Replace `old` with `new` even if it spans several runs; formatting of the first
    run touched is kept."""
    runs = p.runs
    full = "".join(r.text for r in runs)
    i = full.find(old)
    if i < 0:
        return False
    j = i + len(old)
    pos = 0
    first = True
    for r in runs:
        a, b = pos, pos + len(r.text)
        pos = b
        if b <= i or a >= j:
            continue
        s, e = max(i, a) - a, min(j, b) - a
        if first:
            r.text = r.text[:s] + new + r.text[e:]
            first = False
        else:
            r.text = r.text[:s] + r.text[e:]
    return True


def set_par_text(p, text):
    """Replace all text keeping the first run's formatting."""
    if not p.runs:
        p.add_run(text)
        return
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


def clear_par(p):
    for r in p.runs:
        r.text = ""


def fix_mojibake(p):
    """The template has U+FFFD where curly quotes/apostrophes were lost."""
    for r in p.runs:
        pass
    full = "".join(r.text for r in p.runs)
    if "�" not in full:
        return
    # decide replacement per occurrence using neighbours in the joined text
    out = []
    for k, ch in enumerate(full):
        if ch != "�":
            out.append(ch)
            continue
        prev = full[k - 1] if k else " "
        nxt = full[k + 1] if k + 1 < len(full) else " "
        if prev.isalpha() and nxt == "s":
            out.append("’")
        elif prev in " -\t\n(" or k == 0:
            out.append("“")
        else:
            out.append("”")
    new = "".join(out)
    # write back run by run, preserving run lengths
    pos = 0
    for r in p.runs:
        n = len(r.text)
        r.text = new[pos:pos + n]
        pos += n


def delete_par(p):
    el = p._p
    el.getparent().remove(el)


def has_sect(p):
    return bool(p._p.xpath("./w:pPr/w:sectPr"))


# --------------------------------------------------------------- front matter edits --
for p in P:
    fix_mojibake(p)
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                fix_mojibake(p)

# Title page
set_par_text(P[0], "“" + TITLE + "”")
for i in (1, 3, 5, 8, 14):           # "<Font Size ..>" instruction placeholders
    clear_par(P[i])
set_par_text(P[7], DEPT_UP)
replace_in_par(P[10], "OFENGINEERING", "OF ENGINEERING")
replace_in_par(P[11], "Affilliated", "Affiliated")
set_par_text(P[16], "Name and designation")
set_par_text(P[19], "Submitted by")
replace_in_par(P[21], "<Font Size 16>", "")
replace_in_par(P[25], "------------------------", DEPT_UP)
replace_in_par(P[27], "ENGINEERING,", "ENGINEERING,")

# Certificate
replace_in_par(P[84], "COLLGE", "COLLEGE")
replace_in_par(P[86], "-----------------------", DEPT)
replace_in_par(P[93], "-“PROJECT TITLE”", "“" + TITLE + "”")
replace_in_par(P[93], "“PROJECT TITLE”", "“" + TITLE + "”")
replace_in_par(P[94], "Affilliated", "Affiliated")
replace_in_par(P[94], "(Name of Department)", DEPT)
replace_in_par(P[94], "(Name of ", "(")
replace_in_par(P[94], "Department)  for", DEPT + ")  for")
replace_in_par(P[115], "COLLGE", "COLLEGE")
replace_in_par(P[116], "______________________", DEPT)

# Declaration
replace_in_par(P[120], "“PROJECT TITLE”", "“" + TITLE + "”")
replace_in_par(P[120], "(Name of Department)", DEPT)
replace_in_par(P[120], "Affilliated", "Affiliated")
replace_in_par(P[120], "(Name of ", "(")
replace_in_par(P[120], "Department)", DEPT + ")")

# Acknowledgement (names stay placeholders)
replace_in_par(P[133], "Proffessor", "Professor")
replace_in_par(P[133], "_____________ Department", DEPT + " Department")
replace_in_par(P[139], "_________________________", DEPT)
for ph in ("______________________,",):
    pass

# Abstract page: remove blank spacer paragraphs, then insert text
for i in list(range(147, 157)) + list(range(158, 198)):
    if not has_sect(P[i]):
        delete_par(P[i])
P[157].paragraph_format.page_break_before = True
P[157].alignment = WD_ALIGN_PARAGRAPH.CENTER


def new_par_after(ref, text="", bold=False, italic=False, align=None, size=12, space_after=6,
                  line=1.5, left=None, first=None, font="Times New Roman"):
    p = copy.deepcopy(ref._p)
    for ch in list(p):
        if ch.tag != qn("w:pPr"):
            p.remove(ch)
    ppr = p.find(qn("w:pPr"))
    if ppr is not None:
        for tag in ("w:sectPr", "w:pageBreakBefore", "w:numPr", "w:ind"):
            for el in ppr.findall(qn(tag)):
                ppr.remove(el)
    ref._p.addnext(p)
    from docx.text.paragraph import Paragraph
    para = Paragraph(p, ref._parent)
    if align is not None:
        para.alignment = align
    pf = para.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = line
    if left is not None:
        pf.left_indent = Inches(left)
    if first is not None:
        pf.first_line_indent = Inches(first)
    if text:
        add_rich(para, text, size=size, bold=bold, italic=italic, font=font)
    return para


def set_font(run, size=12, bold=None, italic=None, font="Times New Roman", color=None):
    run.font.name = font
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), font)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


TOKEN = re.compile(r"(\*\*.+?\*\*|__.+?__|_\{.+?\})")


def add_rich(para, text, size=12, bold=False, italic=False, font="Times New Roman"):
    for part in TOKEN.split(text):
        if not part:
            continue
        b, it = bold, italic
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        elif part.startswith("__") and part.endswith("__"):
            part, it = part[2:-2], True
        sub = False
        if part.startswith("_{") and part.endswith("}"):
            part, sub = part[2:-1], True
        r = para.add_run(part)
        set_font(r, size=size, bold=b, italic=it, font=font)
        if sub:
            r.font.subscript = True


ref = P[157]
last = ref
for k, txt in enumerate(C.ABSTRACT):
    last = new_par_after(ref if k == 0 else last, txt, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
kw = new_par_after(last, "", align=WD_ALIGN_PARAGRAPH.JUSTIFY)
add_rich(kw, "**Keywords:** " + C.KEYWORDS)

# Index page: replace template list with the real outline
for i in range(199, 222):
    if not has_sect(P[i]):
        delete_par(P[i])
P[198].paragraph_format.page_break_before = True
P[198].alignment = WD_ALIGN_PARAGRAPH.CENTER
cur = P[198]
for chap, subs in C.INDEX:
    cur = new_par_after(cur, chap, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=1, line=1.0, size=11)
    cur.paragraph_format.space_before = Pt(4)
    for s in subs:
        cur = new_par_after(cur, s, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0, line=1.0, left=0.4, size=10)

# ------------------------------------------------------------------ styles --
styles = doc.styles


def make_style(name, size, bold, align, before, after, line=1.5, italic=False, keep_next=False,
               outline=None, font="Times New Roman"):
    from docx.enum.style import WD_STYLE_TYPE
    try:
        st = styles[name]
    except KeyError:
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = styles["Normal"]
    st.font.name = font
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.italic = italic
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), font)
    pf = st.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    pf.keep_with_next = keep_next
    if outline is not None:
        ppr = st.element.get_or_add_pPr()
        ol = OxmlElement("w:outlineLvl")
        ol.set(qn("w:val"), str(outline))
        ppr.append(ol)
    return st


make_style("Thesis Body", 12, False, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 6)
make_style("Chapter Heading", 16, True, WD_ALIGN_PARAGRAPH.CENTER, 0, 18, line=1.15, keep_next=True, outline=0)
make_style("Section Heading", 14, True, WD_ALIGN_PARAGRAPH.LEFT, 14, 6, line=1.15, keep_next=True, outline=1)
make_style("Subsection Heading", 12, True, WD_ALIGN_PARAGRAPH.LEFT, 10, 4, line=1.15, keep_next=True, outline=2)
make_style("Figure Caption", 11, False, WD_ALIGN_PARAGRAPH.CENTER, 4, 12, line=1.1, italic=True)
make_style("Table Caption", 11, True, WD_ALIGN_PARAGRAPH.CENTER, 10, 4, line=1.1, keep_next=True)
make_style("Thesis Code", 9, False, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, line=1.0, font="Consolas")
make_style("Front Heading", 14, True, WD_ALIGN_PARAGRAPH.CENTER, 0, 12, line=1.15)

# ------------------------------------------------------------------ TOC section --
toc_head = P[223]
toc_head.paragraph_format.page_break_before = False
toc_head.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_field(para, instr, placeholder="Right-click and choose Update Field."):
    r = para.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r._r.append(f1)
    r2 = para.add_run()
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = " " + instr + " "; r2._r.append(it)
    r3 = para.add_run()
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate"); r3._r.append(f2)
    r4 = para.add_run(placeholder)
    r5 = para.add_run()
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end"); r5._r.append(f3)


def body_par(style="Thesis Body", text="", **kw):
    p = doc.add_paragraph(style=style)
    if text:
        add_rich(p, text, size=kw.get("size", 12), bold=kw.get("bold", False), italic=kw.get("italic", False))
    return p


def page_break():
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    return p


toc_p = body_par("Normal")
add_field(toc_p, 'TOC \\h \\z \\t "Chapter Heading,1,Section Heading,2,Subsection Heading,3"')

pb = page_break()
h = body_par("Front Heading", "LIST OF FIGURES")
fl = body_par("Normal"); add_field(fl, 'TOC \\h \\z \\t "Figure Caption,1"')
page_break()
h = body_par("Front Heading", "LIST OF TABLES")
tl = body_par("Normal"); add_field(tl, 'TOC \\h \\z \\t "Table Caption,1"')

# new section for the body
sec_prev = doc.sections[-1]
body_sec = doc.add_section(WD_SECTION.NEW_PAGE)
for s in (doc.sections[-2], body_sec):
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(0.95)
    s.left_margin = Inches(1.25)
    s.right_margin = Inches(1.0)
    s.footer_distance = Inches(0.45)
# restart page numbering at 1 for the body, number in centre footer
sectPr = body_sec._sectPr
pg = sectPr.find(qn("w:pgNumType"))
if pg is None:
    pg = OxmlElement("w:pgNumType")
    sectPr.append(pg)
pg.set(qn("w:start"), "1")
body_sec.footer.is_linked_to_previous = False
fp = body_sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in list(fp.runs):
    r.text = ""
rr = fp.add_run()
for typ, txt in (("begin", None), (None, " PAGE "), ("end", None)):
    if typ:
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), typ); rr._r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; rr._r.append(it)
set_font(rr, size=11)

# ------------------------------------------------------------------ body builders --
fig_counter = {}
tbl_counter = {}


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def set_borders(table, color="444444", sz=4):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement("w:" + edge)
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def cell_margins(table, top=40, bottom=40, left=70, right=70):
    tblPr = table._tbl.tblPr
    m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement("w:" + k); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    tblPr.append(m)


def add_table(caption, header, rows, widths):
    cp = body_par("Table Caption", caption)
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_borders(t)
    cell_margins(t)
    for i, htxt in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        pr = c.paragraphs[0]
        pr.paragraph_format.space_after = Pt(0)
        pr.paragraph_format.line_spacing = 1.0
        add_rich(pr, htxt, size=10, bold=True)
        shade(c, "E7E6E6")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            pr = cells[i].paragraphs[0]
            pr.paragraph_format.space_after = Pt(0)
            pr.paragraph_format.line_spacing = 1.0
            add_rich(pr, str(v), size=10)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    if len(rows) <= 10:
        for row in t.rows[:-1]:
            for cell in row.cells:
                for pp in cell.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    body_par("Normal").paragraph_format.space_after = Pt(6)


def add_figure(fname, caption, width):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(FIG / fname), width=Inches(width))
    body_par("Figure Caption", caption)


def add_note(text):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t, color="B58900", sz=8)
    cell_margins(t, 80, 80, 120, 120)
    c = t.rows[0].cells[0]
    c.width = Inches(6.0)
    shade(c, "FFF8E1")
    pr = c.paragraphs[0]
    pr.paragraph_format.line_spacing = 1.15
    pr.paragraph_format.space_after = Pt(0)
    add_rich(pr, text, size=10.5, italic=False)
    body_par("Normal").paragraph_format.space_after = Pt(6)


def add_list(items, numbered):
    for n, it in enumerate(items, 1):
        p = body_par("Thesis Body")
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(4)
        add_rich(p, (f"{n}.  " if numbered else "•  ") + it)


def add_code(text):
    lines = text.split("\n")
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t, color="BBBBBB", sz=4)
    cell_margins(t, 60, 60, 100, 100)
    c = t.rows[0].cells[0]
    c.width = Inches(6.0)
    shade(c, "F5F5F5")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
    first = True
    for ln in lines:
        pr = c.paragraphs[0] if first else c.add_paragraph()
        first = False
        pr.style = styles["Thesis Code"]
        r = pr.add_run(ln if ln else " ")
        set_font(r, size=8.5, font="Consolas")
    body_par("Normal").paragraph_format.space_after = Pt(6)


def chapter_number_from(title_text):
    return None


chapter_idx = 0
for item in C.BODY:
    kind = item[0]
    if kind == "ch":
        chapter_idx += 1
        p = doc.add_paragraph(style="Chapter Heading")
        p.paragraph_format.page_break_before = True
        r = p.add_run(f"CHAPTER {chapter_idx}")
        set_font(r, size=16, bold=True)
        r.add_break()
        r2 = p.add_run(item[1])
        set_font(r2, size=16, bold=True)
    elif kind == "h2":
        p = doc.add_paragraph(style="Section Heading")
        set_font(p.add_run(item[1]), size=14, bold=True)
    elif kind == "h3":
        p = doc.add_paragraph(style="Subsection Heading")
        set_font(p.add_run(item[1]), size=12, bold=True)
    elif kind == "p":
        body_par("Thesis Body", item[1])
    elif kind == "bul":
        add_list(item[1], False)
    elif kind == "num":
        add_list(item[1], True)
    elif kind == "eq":
        p = body_par("Thesis Body")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_rich(p, item[1], size=12, italic=True)
    elif kind == "fig":
        add_figure(item[1], item[2], item[3])
    elif kind == "tbl":
        add_table(item[1], item[2], item[3], item[4])
    elif kind == "note":
        add_note(item[1])
    elif kind == "code":
        add_code(item[1])
    elif kind == "pb":
        page_break()

# References
p = doc.add_paragraph(style="Chapter Heading")
p.paragraph_format.page_break_before = True
set_font(p.add_run("REFERENCES"), size=16, bold=True)
for n, ref_txt in enumerate(C.REFERENCES, 1):
    p = body_par("Thesis Body")
    p.paragraph_format.left_indent = Inches(0.45)
    p.paragraph_format.first_line_indent = Inches(-0.45)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_rich(p, f"[{n}]  " + ref_txt)
add_note(C.REF_NOTE)

# Appendix
p = doc.add_paragraph(style="Chapter Heading")
p.paragraph_format.page_break_before = True
set_font(p.add_run("APPENDIX"), size=16, bold=True)
p = doc.add_paragraph(style="Section Heading"); set_font(p.add_run("A. REPOSITORY LAYOUT AND RUN COMMANDS"), size=14, bold=True)
body_par("Thesis Body", "Source code: https://github.com/aarshnomorecool/GLOWkrishna")
add_code(C.APPENDIX_LAYOUT)
add_code(C.APPENDIX_A)
p = doc.add_paragraph(style="Section Heading"); set_font(p.add_run("B. REST API SUMMARY"), size=14, bold=True)
add_table("Table A.1: API endpoints", ["Endpoint", "Description"], C.APPENDIX_B, [2.3, 3.95])
p = doc.add_paragraph(style="Section Heading"); set_font(p.add_run("C. KEY SOURCE EXCERPTS"), size=14, bold=True)
body_par("Thesis Body", "Monotonic constraints and model construction (src/features.py, src/baseline_model.py):")
add_code(C.APPENDIX_C1)
body_par("Thesis Body", "Scenario application in the Policy Sandbox (src/simulate.py, abridged):")
add_code(C.APPENDIX_C2)
for title, ph in C.APPENDIX_PLACEHOLDERS:
    p = doc.add_paragraph(style="Section Heading")
    p.paragraph_format.page_break_before = True
    set_font(p.add_run(title), size=14, bold=True)
    q = body_par("Thesis Body", ph)
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in q.runs:
        r.font.italic = True

# core properties
doc.core_properties.title = TITLE
doc.core_properties.subject = "B.Tech project report"
doc.core_properties.author = ""

doc.save(str(OUT))
print("saved", OUT)
