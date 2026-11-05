"""
PDF report generator for the dashboard's Reports view.

Every number in the report comes from the persisted models / processed tables through the same code
paths the dashboard uses (src/api.py -> src/simulate.py). Nothing here is hard-coded or invented: if a
metric file is missing, that paragraph is simply left out.

Built with reportlab (layout) and matplotlib (charts). Fonts are DejaVu Sans, which ships with
matplotlib, so signs such as the minus sign render correctly.
"""

import datetime
import io
import json
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.config import MODELS_DIR, RISK_CATEGORIES, RISK_COLORS  # noqa: E402

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(_FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

_LOCK = threading.Lock()      # matplotlib's pyplot state is not thread-safe

INK = colors.HexColor("#17313b")
MUTED = colors.HexColor("#5a6b72")
RULE = colors.HexColor("#d6dde0")

S_TITLE = ParagraphStyle("t", fontName="DV-B", fontSize=17, leading=21, textColor=INK, spaceAfter=2)
S_SUB = ParagraphStyle("s", fontName="DV", fontSize=9.5, leading=13, textColor=MUTED)
S_H = ParagraphStyle("h", fontName="DV-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=1)
S_P = ParagraphStyle("p", fontName="DV", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=4)
S_SMALL = ParagraphStyle("sm", fontName="DV", fontSize=8, leading=11, textColor=MUTED)
S_CELL = ParagraphStyle("c", fontName="DV", fontSize=8.6, leading=11.2, textColor=INK)
S_CELLB = ParagraphStyle("cb", parent=S_CELL, fontName="DV-B")
S_BUL = ParagraphStyle("b", parent=S_P, leftIndent=11, bulletIndent=2, spaceAfter=2)


def _png(fig, dpi=170):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "−")


def _signed(x, nd=2):
    return (f"{x:+.{nd}f}").replace("-", "−")


def _load_metrics():
    p = MODELS_DIR / "metrics.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------------- charts
def _drawdown_chart(drawdown, base_drawdown, scenario_changed):
    years = [int(y) for y in drawdown["years"]]
    split = drawdown["splitIndex"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.plot(years[:split], drawdown["observed"][:split], color="#17313b", lw=1.8, label="Observed (season-balanced yearly mean)")
    ax.plot(years[split - 1:], [drawdown["observed"][split - 1]] + drawdown["observed"][split:],
            color="#1f8a70", lw=1.8, ls="--", label="Projection, selected scenario" if scenario_changed else "Projection, no change")
    if scenario_changed:
        by = [int(y) for y in base_drawdown["years"]]
        bs = base_drawdown["splitIndex"]
        ax.plot(by[bs - 1:], [base_drawdown["observed"][bs - 1]] + base_drawdown["observed"][bs:],
                color="#8a97a0", lw=1.4, ls=":", label="Projection, no change")
    ax.plot(years, drawdown["baseline"], color="#c8872b", lw=1.2, ls="-.", label="Climate-only baseline")
    ax.invert_yaxis()
    ax.set_ylabel("Depth to water (m below ground)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _png(fig)


def _district_chart(rows):
    rows = sorted(rows, key=lambda r: r["residual"])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh([r["name"] for r in rows], [r["residual"] for r in rows], color=[RISK_COLORS[r["cat"]] for r in rows])
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("Modelled human-attributable residual, scenario (m)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def _attribution_chart(natural, human):
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7.0, 0.95))
    ax.barh([""], [natural], color="#5c8fd6")
    ax.barh([""], [human], left=[natural], color="#d67c5c")
    ax.set_xlim(0, 100)
    ax.axis("off")
    ax.legend(handles=[Patch(color="#5c8fd6", label=f"Natural climate {natural:.0f}%"),
                       Patch(color="#d67c5c", label=f"Human extraction {human:.0f}%")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8)
    return _png(fig)


# ---------------------------------------------------------------------------------- tables
def _kv_table(rows, widths):
    data = [[Paragraph(k, S_CELLB), Paragraph(v, S_CELL)] for k, v in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f6f7")),
    ]))
    return t


def _grid_table(header, rows, widths, cat_cols=()):
    data = [[Paragraph(h, S_CELLB) for h in header]] + [[Paragraph(str(c), S_CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef0")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for ci in cat_cols:
        for ri, r in enumerate(rows, start=1):
            if r[ci] in RISK_COLORS:
                st.append(("TEXTCOLOR", (ci, ri), (ci, ri), colors.HexColor(RISK_COLORS[r[ci]])))
    t.setStyle(TableStyle(st))
    return t


# ---------------------------------------------------------------------------------- report
def build_report_pdf(*args, **kwargs):
    with _LOCK:
        return _build_report_pdf(*args, **kwargs)


def _build_report_pdf(brand, scope_name, tehsil_id, payload, base_payload, status_df, inputs, display_names):
    """
    brand        product name shown in the header
    scope_name   "Katol" or "District-wide"
    tehsil_id    slug, or None for the district report
    payload      /api/simulate result for the chosen sliders (dict with 'tehsil' and 'all_tehsils')
    base_payload same call with all sliders at zero ("no change")
    status_df    data/processed/tehsil_status.csv (today's status)
    inputs       dict(rainfall=, drip=, borewell=)
    display_names dict slug -> display name
    """
    metrics = _load_metrics()
    scenario_changed = any(abs(float(inputs[k])) > 1e-9 for k in ("rainfall", "drip", "borewell"))
    today = datetime.date.today().strftime("%d %b %Y")
    scen_txt = (f"rainfall {inputs['rainfall']:+g}%, drip-irrigation adoption {inputs['drip']:g}%, "
                f"borewell growth {inputs['borewell']:+g}%/yr") if scenario_changed else "none (today's conditions)"

    st = status_df.set_index("tehsil")
    ranked = status_df.sort_values("latest_human_residual_m", ascending=False)["tehsil"].tolist()
    story = []
    story += [Paragraph(f"{brand}: groundwater outlook report", S_TITLE),
              Paragraph(f"{scope_name} &nbsp;|&nbsp; Nagpur district, Maharashtra &nbsp;|&nbsp; generated {today}", S_SUB),
              Spacer(1, 5 * mm)]

    W = A4[0] - 36 * mm
    if tehsil_id:
        t = payload["tehsil"]
        name = t["tehsil"]
        row = st.loc[name]
        base_t = base_payload["tehsil"]
        rank = ranked.index(name) + 1
        scen_cat = t["risk_category"]
        facts = [
            ("Risk category today", f"<b>{row['risk_category']}</b>"),
            ("Risk category, selected scenario", f"<b>{scen_cat}</b>" if scenario_changed else "same as today (no scenario applied)"),
            ("Modelled human-attributable residual (latest year)",
             f"{_signed(row['latest_human_residual_m'])} m &nbsp;(rank {rank} of {len(ranked)}; 1 = highest modelled pressure)"),
            ("Latest observed depth to water",
             f"{_fmt(row['latest_observed_water_table_m'])} m below ground (season-balanced mean, {int(row['last_year'])})"),
            ("Share of modelled change in the water table",
             f"{row['human_pct']:.0f}% attributed to groundwater extraction, {row['natural_pct']:.0f}% to climate "
             f"({int(row['first_year'])} to {int(row['last_year'])})"),
            ("Scenario assumptions", scen_txt),
        ]
        if scenario_changed:
            facts.append(("Projected depth in 10 years",
                          f"{_fmt(t['drawdown']['observed'][-1])} m with the scenario, "
                          f"{_fmt(base_t['drawdown']['observed'][-1])} m with no change"))
        story += [_kv_table(facts, [62 * mm, W - 62 * mm]), Spacer(1, 3 * mm)]

        summary = (f"{name} is in the <b>{row['risk_category']}</b> category today. Roughly {row['human_pct']:.0f}% of the "
                   f"modelled change in its water table is attributed to groundwater extraction rather than rainfall and "
                   f"temperature.")
        if scenario_changed:
            if scen_cat != row["risk_category"]:
                summary += f" Under the selected scenario the category changes to <b>{scen_cat}</b>."
            else:
                summary += " Under the selected scenario the category does not change."
            if abs(inputs["rainfall"]) > 1e-9 and abs(inputs["drip"]) < 1e-9 and abs(inputs["borewell"]) < 1e-9:
                summary += (" The category depends on the human-attributable residual only, so a rainfall change moves the "
                            "projected depth but not the category.")
        story += [Paragraph("Summary", S_H), Paragraph(summary, S_P),
                  Paragraph("Drawdown: depth to water, observed and projected", S_H),
                  Image(_drawdown_chart(t["drawdown"], base_t["drawdown"], scenario_changed), width=W, height=W * 0.43),
                  Paragraph("Lower on the chart means deeper water. The climate-only baseline is what Stage 1 predicts from "
                            "rainfall, temperature and season alone; the gap between it and the observed line is the part "
                            "attributed to human activity.", S_SMALL),
                  Paragraph("Natural versus human share", S_H),
                  Image(_attribution_chart(row["natural_pct"], row["human_pct"]), width=W, height=W * 0.135)]
    else:
        all_t = payload["all_tehsils"]
        base_all = base_payload["all_tehsils"]
        cnt = lambda src: {c: sum(1 for v in src.values() if v["risk_category"] == c) for c in RISK_CATEGORIES}
        today_cnt = {c: int((status_df["risk_category"] == c).sum()) for c in RISK_CATEGORIES}
        scen_cnt = cnt(all_t)
        story += [Paragraph("Summary", S_H),
                  Paragraph(f"Risk categories across Nagpur's 14 tehsils today: " + ", ".join(
                      f"{today_cnt[c]} {c}" for c in RISK_CATEGORIES) + f". Scenario assumptions: {scen_txt}.", S_P)]
        if scenario_changed:
            story.append(Paragraph("Under the selected scenario: " + ", ".join(f"{scen_cnt[c]} {c}" for c in RISK_CATEGORIES) + ".", S_P))
        rows = []
        for slug, v in sorted(all_t.items(), key=lambda kv: -kv[1]["human_residual_pred_m"]):
            nm = v["tehsil"]
            rows.append([display_names.get(slug, nm), st.loc[nm, "risk_category"], v["risk_category"],
                         _fmt(st.loc[nm, "latest_observed_water_table_m"]),
                         _signed(v["human_residual_pred_m"]), f"{st.loc[nm, 'human_pct']:.0f}%"])
        chart_rows = [{"name": r[0], "residual": float(all_t[s]["human_residual_pred_m"]), "cat": all_t[s]["risk_category"]}
                      for r, s in zip(rows, [k for k, _ in sorted(all_t.items(), key=lambda kv: -kv[1]["human_residual_pred_m"])])]
        story += [Paragraph("Modelled human-attributable residual by tehsil", S_H),
                  Image(_district_chart(chart_rows), width=W, height=W * 0.5),
                  Paragraph("Tehsil table", S_H),
                  _grid_table(["Tehsil", "Category today", "Category, scenario", "Depth to water (m)",
                               "Residual, scenario (m)", "Human share"], rows,
                              [34 * mm, 30 * mm, 32 * mm, 26 * mm, 30 * mm, W - 152 * mm], cat_cols=(1, 2))]

    # ---- method, accuracy, limits ------------------------------------------------------
    story += [Paragraph("How the numbers are produced", S_H),
              Paragraph("Stage 1 predicts the water table from natural drivers only (rainfall, temperature, rain-recharge, "
                        "season, tehsil). The difference between the observed water table and that prediction is modelled in "
                        "Stage 2 from groundwater-extraction figures (irrigation draft, domestic and industrial draft, stage of "
                        "extraction). Risk categories are quartile ranks of the Stage 2 output across the 14 tehsils.", S_P)]
    lim = [
        "The human-attributable residual is an <b>estimate</b>: it also absorbs model error and unmeasured natural factors. "
        "It is not a measured quantity.",
        "Risk categories are <b>relative rankings</b> of Nagpur's 14 tehsils, not the official stage-of-extraction classification.",
        "Scenario effects use assumed elasticities (full drip adoption cuts irrigation draft by 60%; each 1% of annual "
        "borewell growth raises it by 0.5%). They are scenarios, not forecasts.",
        "Extraction figures come from three assessment years (2011-12, 2023, 2024) and are interpolated between them. "
        "Water-level rounds with no readings are filled from measured neighbouring tehsils and years.",
    ]
    if metrics:
        b, r = metrics["baseline"], metrics["residual"]
        clim = metrics["benchmarks_holdout"]["tehsil_season_climatology"]
        lim.insert(0, (f"On held-out years ({metrics['split']['train_before_year']} onwards) Stage 1 has a mean absolute error of "
                       f"{b['mae']:.2f} m (R² {b['r2']:.2f}); a plain tehsil-by-season average scores R² {clim['r2']:.2f}, so "
                       f"weather adds little beyond seasonality. Stage 2 has no useful skill forward in time (R² "
                       f"{_fmt(r['r2'])}); it ranks tehsils moderately (Spearman {r['leave_one_tehsil_out']['tehsil_mean_spearman']:.2f}, "
                       f"14 tehsils)."))
    story.append(Paragraph("Accuracy and limits", S_H))
    story += [Paragraph(x, S_BUL, bulletText="•") for x in lim]
    story += [Paragraph("Data: GSDA/CGWB observation-well readings and Dynamic Ground Water Resource Assessments (2011-12, 2023, "
                        "2024); NASA POWER rainfall and temperature; Survey of India tehsil boundaries.", S_SMALL)]

    buf = io.BytesIO()

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"{brand} | {scope_name} | {today}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=17 * mm, title=f"{brand}: {scope_name} outlook", author=brand)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
