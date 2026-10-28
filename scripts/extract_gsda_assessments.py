"""
Extract taluka-level Dynamic Ground Water Resource Assessment figures for Nagpur district
from the three GSDA/CGWB state reports (text-readable PDFs):

    2011-12 report  : taluka tables "Total" rows (recharge table, draft table, recharge-worthy area)
    2022-23 report  : two tables (recharge ; extraction + stage + category)
    2024 report     : one wide table

Output: data/raw/extracted/gsda_assessments_nagpur.csv
  columns: tehsil_src, snapshot_year, area_ha, rain_recharge_ham, total_recharge_ham,
           extractable_ham, irrigation_ham, dom_ind_ham, total_extraction_ham, stage_pct, category

All quantities are in ham (hectare-metre) except area_ha (hectares) and stage_pct (%).
The 2011-12 report treats Nagpur Urban+Rural as one taluka ("Nagpur").

Run: python scripts/extract_gsda_assessments.py [--src D:/NAKSHA/research-paper]
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR  # noqa: E402

DEFAULT_SRC = Path("D:/NAKSHA/research-paper")
OUT = DATA_DIR / "raw" / "extracted" / "gsda_assessments_nagpur.csv"
NUM = re.compile(r"^-?\d+(\.\d+)?$")
CATS = {"safe", "semi_critical", "semi-critical", "critical", "over_exploited", "over-exploited",
        "semi", "overexploited", "over exploited", "semicritical"}


def tokens_of(page):
    return [t.strip() for t in page.get_text().split("\n") if t.strip()]


def row_after(tokens, i, stop_on_next_row=True):
    """numbers (and a trailing category word) following index i, until the next serial-number row."""
    nums, cat = [], None
    j = i
    n = len(tokens)
    while j < n:
        t = tokens[j]
        if NUM.match(t):
            # serial number of the next row? -> int followed by an alphabetic token
            if stop_on_next_row and re.fullmatch(r"\d+", t) and j + 1 < n and re.match(r"^[A-Za-z]", tokens[j + 1]) and nums:
                break
            nums.append(float(t))
        elif t.lower().replace(" ", "_") in {c.replace(" ", "_") for c in CATS}:
            cat = t.lower()
            j += 1
            break
        elif t in ("-", "No", "Yes"):
            pass
        else:
            break
        j += 1
    return nums, cat


def parse_2011(doc):
    """Returns (recharge rows, draft rows, recharge-worthy area) keyed by taluka name."""
    rec, drf, area = {}, {}, {}
    for pno in range(len(doc)):
        toks = tokens_of(doc[pno])
        if not any("Nagpur" in t for t in toks):
            continue
        for i, t in enumerate(toks):
            is_nagpur = (t == "Nagpur") or bool(re.fullmatch(r"\d+\s+Nagpur", t))
            if not is_nagpur or i + 3 >= len(toks):
                continue
            name = toks[i + 1]
            if not re.fullmatch(r"[A-Za-z ]+", name):
                continue
            if toks[i + 2] == "Total":
                nums, _ = row_after(toks, i + 3)
                nums = nums[:7]          # a page number can trail the last row on a page
                if len(nums) == 7:
                    if abs(100 * nums[3] / max(nums[0], 1e-9) - nums[6]) < 1.5:
                        drf[name] = nums     # net avail, irrigation, dom+ind, total draft, provision, future, stage
                    elif abs(sum(nums[:4]) - nums[4]) < 1.0:
                        rec[name] = nums     # rain-mon, oth-mon, rain-nm, oth-nm, total, nat.disch, net avail
            elif toks[i + 2] in ("Hardrock", "Alluvial", "Hard Rock", "Softrock", "Soft rock"):
                nums, _ = row_after(toks, i + 3)
                if len(nums) >= 4:
                    area[name] = nums[3]     # geographical, hilly, poor-quality, recharge-worthy
    return rec, drf, area


def build_2011(src):
    doc = pymupdf.open(src / "1583759068_Talukawise_GWA2011-12_compressed.pdf")
    rec, drf, area = parse_2011(doc)
    rows = []
    for name in sorted(set(rec) | set(drf)):
        r, d = rec.get(name), drf.get(name)
        rows.append({
            "tehsil_src": name, "snapshot_year": 2011, "area_ha": area.get(name),
            "rain_recharge_ham": (r[0] + r[2]) if r else None,
            "total_recharge_ham": r[4] if r else None,
            "extractable_ham": r[6] if r else (d[0] if d else None),
            "irrigation_ham": d[1] if d else None, "dom_ind_ham": d[2] if d else None,
            "total_extraction_ham": d[3] if d else None, "stage_pct": d[6] if d else None, "category": None,
        })
    return rows


def parse_wide(doc, year, label):
    """2022-23 (two tables) and 2024 (one table)."""
    rows = {}
    for pno in range(len(doc)):
        toks = tokens_of(doc[pno])
        if not any(t.upper() == "NAGPUR" for t in toks):
            continue
        for i, t in enumerate(toks):
            if t.upper() == "NAGPUR" and i + 2 < len(toks) and i >= 1 and re.fullmatch(r"\d+|MH\d+", toks[i - 1]):
                name = toks[i + 1]
                if not re.fullmatch(r"[A-Za-z ]+", name):
                    continue
                nums, cat = row_after(toks, i + 2)
                rows.setdefault(name.upper(), []).append((pno + 1, nums, cat))
    return rows


def build_2022(src):
    doc = pymupdf.open(src / "GWRE-State-Report-2022-23_compressed.pdf")
    rows = parse_wide(doc, 2023, "2022-23")
    out = []
    for name, parts in rows.items():
        rec = next((p for p in parts if len(p[1]) == 8), None)       # area + 4 recharge + total + discharge + extractable
        ext = next((p for p in parts if len(p[1]) == 9 and p[2]), None)
        if rec is None and ext is None:
            out.append({"tehsil_src": name, "snapshot_year": 2023, "_raw": parts})
            continue
        r = rec[1] if rec else None
        e = ext[1] if ext else None
        out.append({
            "tehsil_src": name, "snapshot_year": 2023,
            "area_ha": r[0] if r else (e[0] if e else None),
            "rain_recharge_ham": (r[1] + r[3]) if r else None,
            "total_recharge_ham": r[5] if r else None,
            "extractable_ham": r[7] if r else (e[1] if e else None),
            "irrigation_ham": e[2] if e else None, "dom_ind_ham": (e[3] + e[4]) if e else None,
            "total_extraction_ham": e[5] if e else None, "stage_pct": e[8] if e else None,
            "category": ext[2] if ext else None,
        })
    return out


def build_2024(src):
    doc = pymupdf.open(src / "Maharashtra-GWRE-2024.pdf")
    rows = parse_wide(doc, 2024, "2024")
    out = []
    for name, parts in rows.items():
        p = max(parts, key=lambda x: len(x[1]))
        n = p[1]
        if len(n) < 16:
            out.append({"tehsil_src": name, "snapshot_year": 2024, "_raw": parts})
            continue
        # geo area, RW area, rainMON, othMON, rainNM, othNM, total, natdisch, extractable,
        # irrigation, industrial, domestic, total extraction, allocation, net avail, stage
        out.append({
            "tehsil_src": name, "snapshot_year": 2024, "area_ha": n[1],
            "rain_recharge_ham": n[2] + n[4], "total_recharge_ham": n[6], "extractable_ham": n[8],
            "irrigation_ham": n[9], "dom_ind_ham": n[10] + n[11], "total_extraction_ham": n[12],
            "stage_pct": n[15], "category": p[2],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    src = Path(ap.parse_args().src)
    rows = build_2011(src) + build_2022(src) + build_2024(src)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns if c != "_raw"]
    df[cols].to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df[cols].sort_values(["snapshot_year", "tehsil_src"]).to_string())
    bad = df[df.get("_raw").notna()] if "_raw" in df else []
    if len(bad):
        print("\nUNPARSED:", bad[["tehsil_src", "snapshot_year", "_raw"]].to_string())


if __name__ == "__main__":
    main()
