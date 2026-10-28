"""
Extract Nagpur-district well readings from the GSDA/CGWB state-wide water-level PDFs.

Each PDF is a long table with the columns
    State | District | Block | Village | Latitude | Longitude | Date | WL(mbgl)
printed one value per line.  We scan every page, keep rows where State == Maharashtra and
District == Nagpur, and write one tidy CSV.

Input : the PDFs in SRC_DIR (override with --src)
Output: data/raw/extracted/gsda_wl_nagpur.csv
        columns: season_file, block, village, lat, lon, date, year, month, wl_mbgl

Run: python scripts/extract_gsda_waterlevels.py [--src D:/NAKSHA/research-paper]
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR  # noqa: E402

DEFAULT_SRC = Path("D:/NAKSHA/research-paper")
FILES = {
    "January": "January_WL_1994-2025.pdf",
    "PreMonsoon_2004_2013": "pre-monsoon_2004-2013.pdf",
    "PreMonsoon_2014_2024": "pre-monsoon_2014-2024.pdf",
    "August": "August_WL_1994-2025.pdf",
    "PostMonsoon": "post-monsoon_wl_1994-2023_compressed.pdf",
}
OUT = DATA_DIR / "raw" / "extracted" / "gsda_wl_nagpur.csv"
DATE_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
FLOAT_RE = re.compile(r"^-?\d+(\.\d+)?$")


def parse_date(s: str):
    s = s.replace("/", "-")
    for fmt in ("%d-%m-%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def rows_from_tokens(tokens, tag):
    out = []
    i = 0
    n = len(tokens)
    while i < n - 1:
        if tokens[i] == "Maharashtra" and tokens[i + 1] == "Nagpur":
            # State, District, Block, Village, lat, lon, date, wl  (village can wrap across lines)
            j = i + 2
            block = tokens[j]
            # find the date token within the next few lines; everything between block and lat is village
            k = j + 1
            village_parts = []
            while k < n and not FLOAT_RE.match(tokens[k]):
                village_parts.append(tokens[k])
                k += 1
            if k + 3 < n and DATE_RE.match(tokens[k + 2]) and FLOAT_RE.match(tokens[k + 3]):
                dt = parse_date(tokens[k + 2])
                if dt is not None:
                    out.append({
                        "season_file": tag, "block": block, "village": " ".join(village_parts),
                        "lat": float(tokens[k]), "lon": float(tokens[k + 1]),
                        "date": dt.date().isoformat(), "year": dt.year, "month": dt.month,
                        "wl_mbgl": float(tokens[k + 3]),
                    })
                i = k + 4
                continue
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    args = ap.parse_args()
    src = Path(args.src)
    all_rows = []
    for tag, fname in FILES.items():
        path = src / fname
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        doc = pymupdf.open(path)
        found = 0
        for pno in range(len(doc)):
            txt = doc[pno].get_text()
            if "Nagpur" not in txt:
                continue
            toks = [t.strip() for t in txt.split("\n") if t.strip()]
            rows = rows_from_tokens(toks, tag)
            found += len(rows)
            all_rows.extend(rows)
        print(f"{tag:24s} pages={len(doc):5d} nagpur_rows={found}", flush=True)
    df = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
