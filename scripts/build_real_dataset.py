"""
Build the REAL Nagpur tehsil-season dataset that replaces the synthetic panel.

Inputs (all produced by earlier scripts, nothing is invented here):
  data/raw/extracted/gsda_wl_nagpur.csv          scripts/extract_gsda_waterlevels.py   (GSDA/CGWB well readings)
  data/raw/extracted/gsda_assessments_nagpur.csv scripts/extract_gsda_assessments.py   (GWRE 2011-12, 2023, 2024)
  data/raw/nasa_power_climate.csv                scripts/fetch_nasa_power.py           (NASA POWER rain + temperature)
  data/geo/nagpur_tehsils.geojson                tehsil polygons

Output: data/processed/nagpur_blocks.csv  (+ the old synthetic panel is kept as nagpur_blocks_synthetic.csv)

Columns
  tehsil, year, season
  rainfall_mm, temperature_c              NASA POWER, season totals/means
  recharge_proxy                          rain-recharge depth (mm) = tehsil infiltration fraction x season rainfall
  irrigation_draft_mm                     GEC irrigation groundwater draft / recharge-worthy area (mm), interpolated
  domestic_industrial_draft_mm            GEC domestic+industrial draft / recharge-worthy area (mm), interpolated
  stage_of_extraction_pct                 reported stage of groundwater extraction (%), interpolated
  observed_water_table_m                  GSDA well readings, well-fixed-effect adjusted median (m below ground);
                                          missing rounds gap-filled from neighbouring measurements (see fill_gaps)
  gap_filled                              True where the value was gap-filled (kept out of training and scoring)
  n_wells                                 number of wells behind the value (0 for gap-filled); not a model input
  rainfall_mm_lag1, observed_water_table_m_lag1   previous period in chronological order

Season <-> measurement month: Winter=Jan, Pre-Monsoon=May, Monsoon=Aug, Post-Monsoon=Nov.

Run: python scripts/build_real_dataset.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, GEO_DIR, PROCESSED_DIR, TEHSIL_NAMES, YEAR_END, YEAR_START  # noqa: E402

RAW = DATA_DIR / "raw"
EXT = RAW / "extracted"
OUT = PROCESSED_DIR / "nagpur_blocks.csv"
SYN_BACKUP = PROCESSED_DIR / "nagpur_blocks_synthetic.csv"

MONTH_TO_SEASON = {1: "Winter", 5: "Pre-Monsoon", 8: "Monsoon", 11: "Post-Monsoon"}
CHRONO = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]       # order within a calendar year
BLOCK_TO_TEHSIL = {
    "Bhivapur": "Bhiwapur", "Hingna": "Hingna", "Kalmeshwar": "Kalmeshwar", "Kamptee": "Kamptee",
    "Katol": "Katol", "Kuhi": "Kuhi", "Mouda": "Mouda", "Narkhed": "Narkhed", "Parseoni": "Parseoni",
    "Ramtek": "Ramtek", "Saoner": "Savner", "Umred": "Umred",
}
ASSESS_TO_TEHSIL = {
    "BHIWAPUR": ["Bhiwapur"], "HINGANA": ["Hingna"], "KALAMESHWAR": ["Kalmeshwar"], "KAMTEE": ["Kamptee"],
    "KATOL": ["Katol"], "KUHI": ["Kuhi"], "MOUDA": ["Mouda"], "NARKHED": ["Narkhed"], "PARSHIONI": ["Parseoni"],
    "RAMTEK": ["Ramtek"], "SAONER": ["Savner"], "UMRED": ["Umred"],
    "NAGPUR": ["Nagpur Urban", "Nagpur Rural"],          # 2011-12 and 2023 reports list Nagpur as one taluka
    "NAGPUR RURAL": ["Nagpur Rural"], "NAGPUR URBAN": ["Nagpur Urban"],
}
MIN_READINGS_PER_WELL = 5
MIN_WELLS_PER_CELL = 2


def load_polygons():
    g = json.loads((GEO_DIR / "nagpur_tehsils.geojson").read_text())
    return {f["properties"]["tehsil"]: shape(f["geometry"]) for f in g["features"]}


def assign_tehsil(df, polys):
    tehsil = df["block"].map(BLOCK_TO_TEHSIL)
    # GSDA lists Nagpur Urban + Rural as one block "Nagpur": split by well location.
    is_ng = df["block"] == "Nagpur"
    ng_polys = {k: polys[k] for k in ("Nagpur Urban", "Nagpur Rural")}

    def locate(lon, lat):
        p = Point(lon, lat)
        for k, poly in ng_polys.items():
            if poly.contains(p):
                return k
        return min(ng_polys, key=lambda k: ng_polys[k].distance(p))      # boundary wells -> nearest

    tehsil.loc[is_ng] = [locate(lo, la) for lo, la in zip(df.loc[is_ng, "lon"], df.loc[is_ng, "lat"])]
    return tehsil


def well_levels(wl):
    """Adjusted tehsil-season-year water level that is robust to the changing well network."""
    wl = wl.copy()
    wl["season"] = wl["month"].map(MONTH_TO_SEASON)
    wl["well"] = (wl["tehsil"] + "|" + wl["village"].astype(str) + "|"
                  + wl["lat"].round(3).astype(str) + "|" + wl["lon"].round(3).astype(str))
    # one value per well-year-season (duplicate readings averaged)
    wl = wl.groupby(["tehsil", "well", "season", "year"], as_index=False)["wl_mbgl"].mean()
    counts = wl.groupby(["well", "season"])["year"].transform("count")
    wl = wl[counts >= MIN_READINGS_PER_WELL].copy()                          # wells with a usable history
    wl["mu"] = wl.groupby(["well", "season"])["wl_mbgl"].transform("mean")
    wl["anom"] = wl["wl_mbgl"] - wl["mu"]
    base = wl.groupby(["tehsil", "season", "well"])["mu"].first().groupby(["tehsil", "season"]).median()
    cell = wl.groupby(["tehsil", "season", "year"]).agg(
        anom=("anom", "median"), n_wells=("well", "nunique"), wl_raw_median_m=("wl_mbgl", "median")
    ).reset_index()
    cell = cell[cell["n_wells"] >= MIN_WELLS_PER_CELL]
    cell["observed_water_table_m"] = [
        base.loc[(t, s)] + a for t, s, a in zip(cell["tehsil"], cell["season"], cell["anom"])
    ]
    return cell.drop(columns="anom")


def assessment_features(climate):
    a = pd.read_csv(EXT / "gsda_assessments_nagpur.csv")
    a["key"] = a["tehsil_src"].str.upper().str.strip()
    # per-taluka areas: the 2011-12 recharge-worthy-area column was unreliable, so denominators come from
    # the later reports (same taluka), falling back to the snapshot's own area.
    area_ref = a[a["snapshot_year"] >= 2023].groupby("key")["area_ha"].median()
    rows = []
    for _, r in a.iterrows():
        tehs = ASSESS_TO_TEHSIL.get(r["key"])
        if not tehs:
            continue
        area = area_ref.get(r["key"], r["area_ha"]) if r["snapshot_year"] == 2011 else r["area_ha"]
        for t in tehs:
            rows.append({
                "tehsil": t, "snapshot_year": int(r["snapshot_year"]),
                "rain_recharge_mm": r["rain_recharge_ham"] / area * 1000.0,
                "irrigation_draft_mm": r["irrigation_ham"] / area * 1000.0,
                "domestic_industrial_draft_mm": r["dom_ind_ham"] / area * 1000.0,
                "stage_of_extraction_pct": r["stage_pct"],
            })
    snap = pd.DataFrame(rows)
    # for Nagpur Urban/Rural prefer their own 2024 rows; 2011/2023 share the combined taluka value
    snap = snap.drop_duplicates(["tehsil", "snapshot_year"], keep="last")

    annual_rain = climate.groupby(["tehsil", "year"])["rainfall_mm"].sum().groupby("tehsil").mean()
    years = np.arange(YEAR_START, YEAR_END + 1)
    out = []
    for t, g in snap.groupby("tehsil"):
        g = g.sort_values("snapshot_year")
        infil = (g["rain_recharge_mm"] / annual_rain.loc[t]).mean()          # rainfall infiltration fraction
        d = {"tehsil": t, "infiltration_fraction": infil}
        for col in ("irrigation_draft_mm", "domestic_industrial_draft_mm", "stage_of_extraction_pct"):
            d[col] = np.interp(years, g["snapshot_year"], g[col])
        for i, y in enumerate(years):
            out.append({"tehsil": t, "year": int(y), "infiltration_fraction": infil,
                        **{c: d[c][i] for c in ("irrigation_draft_mm", "domestic_industrial_draft_mm",
                                                  "stage_of_extraction_pct")}})
    return pd.DataFrame(out), snap


def fill_gaps(grid):
    """Fill missing water-level cells automatically from REAL neighbouring measurements.

    fill(t, s, y) = mean level of tehsil t in season s  +  district anomaly of round (s, y)

    where the district anomaly is the median, over the tehsils that WERE measured in that round, of
    (level - that tehsil-season mean). A round with fewer than 3 measured tehsils (e.g. May 2020, May 2021,
    Nov 2024) takes its anomaly by linear interpolation along the same season's earlier/later rounds.
    The original cells are untouched; filled cells are recorded in the boolean column `gap_filled`,
    which keeps the model from being trained or scored on its own gap-fills."""
    g = grid.copy()
    g["gap_filled"] = g["observed_water_table_m"].isna()
    base = g.groupby(["tehsil", "season"])["observed_water_table_m"].transform("mean")
    anom = g["observed_water_table_m"] - base
    rnd = anom.groupby([g["season"], g["year"]]).agg(["median", "count"])
    rnd.loc[rnd["count"] < 3, "median"] = np.nan
    rnd_anom = {}
    for season, sub in rnd.groupby(level="season"):
        ser = sub["median"].droplevel("season").reindex(range(YEAR_START, YEAR_END + 1))
        rnd_anom[season] = ser.interpolate(limit_direction="both")
    fill_anom = pd.Series([rnd_anom[se].loc[y] for se, y in zip(g["season"], g["year"])], index=g.index)
    fill = base + fill_anom
    g["observed_water_table_m"] = g["observed_water_table_m"].fillna(fill)
    g["n_wells"] = g["n_wells"].fillna(0)
    return g


def main():
    polys = load_polygons()
    wl = pd.read_csv(EXT / "gsda_wl_nagpur.csv")
    n0 = len(wl)
    wl = wl[(wl["wl_mbgl"] > 0) & (wl["wl_mbgl"] <= 60)].copy()
    print(f"well readings: {n0} -> {len(wl)} after dropping non-positive / >60 m values")
    wl["tehsil"] = assign_tehsil(wl, polys)
    # sanity: how well do labelled blocks agree with the polygons?
    lab = wl[wl["block"] != "Nagpur"]
    inside = [polys[t].contains(Point(lo, la)) for t, lo, la in zip(lab["tehsil"], lab["lon"], lab["lat"])]
    print(f"labelled-block wells inside their own tehsil polygon: {np.mean(inside):.1%}")

    cell = well_levels(wl)
    cell = cell[(cell["year"] >= YEAR_START) & (cell["year"] <= YEAR_END)]

    climate = pd.read_csv(RAW / "nasa_power_climate.csv")
    feats, snap = assessment_features(climate)

    # full chronological grid, so lags refer to the true previous period
    grid = pd.MultiIndex.from_product([TEHSIL_NAMES, range(YEAR_START, YEAR_END + 1), CHRONO],
                                      names=["tehsil", "year", "season"]).to_frame(index=False)
    grid["order"] = grid["season"].map({s: i for i, s in enumerate(CHRONO)})
    grid = grid.sort_values(["tehsil", "year", "order"]).reset_index(drop=True)
    grid = grid.merge(climate, on=["tehsil", "year", "season"], how="left")
    grid = grid.merge(feats, on=["tehsil", "year"], how="left")
    grid = grid.merge(cell[["tehsil", "season", "year", "observed_water_table_m", "n_wells", "wl_raw_median_m"]],
                      on=["tehsil", "season", "year"], how="left")
    grid["recharge_proxy"] = grid["infiltration_fraction"] * grid["rainfall_mm"]
    n_gaps = int(grid["observed_water_table_m"].isna().sum())
    grid = fill_gaps(grid)
    grid["observed_water_table_m"] = grid["observed_water_table_m"].clip(lower=0.0)   # depth cannot be negative
    grid["rainfall_mm_lag1"] = grid.groupby("tehsil")["rainfall_mm"].shift(1)
    grid["observed_water_table_m_lag1"] = grid.groupby("tehsil")["observed_water_table_m"].shift(1)

    keep = ["tehsil", "year", "season", "rainfall_mm", "temperature_c", "recharge_proxy",
            "irrigation_draft_mm", "domestic_industrial_draft_mm", "stage_of_extraction_pct",
            "observed_water_table_m", "gap_filled", "n_wells", "rainfall_mm_lag1", "observed_water_table_m_lag1"]
    full = grid[keep]
    missing_target = n_gaps
    df = full.dropna(subset=["observed_water_table_m", "rainfall_mm_lag1"]).copy()   # only each tehsil's first round lacks a lag
    df["n_wells"] = df["n_wells"].astype(int)
    df = df.round({"rainfall_mm": 1, "temperature_c": 2, "recharge_proxy": 2, "irrigation_draft_mm": 2,
                   "domestic_industrial_draft_mm": 2, "stage_of_extraction_pct": 2,
                   "observed_water_table_m": 3, "rainfall_mm_lag1": 1,
                   "observed_water_table_m_lag1": 3})

    if OUT.exists() and not SYN_BACKUP.exists():
        shutil.copy(OUT, SYN_BACKUP)
        print(f"kept synthetic panel as {SYN_BACKUP.name}")
    df.to_csv(OUT, index=False)
    snap.to_csv(EXT / "assessment_snapshots_by_tehsil.csv", index=False)

    print(f"\nwrote {len(df)} rows -> {OUT}")
    print(f"grid cells {len(full)}; {missing_target} had no usable well reading and were gap-filled from neighbouring measurements")
    print("gap-filled rows kept:", int(df["gap_filled"].sum()), "| measured rows:", int((~df["gap_filled"]).sum()))
    print("\nrows per tehsil:"); print(df.groupby("tehsil").size().to_string())
    print("\nrows per year:"); print(df.groupby("year").size().to_string())
    print("\nwells per cell (n_wells): median", df["n_wells"].median(), " min", df["n_wells"].min())
    print("\nobserved_water_table_m"); print(df["observed_water_table_m"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
