"""
Fetch real daily rainfall + temperature for every tehsil centroid from the
NASA POWER API (free, no key) and aggregate to the project's 4 seasons.

Output: data/raw/nasa_power_climate.csv with columns
    tehsil, year, season, rainfall_mm, temperature_c

Season month mapping (calendar-year based, matches Vidarbha climatology):
    Pre-Monsoon  Mar-May
    Monsoon      Jun-Sep
    Post-Monsoon Oct-Nov
    Winter       Dec, Jan, Feb  (all assigned to the calendar year they fall in)

Caveat: NASA POWER is a ~0.5 degree satellite/reanalysis product, not a rain-gauge
record. Treat it as a good regional signal, not tehsil-exact ground truth.

Run: python scripts/fetch_nasa_power.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, TEHSILS, YEAR_END, YEAR_START  # noqa: E402

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
SEASON_OF_MONTH = {
    3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
    12: "Winter", 1: "Winter", 2: "Winter",
}
OUT_PATH = DATA_DIR / "raw" / "nasa_power_climate.csv"


def fetch_tehsil(name: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{YEAR_START}0101",
        "end": f"{YEAR_END}1231",
        "format": "JSON",
    }
    resp = requests.get(API, params=params, timeout=120)
    resp.raise_for_status()
    p = resp.json()["properties"]["parameter"]
    df = pd.DataFrame({"temp": p["T2M"], "rain": p["PRECTOTCORR"]})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, pd.NA).dropna()  # -999 is POWER's fill value
    df["year"] = df.index.year
    df["season"] = df.index.month.map(SEASON_OF_MONTH)
    out = df.groupby(["year", "season"]).agg(
        rainfall_mm=("rain", "sum"), temperature_c=("temp", "mean")
    ).reset_index()
    out.insert(0, "tehsil", name)
    return out


def main():
    frames = []
    for name, (lat, lon) in TEHSILS.items():
        print(f"fetching {name} ({lat}, {lon}) ...", flush=True)
        frames.append(fetch_tehsil(name, lat, lon))
        time.sleep(1)  # be polite to the public API
    result = pd.concat(frames, ignore_index=True).round({"rainfall_mm": 1, "temperature_c": 2})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(result)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
