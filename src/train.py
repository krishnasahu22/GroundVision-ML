"""
Orchestrates the full two-stage pipeline end to end:

  1. Fit the climate baseline model (stage 1, natural features only), score it on a
     forward-in-time holdout, and produce cross-fitted (out-of-fold) predictions.
  2. Form the residual: observed - out-of-fold baseline prediction.
  3. Fit the residual/attribution model (stage 2) on that residual against
     anthropogenic features only — this is what the sandbox later reruns
     under perturbed slider inputs.
  4. Classify each tehsil's latest-year modelled human-attributable pressure into
     an IN-GRES-style risk category (quartile thresholds).
  5. Persist model artifacts + enriched dataframes + metrics for the dashboard and report.

Run: python -m src.train
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_model import TEST_START_YEAR, predict_baseline, save_baseline_model, train_baseline_model  # noqa: E402
from src.config import MODELS_DIR, PROCESSED_DIR, TEHSIL_NAMES  # noqa: E402
from src.features import add_encodings  # noqa: E402
from src.residual_model import leave_one_tehsil_out, predict_residual, save_residual_model, train_residual_model  # noqa: E402
from src.risk import classify_series, fit_thresholds, save_thresholds  # noqa: E402
from src.simulate import balanced_year_means  # noqa: E402

RAW_DATA_PATH = PROCESSED_DIR / "nagpur_blocks.csv"
ENRICHED_PATH = PROCESSED_DIR / "nagpur_blocks_enriched.csv"
STATUS_PATH = PROCESSED_DIR / "tehsil_status.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"


def load_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found — run scripts/build_real_dataset.py first "
            "(or scripts/generate_synthetic_data.py for the legacy synthetic panel)."
        )
    return pd.read_csv(RAW_DATA_PATH)


def _well_covered_years(sub: pd.DataFrame) -> list:
    counts = sub.groupby("year").size()
    return sorted(counts[counts >= 3].index)


def build_tehsil_status(enriched: pd.DataFrame) -> pd.DataFrame:
    """One row per tehsil: latest-year status (from the final models, so it matches the
    sandbox's 'no change' scenario) + how the latest year differs from the first three
    well-covered years, split into natural (baseline) and human (modelled residual) parts."""
    rows = []
    for tehsil in TEHSIL_NAMES:
        sub = enriched[enriched["tehsil"] == tehsil]
        years = _well_covered_years(sub) or sorted(sub["year"].unique())
        last_year = years[-1]
        early_years = years[:3]
        latest = sub[sub["year"] == last_year]
        early = sub[sub["year"].isin(early_years)]

        # season-balanced yearly means, so a missing season in one year cannot masquerade as a change
        b_year = balanced_year_means(sub, "baseline_pred_final")
        r_year = balanced_year_means(sub, "residual_pred_final")
        natural_change = b_year.loc[last_year] - b_year.loc[early_years].mean()
        human_change = r_year.loc[last_year] - r_year.loc[early_years].mean()
        total_change = natural_change + human_change

        denom = abs(natural_change) + abs(human_change)
        natural_pct = 100.0 * abs(natural_change) / denom if denom > 1e-9 else 50.0
        human_pct = 100.0 - natural_pct

        rows.append({
            "tehsil": tehsil,
            "first_year": int(early_years[0]),
            "last_year": int(last_year),
            "latest_observed_water_table_m": round(balanced_year_means(sub, "observed_water_table_m").loc[last_year], 3),
            "latest_baseline_pred_m": round(latest["baseline_pred_final"].mean(), 3),
            "latest_human_residual_m": round(latest["residual_pred_final"].mean(), 3),
            "natural_change_m": round(natural_change, 3),
            "human_change_m": round(human_change, 3),
            "total_change_m": round(total_change, 3),
            "natural_pct": round(natural_pct, 1),
            "human_pct": round(human_pct, 1),
        })
    return pd.DataFrame(rows)


def benchmark_metrics(df: pd.DataFrame) -> dict:
    """Context for the forward-in-time score: how much better than trivial references is Stage 1?"""
    tr = df[df["year"] < TEST_START_YEAR]
    te = df[df["year"] >= TEST_START_YEAR]
    clim = tr.groupby(["tehsil", "season"])["observed_water_table_m"].mean().rename("clim")
    te = te.join(clim, on=["tehsil", "season"]).dropna(subset=["clim"])
    out = {
        "tehsil_season_climatology": {
            "mae": float(mean_absolute_error(te["observed_water_table_m"], te["clim"])),
            "r2": float(r2_score(te["observed_water_table_m"], te["clim"])),
            "n": int(len(te)),
        },
        "persistence_previous_season": {
            "mae": float(mean_absolute_error(df[df["year"] >= TEST_START_YEAR]["observed_water_table_m"],
                                             df[df["year"] >= TEST_START_YEAR]["observed_water_table_m_lag1"])),
            "r2": float(r2_score(df[df["year"] >= TEST_START_YEAR]["observed_water_table_m"],
                                 df[df["year"] >= TEST_START_YEAR]["observed_water_table_m_lag1"])),
        },
    }
    return out


def main():
    print("Loading data...")
    df_all = load_data()
    if "gap_filled" not in df_all:
        df_all["gap_filled"] = False
    # Models are fitted and scored ONLY on measured rows; gap-filled rows are never training or test targets.
    df = df_all[~df_all["gap_filled"]].reset_index(drop=True)
    print(f"  {len(df_all)} rows ({len(df)} measured + {int(df_all['gap_filled'].sum())} gap-filled), "
          f"{df['tehsil'].nunique()} tehsils, years {df['year'].min()}-{df['year'].max()}")

    print("Training Stage 1 — Climate Baseline Model (natural features only)...")
    baseline_fit = train_baseline_model(df)
    print(f"  Baseline: MAE={baseline_fit.mae:.3f} m, RMSE={baseline_fit.rmse:.3f} m, R2={baseline_fit.r2:.3f} "
          f"(train <{TEST_START_YEAR}: {baseline_fit.n_train} rows; forward holdout >={TEST_START_YEAR}: {baseline_fit.n_test} rows)")
    save_baseline_model(baseline_fit.model, baseline_fit.feature_columns)

    encoded_all = add_encodings(df_all)
    baseline_final = predict_baseline(baseline_fit.model, baseline_fit.feature_columns, encoded_all)
    measured = (~df_all["gap_filled"]).to_numpy()
    oof_all = baseline_final.copy()                       # gap-filled rows were never trained on, so this is out-of-sample
    oof_all[measured] = baseline_fit.oof_pred             # measured rows: cross-fitted predictions

    enriched = df_all.copy()
    enriched["baseline_pred"] = oof_all                                     # out-of-fold (honest) baseline
    enriched["human_residual"] = enriched["observed_water_table_m"] - enriched["baseline_pred"]
    enriched["baseline_pred_final"] = baseline_final                        # final model, used by status/sandbox

    print("Training Stage 2 — Residual Extraction Model (anthropogenic features only)...")
    residual_fit = train_residual_model(df, baseline_fit.oof_pred)
    print(f"  Residual:  MAE={residual_fit.mae:.3f} m, RMSE={residual_fit.rmse:.3f} m, R2={residual_fit.r2:.3f} "
          f"(forward holdout)")
    save_residual_model(residual_fit.model, residual_fit.feature_columns)
    enriched["residual_pred_final"] = predict_residual(residual_fit.model, residual_fit.feature_columns, encoded_all)

    print("Building tehsil status summary + fitting risk thresholds...")
    status = build_tehsil_status(enriched)
    thresholds = fit_thresholds(status["latest_human_residual_m"])
    save_thresholds(thresholds)
    status["risk_category"] = classify_series(status["latest_human_residual_m"], thresholds)

    enriched.to_csv(ENRICHED_PATH, index=False)
    status.to_csv(STATUS_PATH, index=False)

    loto_pred, loto_y = leave_one_tehsil_out(df, baseline_fit.oof_pred)
    tm = pd.DataFrame({"t": df["tehsil"], "obs": loto_y.values, "pred": loto_pred}).groupby("t").mean()
    loto = {
        "r2_rows": float(r2_score(loto_y, loto_pred)),
        "tehsil_mean_pearson": float(np.corrcoef(tm["obs"], tm["pred"])[0, 1]),
        "tehsil_mean_spearman": float(tm["obs"].rank().corr(tm["pred"].rank())),
    }
    print(f"  Residual leave-one-tehsil-out: R2={loto['r2_rows']:.3f}, tehsil-mean Spearman={loto['tehsil_mean_spearman']:.2f}")
    meas = enriched[~enriched["gap_filled"]]
    oof_r2 = float(r2_score(meas["observed_water_table_m"], meas["baseline_pred"]))
    corr = float(np.corrcoef(meas["human_residual"], meas["residual_pred_final"])[0, 1])
    metrics = {
        "dataset": {
            "rows": int(len(df)), "rows_total_with_gap_filled": int(len(df_all)),
            "gap_filled_rows": int(df_all["gap_filled"].sum()), "tehsils": int(df["tehsil"].nunique()),
            "year_min": int(df["year"].min()), "year_max": int(df["year"].max()),
            "target_mean_m": float(df["observed_water_table_m"].mean()),
            "target_std_m": float(df["observed_water_table_m"].std()),
            "target_min_m": float(df["observed_water_table_m"].min()),
            "target_max_m": float(df["observed_water_table_m"].max()),
            "median_wells_per_cell": float(df["n_wells"].median()) if "n_wells" in df else None,
        },
        "split": {"scheme": "forward-in-time", "train_before_year": TEST_START_YEAR,
                  "n_train": baseline_fit.n_train, "n_test": baseline_fit.n_test},
        "baseline": {"mae": baseline_fit.mae, "rmse": baseline_fit.rmse, "r2": baseline_fit.r2,
                     "oof_r2_all_rows": oof_r2},
        "residual": {"mae": residual_fit.mae, "rmse": residual_fit.rmse, "r2": residual_fit.r2,
                     "residual_std_m": float(meas["human_residual"].std()),
                     "corr_observed_vs_modelled_residual_all_rows": corr,
                     "leave_one_tehsil_out": loto},
        "benchmarks_holdout": benchmark_metrics(df),
        "importance": {
            "baseline": dict(sorted(zip(baseline_fit.feature_columns, map(float, baseline_fit.model.feature_importances_)),
                                    key=lambda kv: -kv[1])[:10]),
            "residual": dict(sorted(zip(residual_fit.feature_columns, map(float, residual_fit.model.feature_importances_)),
                                    key=lambda kv: -kv[1])),
        },
        "thresholds": thresholds,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote {ENRICHED_PATH}\nWrote {STATUS_PATH}\nWrote {METRICS_PATH}")
    print("\nRisk category counts:")
    print(status["risk_category"].value_counts().reindex(
        ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]
    ).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
