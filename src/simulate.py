"""
Policy Sandbox — the interactive what-if layer described in claude.md:
"officials adjust sliders ... and see the projected drawdown curve and
risk map update in response."

Both trained stages are already persisted (models/*.joblib) — nothing
here retrains anything. A slider move only triggers fast inference against
those fixed artifacts, which is what keeps this feeling instant. Sliders
map to interventions officials actually think in (rainfall change,
drip-irrigation adoption, borewell growth), not raw model columns, since a
% change in "recharge_proxy" means nothing to a DM.

A scenario is evaluated on ALL seasons of a tehsil's latest well-covered
year (not one arbitrary season) and the predictions are averaged, so the
"no change" scenario reproduces the dashboard's "today" status exactly.
"""

from dataclasses import dataclass

import pandas as pd

from src.baseline_model import load_baseline_model, predict_baseline
from src.config import TEHSIL_NAMES
from src.features import add_encodings
from src.residual_model import load_residual_model, predict_residual
from src.risk import classify, load_thresholds

# Assumed elasticities (NOT calibrated parameters — flagged as assumptions in the report/UI):
DRIP_MAX_SAVING = 0.6        # irrigation draft falls by 60% at 100% drip adoption
BOREWELL_ELASTICITY = 0.5    # +1% borewell growth/yr -> +0.5%/yr irrigation draft


@dataclass
class SandboxInputs:
    rainfall_pct_change: float = 0.0            # persistent shift, e.g. -20 = 20% less rainfall every year
    drip_irrigation_adoption_pct: float = 0.0   # 0-100, held constant once adopted
    borewell_growth_pct: float = 0.0            # annual compounding growth rate


@dataclass
class ModelBundle:
    baseline_model: object
    baseline_cols: list
    residual_model: object
    residual_cols: list
    thresholds: dict


def load_models() -> ModelBundle:
    """Load once per process and hand the bundle to every scenario call
    instead of hitting disk on every slider tick."""
    baseline_model, baseline_cols = load_baseline_model()
    residual_model, residual_cols = load_residual_model()
    thresholds = load_thresholds()
    # tiny batches: thread start-up costs more than the trees do, so run inference single-threaded
    baseline_model.set_params(n_jobs=1)
    residual_model.set_params(n_jobs=1)
    return ModelBundle(baseline_model, baseline_cols, residual_model, residual_cols, thresholds)


def _latest_rows(enriched: pd.DataFrame, tehsil: str) -> pd.DataFrame:
    """Rows of the tehsil's most recent year that has at least 3 seasons of data."""
    sub = enriched[enriched["tehsil"] == tehsil]
    counts = sub.groupby("year").size()
    ok = counts[counts >= 3].index
    year = ok.max() if len(ok) else sub["year"].max()
    return sub[sub["year"] == year].copy()


def _predict_rows(bundle: ModelBundle, rows: pd.DataFrame) -> tuple:
    """Stage 1 and Stage 2 stay genuinely separate calls."""
    encoded = add_encodings(rows)
    baseline_pred = predict_baseline(bundle.baseline_model, bundle.baseline_cols, encoded)
    residual_pred = predict_residual(bundle.residual_model, bundle.residual_cols, encoded)
    return baseline_pred, residual_pred


def _apply_scenario_rows(rows: pd.DataFrame, inputs: SandboxInputs, years_forward: int) -> pd.DataFrame:
    df = rows.copy()

    rain_factor = max(0.0, 1 + inputs.rainfall_pct_change / 100.0)
    for col in ("rainfall_mm", "rainfall_mm_lag1", "recharge_proxy"):
        df[col] = rows[col] * rain_factor

    # Drip adoption is a state (fraction of irrigated area converted), not a rate;
    # borewell growth is an annual rate, so it compounds with the horizon.
    irr_factor = (1 - DRIP_MAX_SAVING * inputs.drip_irrigation_adoption_pct / 100.0) * (
        (1 + BOREWELL_ELASTICITY * inputs.borewell_growth_pct / 100.0) ** years_forward
    )
    irr_factor = max(0.0, irr_factor)
    old_total = rows["irrigation_draft_mm"] + rows["domestic_industrial_draft_mm"]
    df["irrigation_draft_mm"] = rows["irrigation_draft_mm"] * irr_factor
    new_total = df["irrigation_draft_mm"] + rows["domestic_industrial_draft_mm"]
    # stage of extraction is draft / extractable resource, so it moves with total draft
    df["stage_of_extraction_pct"] = (rows["stage_of_extraction_pct"] * new_total / old_total.where(old_total > 0, 1)).clip(0, 150)
    return df


def balanced_year_means(sub: pd.DataFrame, col: str) -> pd.Series:
    """Yearly means that are not distorted by which seasons happen to have data: each value is
    expressed relative to its own tehsil-season average before averaging within a year.
    (Pre-monsoon, the deepest season, is missing in 2020-21, which would otherwise fake a recovery.)"""
    mu_ts = sub.groupby("season")[col].transform("mean")
    mu_t = sub.groupby("season")[col].mean().mean()
    return (sub[col] - mu_ts + mu_t).groupby(sub["year"]).mean()


def _season_offset(enriched: pd.DataFrame, tehsil: str, rows: pd.DataFrame, col: str) -> float:
    sub = enriched[enriched["tehsil"] == tehsil]
    mu_ts = sub.groupby("season")[col].mean()
    return float(mu_ts.mean() - rows["season"].map(mu_ts).mean())


def scenario_means(enriched: pd.DataFrame, bundle: ModelBundle, tehsil: str, inputs: SandboxInputs,
                   years_forward: int = 1) -> tuple:
    """(baseline_pred_m, human_residual_pred_m) averaged over the latest year's seasons."""
    rows = _latest_rows(enriched, tehsil)
    scen = _apply_scenario_rows(rows, inputs, years_forward)
    b, r = _predict_rows(bundle, scen)
    return float(b.mean()), float(r.mean())


def projection_means(enriched: pd.DataFrame, bundle: ModelBundle, tehsil: str, inputs: SandboxInputs,
                     horizon_years: int = 10) -> list:
    """[(year, baseline_pred_m, human_residual_pred_m)] for each forward year, one batched inference."""
    rows = _latest_rows(enriched, tehsil)
    last_year = int(rows["year"].iloc[0])
    frames = []
    for offset in range(1, horizon_years + 1):
        f = _apply_scenario_rows(rows, inputs, offset)
        f["_offset"] = offset
        frames.append(f)
    allrows = pd.concat(frames, ignore_index=True)
    b, r = _predict_rows(bundle, allrows)
    allrows["_b"], allrows["_r"] = b, r
    means = allrows.groupby("_offset")[["_b", "_r"]].mean()
    # same season balancing as the historical curve, so the two segments join up
    off_b = _season_offset(enriched, tehsil, rows, "baseline_pred_final")
    off_r = _season_offset(enriched, tehsil, rows, "residual_pred_final")
    return [(last_year + int(o), float(m["_b"]) + off_b, float(m["_r"]) + off_r) for o, m in means.iterrows()]


@dataclass
class SandboxResult:
    tehsil: str
    baseline_pred_m: float
    human_residual_pred_m: float
    projected_water_table_m: float
    risk_category: str


def run_scenario(enriched: pd.DataFrame, bundle: ModelBundle, tehsil: str, inputs: SandboxInputs) -> SandboxResult:
    baseline_pred, residual_pred = scenario_means(enriched, bundle, tehsil, inputs, years_forward=1)
    return SandboxResult(
        tehsil=tehsil,
        baseline_pred_m=baseline_pred,
        human_residual_pred_m=residual_pred,
        projected_water_table_m=baseline_pred + residual_pred,
        risk_category=classify(residual_pred, bundle.thresholds),
    )


def run_all_tehsils(enriched: pd.DataFrame, bundle: ModelBundle, inputs: SandboxInputs) -> pd.DataFrame:
    """Same scenario deltas applied to every tehsil at once — this is what
    redraws the choropleth when a slider moves."""
    rows = []
    for tehsil in TEHSIL_NAMES:
        result = run_scenario(enriched, bundle, tehsil, inputs)
        rows.append({
            "tehsil": tehsil,
            "baseline_pred_m": result.baseline_pred_m,
            "human_residual_pred_m": result.human_residual_pred_m,
            "projected_water_table_m": result.projected_water_table_m,
            "risk_category": result.risk_category,
        })
    return pd.DataFrame(rows)


def project_drawdown_curve(
    enriched: pd.DataFrame, bundle: ModelBundle, tehsil: str, inputs: SandboxInputs, horizon_years: int = 10
) -> pd.DataFrame:
    """Historical observed curve + a forward projection under the current
    slider scenario, for the per-block drawdown chart."""
    sub = enriched[enriched["tehsil"] == tehsil]
    hist = balanced_year_means(sub, "observed_water_table_m").rename("water_table_m").reset_index()
    hist["segment"] = "Historical"

    proj = projection_means(enriched, bundle, tehsil, inputs, horizon_years)
    last_year = proj[0][0] - 1
    future = pd.DataFrame(
        [{"year": y, "water_table_m": b + r, "segment": "Projected"} for y, b, r in proj]
    )
    bridge = hist[hist["year"] == last_year].copy()
    bridge["segment"] = "Projected"
    return pd.concat([hist, bridge, future], ignore_index=True)
