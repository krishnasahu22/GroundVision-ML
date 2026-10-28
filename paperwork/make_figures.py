"""
Regenerates every figure and the numbers used in the thesis from the CURRENT data and models.
Writes paperwork/figures/*.png and paperwork/results.json (read by thesis_content.py).

Run from the repo root:  python paperwork/make_figures.py
(Dashboard screenshots come from paperwork/capture_screenshots.py with the API running.)
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

from src.baseline_model import TEST_START_YEAR, _new_model  # noqa: E402
from src.baseline_model import load_baseline_model  # noqa: E402
from src.config import RISK_CATEGORIES, RISK_COLORS, TARGET  # noqa: E402
from src.features import add_encodings, baseline_feature_columns  # noqa: E402
from src.residual_model import load_residual_model  # noqa: E402
from src.simulate import SandboxInputs, load_models, project_drawdown_curve, run_all_tehsils, run_scenario  # noqa: E402

OUT = ROOT / "paperwork" / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

df_all = pd.read_csv(ROOT / "data/processed/nagpur_blocks.csv")
df = df_all[~df_all["gap_filled"]].reset_index(drop=True)      # measured rows: used for every accuracy figure
en = pd.read_csv(ROOT / "data/processed/nagpur_blocks_enriched.csv")   # all rows (incl. gap-filled): used for history/projections
enm = en[~en["gap_filled"]].reset_index(drop=True)
st = pd.read_csv(ROOT / "data/processed/tehsil_status.csv")
metrics = json.loads((ROOT / "models/metrics.json").read_text())
th = metrics["thresholds"]
climate = pd.read_csv(ROOT / "data/raw/nasa_power_climate.csv")
wl = pd.read_csv(ROOT / "data/raw/extracted/gsda_wl_nagpur.csv")
snap = pd.read_csv(ROOT / "data/raw/extracted/assessment_snapshots_by_tehsil.csv")
res = {"metrics": metrics}

# ---- 1 architecture ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 52)


def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)


def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))


box(2, 34, 20, 13, "Natural features\nrainfall, temperature,\nrain-recharge, season,\ntehsil identity", "#dbe9f6")
box(2, 6, 20, 13, "Human-pressure features\nirrigation draft,\ndomestic + industrial draft,\nstage of extraction", "#fbe3d3")
box(32, 34, 20, 13, "Stage 1\nClimate Baseline\nRandomForest\n(monotonic)", "#bcd7ee")
box(32, 6, 20, 13, "Stage 2\nResidual Extraction\nRandomForest\n(monotonic)", "#f6c7a8")
box(62, 20, 16, 13, "Risk\nclassification\n(quartile\nthresholds)", "#e6e6e6")
box(84, 20, 14, 13, "Policy Sandbox\n+ FastAPI\n+ Dashboard", "#d5ecd4")
arr(22, 40.5, 32, 40.5); arr(22, 12.5, 32, 12.5); arr(42, 34, 42, 19.5)
ax.text(43, 26.5, "residual =\nobserved − baseline", fontsize=8, va="center")
arr(52, 12.5, 62, 26); arr(52, 40.5, 62, 28); arr(78, 26.5, 84, 26.5)
ax.text(42, 49.5, "Target: observed depth to water (GSDA/CGWB wells)", fontsize=8, ha="center", style="italic")
fig.savefig(OUT / "fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 2 data overview: mean depth + rainfall by year -------------------------------------------
from src.simulate import balanced_year_means  # noqa: E402
yr = pd.concat([balanced_year_means(g, TARGET) for _, g in df.groupby("tehsil")], axis=1).mean(axis=1)
rain = climate.groupby(["tehsil", "year"])["rainfall_mm"].sum().groupby("year").mean()
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(yr.index, yr.values, color="#1f6f8b", marker="o", lw=1.8, label="Mean depth to water, season-balanced (m bgl, left)")
ax.invert_yaxis(); ax.set_ylabel("Depth to water (m below ground)"); ax.set_xlabel("Year")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.bar(rain.index, rain.values, alpha=0.25, color="#5c8fd6", label="Annual rainfall, NASA POWER (mm, right)")
ax2.set_ylabel("Annual rainfall (mm)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="lower left")
fig.savefig(OUT / "fig_data_overview.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 3 well coverage per tehsil / year ---------------------------------------------------------
cov = df.pivot_table(index="tehsil", columns="year", values="n_wells", aggfunc="median")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
im = ax.imshow(cov.values, aspect="auto", cmap="Blues")
ax.set_yticks(range(len(cov.index))); ax.set_yticklabels(cov.index, fontsize=8)
ax.set_xticks(range(len(cov.columns))); ax.set_xticklabels(cov.columns, rotation=60, fontsize=8)
fig.colorbar(im, ax=ax, label="Median wells per tehsil-season"); ax.spines[:].set_visible(False)
fig.savefig(OUT / "fig_well_coverage.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 4 holdout: predicted vs observed ----------------------------------------------------------
encd = add_encodings(df); cols = baseline_feature_columns()
tr = encd["year"] < TEST_START_YEAR
hold = _new_model().fit(encd.loc[tr, cols], encd.loc[tr, TARGET]); pred = hold.predict(encd.loc[~tr, cols])
obs = encd.loc[~tr, TARGET]
fig, ax = plt.subplots(figsize=(4.8, 4.4)); ax.scatter(obs, pred, s=14, alpha=.7, color="#1f6f8b")
lo, hi = float(min(obs.min(), pred.min())), float(max(obs.max(), pred.max())); ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel("Observed depth to water (m)"); ax.set_ylabel("Stage 1 predicted depth (m)")
ax.set_title(f"Forward holdout {TEST_START_YEAR}–{int(df.year.max())} (n = {int((~tr).sum())})")
fig.savefig(OUT / "fig_pred_vs_obs.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 5 importance ------------------------------------------------------------------------------
bm, bc = load_baseline_model(); rm, rc = load_residual_model()
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, m, c, t, col in ((axs[0], bm, bc, "Stage 1 (climate baseline)", "#1f6f8b"), (axs[1], rm, rc, "Stage 2 (human residual)", "#c8602b")):
    s = pd.Series(m.feature_importances_, index=c).sort_values().tail(8)
    ax.barh([i.replace("_", " ") for i in s.index], s.values, color=col); ax.set_title(t); ax.set_xlabel("Impurity-based importance")
fig.tight_layout(); fig.savefig(OUT / "fig_importance.png", dpi=200); plt.close()

# ---- 6 observed vs baseline for one tehsil -----------------------------------------------------
T = "Katol"
sub = enm[enm.tehsil == T].copy()
sub["t"] = sub["year"] + sub["season"].map({"Winter": 0.04, "Pre-Monsoon": 0.37, "Monsoon": 0.62, "Post-Monsoon": 0.87})
sub = sub.sort_values("t")
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.plot(sub.t, sub[TARGET], "o-", color="#1f6f8b", ms=3, lw=1.2, label="Observed (GSDA wells)")
ax.plot(sub.t, sub.baseline_pred, "s--", color="#e0a030", ms=3, lw=1.2, label="Climate baseline (out-of-fold)")
ax.fill_between(sub.t, sub.baseline_pred, sub[TARGET], where=sub[TARGET] > sub.baseline_pred, color="#d67c5c", alpha=.35, label="Positive residual (deeper than climate predicts)")
ax.invert_yaxis(); ax.set_ylabel("Depth to water (m)"); ax.set_xlabel("Year"); ax.legend(fontsize=8, frameon=False, ncol=1)
ax.set_title(f"{T}: observed vs climate-only baseline")
fig.savefig(OUT / "fig_katol_series.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 7 attribution + risk ----------------------------------------------------------------------
s = st.sort_values("human_pct")
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(s.tehsil, s.natural_pct, color="#5c8fd6", label="Natural climate"); ax.barh(s.tehsil, s.human_pct, left=s.natural_pct, color="#d67c5c", label="Human extraction")
ax.set_xlabel("Share of modelled change in water table (%)"); ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False)
fig.savefig(OUT / "fig_attribution.png", dpi=200, bbox_inches="tight"); plt.close()
s = st.sort_values("latest_human_residual_m")
fig, ax = plt.subplots(figsize=(7, 4.6)); ax.barh(s.tehsil, s.latest_human_residual_m, color=[RISK_COLORS[c] for c in s.risk_category])
for k in ("q25", "q50", "q75"): ax.axvline(th[k], color="k", ls=":", lw=1)
ax.set_xlabel("Latest-year modelled human-attributable residual (m)")
ax.legend(handles=[Patch(color=RISK_COLORS[c], label=c) for c in RISK_CATEGORIES], loc="lower right", frameon=False)
fig.savefig(OUT / "fig_residual_risk.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 8 Stage-2 leave-one-tehsil-out scatter (tehsil means) --------------------------------------
from src.residual_model import leave_one_tehsil_out  # noqa: E402
lp, ly = leave_one_tehsil_out(df, enm["baseline_pred"].values)
tm = pd.DataFrame({"t": df.tehsil, "obs": ly.values, "pred": lp}).groupby("t").mean()
fig, ax = plt.subplots(figsize=(4.8, 4.4)); ax.scatter(tm.pred, tm.obs, color="#c8602b")
for t, r in tm.iterrows(): ax.annotate(t, (r.pred, r.obs), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="#999", lw=.6); ax.axvline(0, color="#999", lw=.6)
ax.set_xlabel("Residual predicted by Stage 2 without seeing the tehsil (m)"); ax.set_ylabel("Observed mean residual (m)")
ax.set_title(f"Leave-one-tehsil-out, 14 tehsil means (Spearman {metrics['residual']['leave_one_tehsil_out']['tehsil_mean_spearman']:.2f})", fontsize=9)
fig.savefig(OUT / "fig_loto.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- 9 scenarios --------------------------------------------------------------------------------
bundle = load_models(); T = "Nagpur Urban"
sc = {"No intervention": SandboxInputs(), "Rainfall −20%": SandboxInputs(rainfall_pct_change=-20),
      "Borewell growth +10%/yr": SandboxInputs(borewell_growth_pct=10),
      "Drip adoption 50%": SandboxInputs(drip_irrigation_adoption_pct=50),
      "Drip 50% + borewell +10%/yr": SandboxInputs(drip_irrigation_adoption_pct=50, borewell_growth_pct=10)}
fig, ax = plt.subplots(figsize=(7.2, 4.2)); rows = []
for i, (n, inp) in enumerate(sc.items()):
    c = project_drawdown_curve(en, bundle, T, inp, 10); h = c[c.segment == "Historical"]; p_ = c[c.segment == "Projected"]
    if i == 0: ax.plot(h.year, h.water_table_m, color="k", lw=1.6, label="Historical (observed, annual mean)")
    ax.plot(p_.year, p_.water_table_m, lw=1.6, ls="--", label=n)
    rs = run_all_tehsils(en, bundle, inp)
    cnt = rs.risk_category.value_counts().reindex(RISK_CATEGORIES).fillna(0).astype(int).to_dict()
    r = run_scenario(en, bundle, T, inp)
    rows.append({"scenario": n, "counts": cnt, "nu_depth_1y": float(r.projected_water_table_m), "nu_depth_10y": float(p_.water_table_m.iloc[-1])})
ax.invert_yaxis(); ax.set_ylabel("Depth to water (m below ground)"); ax.set_xlabel("Year"); ax.set_title(f"{T}: ten-year projection under policy scenarios")
ax.legend(fontsize=8, frameon=False)
fig.savefig(OUT / "fig_scenarios.png", dpi=200, bbox_inches="tight"); plt.close()
res["scenarios"] = rows

base = run_all_tehsils(en, bundle, SandboxInputs())
sens = []
for n, inp in {"Rainfall −20%": SandboxInputs(rainfall_pct_change=-20), "Rainfall −50%": SandboxInputs(rainfall_pct_change=-50),
               "Rainfall +30%": SandboxInputs(rainfall_pct_change=30), "Borewell growth +20%/yr": SandboxInputs(borewell_growth_pct=20),
               "Drip adoption 50%": SandboxInputs(drip_irrigation_adoption_pct=50), "Drip adoption 100%": SandboxInputs(drip_irrigation_adoption_pct=100)}.items():
    r = run_all_tehsils(en, bundle, inp); d = r.projected_water_table_m - base.projected_water_table_m
    sens.append({"slider": n, "mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()),
                 "stage1": float((r.baseline_pred_m - base.baseline_pred_m).abs().mean()),
                 "stage2": float((r.human_residual_pred_m - base.human_residual_pred_m).abs().mean())})
res["sensitivity"] = sens

# ---- numbers for the text ------------------------------------------------------------------------
res["data"] = {
    "well_rows_total": int(len(wl)), "wells_unique": int(wl.groupby(["village", "lat", "lon"]).ngroups),
    "panel_rows": int(len(df)), "panel_rows_total": int(len(df_all)), "gap_filled_rows": int(df_all["gap_filled"].sum()), "rows_per_tehsil": df.groupby("tehsil").size().to_dict(),
    "rows_per_year": {int(k): int(v) for k, v in df.groupby("year").size().items()},
    "median_wells_per_cell": float(df.n_wells.median()), "annual_rain_mean_mm": float(rain.mean()),
    "annual_rain_range_mm": [float(rain.min()), float(rain.max())],
    "mean_depth_by_year": {int(k): float(v) for k, v in yr.items()},
}
res["status"] = st.to_dict("records")
res["snapshots"] = snap.round(2).to_dict("records")
(ROOT / "paperwork" / "results.json").write_text(json.dumps(res, indent=1, default=float))
print("figures written to", OUT); print(json.dumps({k: res[k] for k in ("scenarios", "sensitivity")}, indent=1)[:2500])
