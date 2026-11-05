# GroundVision ML

Machine Learning-Based Groundwater Level Forecasting and Risk Analysis

Groundwater governance dashboard for Nagpur district, separating natural climate-driven depletion from human extraction pressure, and enabling district officials to simulate policy interventions before spending public funds.

## Architecture & Features

- **Two-stage model, kept as genuinely separate inspectable stages**:
  - **Stage 1 — Climate Baseline** (`src/baseline_model.py`): RandomForest trained on rainfall, temperature, rain-recharge depth, local hydrogeology (tehsil), and season. Predicts expected water-table depth under natural climate alone.
  - **Stage 2 — Residual Extraction** (`src/residual_model.py`): RandomForest trained on `observed − baseline_prediction`, regressed against groundwater-extraction figures (irrigation draft, domestic+industrial draft, stage of extraction); trained on cross-fitted (out-of-fold) residuals.
  - Both stages enforce monotonic constraints (`monotonic_cst`).
- **Risk Classification** (`src/risk.py`): Buckets tehsils into Safe, Semi-Critical, Critical, and Over-Exploited using quantile thresholds fit on reference distributions.
- **Policy Sandbox & Simulation Engine** (`src/simulate.py`): Fast single-row inference against persisted artifacts (`models/*.joblib`) producing forward 10-year drawdown projections under perturbed climate and agricultural interventions.
- **FastAPI REST Service** (`src/api.py`):
  - `GET /api/tehsils`: Serves baseline scores, risk categories, attribution splits, and drawdown curves for all 14 tehsils without retraining.
  - `POST /api/simulate`: Runs scenario inference with `{rainfall, drip, borewell, tehsil_id}` and returns updated projections and district-wide scores.
  - `GET /api/report?tehsil=<id>&rainfall=&drip=&borewell=`: Builds a real PDF report (omit `tehsil` for a district-wide report) from the model's own outputs: today's status, the chosen scenario, the no-change reference, charts, and an accuracy-and-limits section. The dashboard's Reports view and "Export report" button download it.
  - `GET /`: Serves the live web dashboard.
- **Frontend Dashboard** (`frontend/` — separated `index.html` + `styles.css` + `app.js`, no logo, no 3D):
  - Interactive choropleth map of Nagpur's 14 tehsils.
  - Policy Sandbox with interactive sliders (rainfall, drip irrigation, borewells) with debounced live API recomputations.
  - Dual-line Chart.js drawdown charts (observed/projected vs. climate baseline).
  - Attribution donut charts and plain-language summary statements.
  - Static hero banner (no WebGL/3D). No logo graphic; sidebar shows plain "GroundVision / Vidarbha pilot" text only.
  - Data sources status, reports generation, and alerts.
  - Interaction contract documented in `docs/INTERACTION_RULES.md`.

## Data (real) and what is NOT real

The model now trains on **real Nagpur data** assembled from public sources:

| Input | Source | Script |
|---|---|---|
| Observed depth to water (target) | GSDA / CGWB observation-well tables (Jan, May, Aug, Nov rounds, 2010-2024; ~10,800 Nagpur readings) | `scripts/extract_gsda_waterlevels.py` |
| Irrigation draft, domestic+industrial draft, stage of extraction, rainfall recharge | GSDA/CGWB Dynamic Ground Water Resource Assessments 2011-12, 2023, 2024 (taluka tables) | `scripts/extract_gsda_assessments.py` |
| Rainfall, temperature | NASA POWER (satellite/reanalysis, not rain gauges) | `scripts/fetch_nasa_power.py` |
| Tehsil boundaries | Survey of India sub-district polygons (india-geodata, CC0) | `data/geo/` |

`scripts/build_real_dataset.py` joins them into `data/processed/nagpur_blocks.csv` (826 tehsil-season rows: 748 with a measured water level plus 78 gap-filled).

**Gap-filling.** A tehsil-season with no usable well reading is filled automatically from real neighbouring measurements: the tehsil's own average for that season plus the district-wide anomaly of that round (or, when too few tehsils were measured, the same season's anomaly interpolated across years). The pipeline therefore never has holes and never returns an empty value. Filled rows are flagged in the `gap_filled` column of the data file and are kept out of model training and scoring, so reported accuracy only reflects measured data. Depth is clipped at 0 m.

Known limits (also stated in the dashboard and the thesis): the human-pressure features exist only as three
assessment snapshots (2011-12, 2023, 2024) interpolated between years; there are no real borewell counts, crop
areas or power data yet; Nagpur Urban/Rural are one taluka in the 2011-12 and 2023 reports; the well network
changes over time (handled with a well-fixed-effect adjustment); Kuhi and Bhiwapur have fewer usable wells; filled values are derived from measured neighbours and cannot capture a purely local event.
Evaluation is forward-in-time (train before 2021, test 2021-2024). See `models/metrics.json` after training.

The earlier synthetic panel is kept as `data/processed/nagpur_blocks_synthetic.csv`
(generated by `scripts/generate_synthetic_data.py`, legacy column names).

## Running the Project

1. Install dependencies:
```bash
pip install -r requirements.txt pymupdf shapely requests
```

2. Rebuild the dataset and retrain (the source PDFs are read from `D:/NAKSHA/research-paper`; pass `--src` to change):
```bash
python scripts/extract_gsda_waterlevels.py
python scripts/extract_gsda_assessments.py
python scripts/fetch_nasa_power.py
python scripts/build_real_dataset.py
python -m src.train
```

3. Launch the FastAPI server:
```bash
uvicorn src.api:app --reload --port 8000
```
*(Or simply run: `python -m src.api`)*

4. Open in your browser:
```
http://localhost:8000
```

## Project Layout

```
src/api.py                           FastAPI application & simulation endpoints (serves frontend/ at /, plus /styles.css, /app.js)
frontend/index.html                  Dashboard markup only (references /styles.css, /app.js, Chart.js CDN)
frontend/styles.css                  All dashboard styling (design tokens, layout, components, responsive)
frontend/app.js                      All dashboard behavior (state, API calls, maps, charts, sandbox, reports)
docs/INTERACTION_RULES.md            Button / interaction / process rules for the frontend
src/config.py                        Tehsil registry, schema, risk categories & color tokens
src/features.py                      Encoding & monotonic constraint definitions
src/baseline_model.py                Stage 1 — Climate Baseline RandomForest
src/residual_model.py                Stage 2 — Residual Extraction RandomForest
src/risk.py                          Quantile-based risk classification
src/train.py                         Pipeline training & artifact generation
src/simulate.py                      Policy Sandbox scenario inference & projection engine
scripts/extract_*.py, fetch_nasa_power.py, build_real_dataset.py   Real-data pipeline
scripts/generate_synthetic_data.py   Legacy synthetic data generator
models/metrics.json                  Held-out metrics written by training
paperwork/                           Project report (thesis) sources and build scripts
models/                              Persisted joblib model artifacts & risk thresholds
data/processed/                      Processed block datasets & summary CSVs
data/geo/                            Tehsil GeoJSON boundaries
```
