# -*- coding: utf-8 -*-
"""
Thesis body content for GroundVision ML (REAL-DATA version). Consumed by build_thesis.py.
Numbers are read from paperwork/results.json (written by make_figures.py) so the text cannot drift
from the code.  Re-run:  python paperwork/make_figures.py  ->  python paperwork/build_thesis.py

Item types:
  ("ch", "TITLE")   ("h2", "1.1 TITLE")   ("h3", "...")   ("p", text; **bold**, __italic__, _{sub})
  ("bul", [..])  ("num", [..])  ("fig", file, caption, width_in)  ("tbl", caption, header, rows, widths)
  ("note", text)  ("eq", text)  ("code", text)  ("pb",)
"""

import json
import re
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / "results.json").read_text(encoding="utf8"))
M = R["metrics"]
B, S2, BM, LO = M["baseline"], M["residual"], M["benchmarks_holdout"], M["residual"]["leave_one_tehsil_out"]
DS, D, TH = M["dataset"], R["data"], M["thresholds"]
SPLIT = M["split"]
CLIM, PERS = BM["tehsil_season_climatology"], BM["persistence_previous_season"]
IMP1, IMP2 = M["importance"]["baseline"], M["importance"]["residual"]


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def sgn(x): return f"{x:+.2f}".replace("-", "\u2212")
def neg(x): return f"{x:.2f}".replace("-", "\u2212")


STATUS = sorted(R["status"], key=lambda r: -r["latest_human_residual_m"])
COUNTS = {c: sum(1 for r in STATUS if r["risk_category"] == c) for c in ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]}
SC = {s["scenario"]: s for s in R["scenarios"]}
SEN = {s["slider"]: s for s in R["sensitivity"]}
N_ROWS = DS["rows"]
HOLD = f"{SPLIT['train_before_year']}\u20132024"

ABSTRACT = [
    "Groundwater supplies most of the irrigation water used in India, and in Vidarbha it is the buffer that "
    "keeps agriculture viable between monsoons. Public tools such as India-WRIS and IN-GRES archive well "
    "readings and report a stage-of-extraction category, but they do not tell a district administrator why a "
    "tehsil's water table is falling, nor what would change if a policy were adopted. A falling water table "
    "may reflect a weak monsoon that will recover, or sustained pumping that will not.",

        "This project presents GroundVision ML, a machine-learning-based groundwater forecasting and risk analysis "
        "decision-support system for the 14 tehsils of Nagpur district, Maharashtra. GroundVision ML uses two deliberately separate, "
    "interpretable stages. Stage 1 is a climate-baseline RandomForest that sees only natural drivers (rainfall, "
    "temperature, rain-recharge, season and tehsil identity). The residual between the observed water table and "
    "this baseline is regressed, in Stage 2, against groundwater-extraction figures (irrigation draft, domestic "
    "and industrial draft, stage of extraction) to estimate human-attributable pressure. Both stages use "
    "monotonic constraints so that a policy simulator cannot show a worse input producing a better outcome. "
    "Tehsils are classified as Safe, Semi-Critical, Critical or Over-Exploited, and a Policy Sandbox lets "
    "officials move rainfall, drip-irrigation and borewell-growth sliders and see the risk map and ten-year "
    "drawdown projection update. The system is served by a FastAPI back end with a browser dashboard.",

    f"The models are trained on real data: about {D['well_rows_total']:,} GSDA/CGWB observation-well readings "
    f"(2010\u20132024, four rounds a year), NASA POWER rainfall and temperature, and taluka-level extraction "
    f"figures from the 2011-12, 2023 and 2024 Dynamic Ground Water Resource Assessments. On a forward-in-time "
    f"holdout ({HOLD}) Stage 1 reaches MAE = {f2(B['mae'])} m and R\u00b2 = {f2(B['r2'])}, but a simple "
    f"tehsil-by-season average does as well (R\u00b2 = {f2(CLIM['r2'])}), so weather adds little beyond "
    f"seasonality in this dataset. Stage 2 has no useful out-of-time skill (R\u00b2 = {neg(S2['r2'])}), because "
    f"the extraction figures exist only as three snapshots; it does rank tehsils moderately well "
    f"(Spearman {f2(LO['tehsil_mean_spearman'])} across 14 tehsils). The report states these limits plainly "
    f"and identifies the missing data (borewell counts, crop areas, annual power use) that would strengthen the "
    f"attribution.",
]

KEYWORDS = ("Groundwater depletion, Nagpur, GSDA, Random Forest, monotonic constraints, residual modelling, "
            "climate\u2013human attribution, policy simulation, decision support, FastAPI")

BODY = []
A = BODY.append

# =====================================================================================
A(("ch", "INTRODUCTION"))
A(("h2", "1.1 BACKGROUND"))
A(("p", "India is the world's largest user of groundwater. Millions of dug wells, tube wells and borewells "
        "supply the irrigation that underpins national food production, and also most rural drinking water. "
        "The expansion of pumped irrigation since the Green Revolution has raised agricultural output, but "
        "it has also pushed abstraction beyond natural recharge in many regions, and observation wells and "
        "satellite gravimetry now show persistent depletion across large parts of the country [1], [2], [3]."))
A(("p", "Vidarbha, in eastern Maharashtra, shows the problem in a different form from the heavily "
        "irrigated north-west. Nagpur district receives most of its rainfall in a short monsoon, depends largely "
        "on shallow weathered and fractured hard-rock aquifers that recharge and drain quickly, and supports "
        "water-intensive crops such as cotton, soybean and orchards alongside rain-fed farming. In such "
        "settings the water table can fall for two very different reasons: a poor monsoon, which is "
        "temporary and reverses when rainfall returns, or sustained pumping, which is not. The correct "
        "administrative response to each is different, and confusing them wastes public money."))
A(("p", "India's Central Ground Water Board (CGWB) and the Maharashtra Groundwater Surveys and Development "
        "Agency (GSDA) classify each assessment unit, normally a taluka, by its stage of groundwater extraction, "
        "the ratio of annual extraction to annual extractable resource. The categories are Safe, Semi-Critical, "
        "Critical and Over-Exploited. These labels are familiar to district officials and GroundVision ML adopts the same "
        "vocabulary so that its output is immediately readable."))

A(("h2", "1.2 PROBLEM STATEMENT"))
A(("p", "Public groundwater platforms such as India-WRIS and IN-GRES are archives and reporting tools. They "
        "record what the water table is and which category a block falls in. They do not (a) separate how much "
        "of an observed depletion is attributable to climate and how much to human extraction, or (b) let a "
        "decision-maker test a proposed intervention before funding it. General-purpose forecasting models "
        "(ARIMA, LSTM) output a single future number without explanation, which officials cannot interrogate "
        "or defend."))
A(("p", "The problem addressed here is therefore: **how can the observed groundwater level of each tehsil be "
        "decomposed into a climate-driven component and a human-attributable component, using a model "
        "transparent enough for administrators to trust, and how can that decomposition be used to "
        "simulate the effect of policy changes on tehsil-level risk?**"))

A(("h2", "1.3 OBJECTIVE"))
A(("p", "The objectives of the project are:"))
A(("num", [
    "To build a **climate baseline model** that predicts the water table from natural drivers alone.",
    "To extract the **human-attributable residual** (observed minus baseline) and model it against "
    "groundwater-extraction figures in a separate, inspectable second stage.",
    "To enforce **monotonic constraints** on both stages so that model behaviour is physically sensible "
    "and a what-if simulator cannot show a worse input producing a better outcome.",
    "To classify each of Nagpur's 14 tehsils into Safe / Semi-Critical / Critical / Over-Exploited using "
    "IN-GRES-style language.",
    "To implement a **Policy Sandbox** in which rainfall, drip-irrigation adoption and borewell growth can be "
    "varied and the risk map and ten-year drawdown curve recomputed from persisted models without retraining.",
    "To deliver the system as a **REST service and browser dashboard** usable by a non-technical official, "
    "with plain-language summaries alongside the charts.",
    "To assemble a **real tehsil-level dataset** from public GSDA/CGWB and NASA sources, and to document "
    "honestly what the evaluation does and does not show.",
]))

A(("h2", "1.4 SIGNIFICANCE"))
A(("p", "The distinction between natural and human-driven depletion is the single most decision-relevant piece "
        "of information for groundwater governance. Where depletion is climate-driven, the appropriate response "
        "is resilience planning (storage, recharge structures, contingency supply). Where it is extraction-"
        "driven, the appropriate responses are demand-side: crop-mix guidance, micro-irrigation, borewell "
        "regulation, and feeder-level power management. A tool that makes the split visible, and lets "
        "officials rehearse the consequences of each response, supports better targeting of limited funds."))
A(("p", "Three design choices distinguish the work. First, the attribution is structural rather "
        "than post-hoc: the baseline model is physically prevented from seeing human variables. Second, the "
        "models are interpretable tree ensembles with monotonic constraints rather than neural networks, so "
        "feature importance and direction of effect can be stated plainly. Third, the simulator reuses "
        "persisted models and performs fast inference, so interaction is immediate."))

A(("h2", "1.5 SCOPE AND LIMITATIONS"))
A(("p", "The pilot covers the 14 tehsils of Nagpur district at seasonal resolution (Pre-Monsoon, Monsoon, "
        "Post-Monsoon, Winter) for 2010\u20132024. The geographic scope is deliberately limited to Nagpur; "
        "hydrogeology varies substantially across India and no pan-India accuracy is claimed. Boundaries are "
        "kept at tehsil level; village-level resolution is outside scope."))
A(("note", "Data scope. The target (depth to water), rainfall, temperature and the extraction figures are real "
           "public data (Section 3.2). Three things are not: there are no real borewell counts, crop-area or "
           "agricultural power data in the model; the extraction figures exist only for three assessment years "
           "(2011-12, 2023, 2024) and are interpolated between them; and the 2011-12 and 2023 reports treat Nagpur "
           "Urban and Rural as a single taluka. Chapter 5 shows what these limits mean for the results."))
A(("p", "A further limitation is conceptual. The residual is not pure human signal: it also absorbs model "
        "error and unmeasured natural variables such as aquifer geology and inter-basin flow. GroundVision ML therefore "
        "describes its output as **estimated human-attributable pressure**, never as a certainty."))

A(("h2", "1.6 ORGANISATION OF THE REPORT"))
A(("p", "Chapter 2 reviews the literature on groundwater depletion, attribution, data-driven modelling and "
        "interpretable learning, and identifies the gap GroundVision ML addresses. Chapter 3 describes the data "
        "pipeline, the two models, the attribution and risk logic, the policy simulator and the system "
        "architecture. Chapter 4 lists software and hardware requirements. Chapter 5 presents and critically "
        "discusses the results. Chapter 6 concludes and sets out future work. The appendices give run commands, "
        "API details and key code excerpts."))

# =====================================================================================
A(("ch", "LITERATURE REVIEW"))
A(("h2", "2.1 GROUNDWATER DEPLETION IN INDIA"))
A(("p", "Rodell et al. used satellite gravimetry from the GRACE mission to estimate that north-western India "
        "lost groundwater at a rate far above natural replenishment during 2002\u20132008, a result that drew "
        "attention to the scale of the problem [2]. MacDonald et al. later combined observations to show "
        "the interaction of depletion and quality problems in the Indo-Gangetic basin [4], and Famiglietti "
        "framed groundwater depletion as a global issue with India as a leading hotspot [5]. Dangar, Asoka "
        "and Mishra review the causes and implications for India and conclude that pumping for irrigation is "
        "the principal driver of the decline, while noting that both in-situ and satellite measurements carry "
        "considerable uncertainty and that monitoring networks need strengthening [1]."))
A(("p", "These studies are mostly continental or regional in scale. Fewer address the tehsil or block scale at "
        "which Indian groundwater is actually administered, and almost none address the hard-rock aquifers "
        "of central India where recharge is rapid and storage is limited."))

A(("h2", "2.2 SEPARATING CLIMATE AND HUMAN DRIVERS"))
A(("p", "Asoka et al. quantified the relative contribution of monsoon precipitation and pumping to changes "
        "in groundwater storage across India, finding that rainfall explains a large share of storage "
        "variability in north-central and southern India, whereas storage variability in north-western India "
        "is dominated by irrigation abstraction, which itself responds to precipitation [6]. This is the "
        "conceptual precedent for GroundVision ML's decomposition. It also illustrates the difficulty: climate and "
        "pumping are coupled, because dry years increase pumping. Cuthbert et al. mapped global patterns of "
        "climate\u2013groundwater interaction and showed that groundwater response times vary widely with "
        "hydrogeology [7]. Gleeson et al. introduced the groundwater footprint, comparing abstraction with "
        "recharge to identify stressed aquifers [8], which is the same logic that underlies the stage-of-"
        "extraction ratio used by the CGWB."))
A(("p", "These works attribute at the scale of large grid cells and rely on gravimetry or physical models. "
        "GroundVision ML takes the same question to tehsil scale using tabular records and a statistical decomposition."))

A(("h2", "2.3 DATA-DRIVEN GROUNDWATER-LEVEL MODELS"))
A(("p", "Rajaee, Ebrahimi and Nourani survey artificial-intelligence methods for groundwater-level modelling "
        "(ANN, ANFIS, SVM, wavelet hybrids) and note the strong dependence of performance on input data "
        "quality and the shortage of studies that address interpretation [9]. Deep sequence models have "
        "been applied successfully; for example Zhang et al. used an LSTM to predict water-table depth in an "
        "agricultural area [10], building on the LSTM architecture of Hochreiter and Schmidhuber [11]. Tree "
        "ensembles, notably Random Forests [12] and gradient boosting [13], are widely used for tabular "
        "environmental data because they handle non-linearity and mixed feature types with little tuning and "
        "provide feature-importance measures. Scikit-learn provides production-quality implementations [14]."))

A(("h2", "2.4 INTERPRETABILITY AND CONSTRAINED LEARNING"))
A(("p", "Rudin argues that for high-stakes decisions one should prefer inherently interpretable models over "
        "post-hoc explanations of black boxes [15]. Groundwater governance is such a setting: an official "
        "must be able to justify a decision. Monotonic constraints are one practical route to "
        "interpretability. By requiring that the prediction be non-decreasing (or non-increasing) in a "
        "chosen feature, the modeller injects domain knowledge, for example that more rainfall never "
        "deepens the water table, and prevents a flexible learner from fitting spurious sign reversals. "
        "Scikit-learn's tree ensembles support such constraints through the monotonic_cst parameter [14]. "
        "This matters particularly for a what-if simulator, where an unconstrained tree can produce a "
        "locally inverted response that offline error metrics would not reveal."))

A(("h2", "2.5 EXISTING PLATFORMS AND THE GAP"))
A(("p", "India-WRIS provides a web portal for water-resources data including groundwater levels, and "
        "CGWB and GSDA publish periodic Dynamic Ground Water Resource Assessments at the taluka level "
        "with the four stage-of-extraction categories [3], [19]\u2013[21]. IN-GRES presents these assessments "
        "interactively. These are authoritative and indispensable as data sources, and GroundVision ML is built on them, "
        "but they report historical or current status. Table 2.1 summarises how GroundVision ML differs."))
A(("tbl", "Table 2.1: Comparison of GroundVision ML with existing approaches",
   ["Capability", "India-WRIS / IN-GRES", "Single-output ML forecasters", "GroundVision ML"],
   [["Historical water-level / status reporting", "Yes", "Partly", "Yes (from supplied data)"],
    ["Natural versus human attribution", "No", "No", "Yes (two-stage)"],
    ["Interpretable feature effects", "Not applicable", "Usually no (LSTM)", "Yes (RF + monotonic)"],
    ["What-if policy simulation", "No", "No", "Yes (Policy Sandbox)"],
    ["Plain-language output for officials", "Limited", "No", "Yes"],
    ["Tehsil-level risk map", "Yes (assessment unit)", "Rarely", "Yes"]],
   [2.0, 1.4, 1.45, 1.4]))
A(("p", "The gap is therefore not a lack of data or of forecasting methods, but a lack of a transparent "
        "attribution-plus-simulation layer at the administrative scale. GroundVision ML is an attempt to fill that "
        "gap with a deliberately modest, inspectable model."))

# =====================================================================================
A(("ch", "METHODOLOGY"))
A(("h2", "3.1 STUDY AREA AND UNIT OF ANALYSIS"))
A(("p", "The study area is Nagpur district, Maharashtra, comprising 14 tehsils: Nagpur Urban, Nagpur Rural, "
        "Kamptee, Hingna, Katol, Narkhed, Savner, Kalmeshwar, Ramtek, Mouda, Parseoni, Umred, Kuhi and "
        "Bhiwapur. The unit of analysis is the tehsil-season, with up to four observations per tehsil per year. "
        "The study window is 2010\u20132024. Tehsil boundaries are the Survey of India sub-district "
        "polygons distributed through the open india-geodata repository (CC0) and stored as GeoJSON "
        "(data/geo/nagpur_tehsils.geojson)."))

A(("h2", "3.2 DATA SCHEMA AND SOURCES"))
A(("p", "All downstream code depends on one tidy table with one row per tehsil, year and season. Table 3.1 lists "
        "each variable, its role and its source. Every input is real public data; nothing in the table is "
        "simulated."))
A(("tbl", "Table 3.1: Model variables, roles and sources",
   ["Variable", "Role", "Source and construction"],
   [["observed_water_table_m", "Target (m below ground)", "GSDA/CGWB observation wells; well-fixed-effect adjusted median per tehsil-season-year"],
    ["rainfall_mm (+ lag1)", "Natural feature", "NASA POWER daily precipitation summed to season"],
    ["temperature_c", "Natural feature", "NASA POWER 2 m air temperature averaged to season"],
    ["recharge_proxy", "Natural feature", "Rain-recharge depth = tehsil infiltration fraction × season rainfall (fraction from assessments)"],
    ["irrigation_draft_mm", "Human-pressure feature", "Assessment irrigation draft ÷ recharge-worthy area; interpolated between report years"],
    ["domestic_industrial_draft_mm", "Human-pressure feature", "Assessment domestic + industrial draft ÷ area; interpolated"],
    ["stage_of_extraction_pct", "Human-pressure feature", "Reported stage of ground water extraction (%); interpolated"],
    ["observed_water_table_m_lag1", "Lag (constructed)", "Previous measurement round (chronological)"],
    ["tehsil, season", "Categorical", "Registry in src/config.py; one-hot encoded"]],
   [1.9, 1.45, 2.9]))

A(("h3", "3.2.1 Groundwater levels"))
A(("p", "The target comes from the GSDA state water-level tables [22]: long printed tables listing, for every "
        "observation well in Maharashtra, the state, district, block (taluka), village, coordinates, date and depth "
        "to water in metres below ground level (mbgl). Four PDFs hold the January, May (pre-monsoon), August "
        "and November (post-monsoon) rounds. The PDFs contain machine-readable text, so a script "
        "(scripts/extract_gsda_waterlevels.py) scans every page and keeps the rows for Maharashtra and Nagpur, "
        f"yielding {D['well_rows_total']:,} readings from {D['wells_unique']} distinct wells. Readings that are "
        "zero or negative, or deeper than 60 m, are discarded as likely recording errors (78 rows)."))
A(("p", "GSDA lists Nagpur Urban and Nagpur Rural together as one block, \u201cNagpur\u201d. Because every "
        "well has coordinates, those wells are assigned to the Urban or Rural polygon by point-in-polygon "
        "testing (wells on a boundary go to the nearest polygon). Wells in the other blocks keep their GSDA "
        "block label; 84.6% of them fall inside the matching polygon, and the remainder lie on administrative "
        "boundaries that differ slightly between the two sources."))
A(("p", "Measurement months map to the four seasons used throughout the project: January = Winter, "
        "May = Pre-Monsoon, August = Monsoon, November = Post-Monsoon."))
A(("p", "**Adjusting for a changing well network.** The number of reporting wells grows from about 40 per "
        "round in the 1990s and 2000s to 150\u2013200 after 2012, and wells enter and leave. A naive average of "
        "all readings would therefore mix real change with changes in which wells were measured. Each "
        "tehsil-season-year value is therefore computed with a well fixed effect: for every well with at least "
        "five readings in a season, its long-run mean for that season is subtracted; the median of these "
        "anomalies across the wells measured in that year is added back to the median of the wells' long-run "
        "means. A tehsil-season-year is kept only if at least two such wells were measured. Cells with too "
        "few wells, and the rounds that simply were not published (for example May 2021 and November 2024), "
        "are left missing rather than filled."))

A(("h3", "3.2.2 Rainfall and temperature"))
A(("p", "Daily rainfall and temperature for each tehsil centroid are downloaded from the NASA POWER service "
        "[17], [18] (scripts/fetch_nasa_power.py) and aggregated to the four seasons (Winter = December\u2013"
        "February, Pre-Monsoon = March\u2013May, Monsoon = June\u2013September, Post-Monsoon = October\u2013November). "
        "NASA POWER is a satellite and reanalysis product at roughly half-degree resolution, not a rain-gauge "
        f"record; it gives a mean annual rainfall of {D['annual_rain_mean_mm']:.0f} mm (range "
        f"{D['annual_rain_range_mm'][0]:.0f}\u2013{D['annual_rain_range_mm'][1]:.0f} mm) for Nagpur. The IMD "
        "0.25° gridded rainfall product [16] is a gauge-based alternative that was not available to the project."))

A(("h3", "3.2.3 Recharge and extraction"))
A(("p", "Human-pressure features come from the taluka tables of the Dynamic Ground Water Resource Assessments "
        "for 2011-12 [19], 2023 [20] and 2024 [21]. A second script (scripts/extract_gsda_assessments.py) reads, for "
        "each Nagpur taluka, the recharge from rainfall, the groundwater extraction for irrigation, the "
        "extraction for domestic and industrial use, the recharge-worthy area and the stage of extraction. "
        "Extraction is divided by area and expressed in millimetres of water over the recharge-worthy area. "
        "The reports contain no borewell counts, so a borewell-density feature could not be built."))
A(("p", "Three snapshots are all the reports provide. They are placed at 2011, 2023 and 2024 and linearly "
        "interpolated to each year, with the 2011 values held constant for 2010. The 2011-12 and 2023 reports "
        "list Nagpur as one taluka, so Nagpur Urban and Rural share those values; the 2024 report separates them. "
        "The rain-recharge depth is turned into a time-invariant infiltration fraction by dividing by the "
        "tehsil's mean annual NASA rainfall, and the model's recharge_proxy is that fraction times each "
        "season's rainfall. **Because the human-pressure features are piecewise-linear interpolations "
        "between three points, they cannot capture year-to-year changes in pumping; this is the main data limitation "
        "of the study** (Chapter 5)."))

A(("h3", "3.2.4 The assembled panel"))
A(("p", f"The final table (scripts/build_real_dataset.py) has {N_ROWS} tehsil-season rows from "
        f"{DS['year_min']} to {DS['year_max']}: of 840 possible cells, 79 have no usable water-level value and "
        "a further 63 are dropped because their preceding round is missing (lag features). The median cell "
        f"is based on {D['median_wells_per_cell']:.0f} wells. Coverage is uneven: Bhiwapur ({D['rows_per_tehsil']['Bhiwapur']} rows) "
        f"and Kuhi ({D['rows_per_tehsil']['Kuhi']}) have few wells with long histories, while Nagpur Urban has by "
        "far the densest network (Figure 5.2)."))

A(("h2", "3.3 FEATURE ENGINEERING"))
A(("p", "Season and tehsil are one-hot encoded into fixed-width columns in a single shared function "
        "(src/features.py) used by both training and inference, so a perturbed row passed from the sandbox "
        "lines up column-for-column with the training matrix. Lag features refer to the previous "
        "measurement round in chronological order (Winter, Pre-Monsoon, Monsoon, Post-Monsoon within a "
        "calendar year)."))
A(("p", "The two stages are given strictly disjoint inputs. Stage 1 uses rainfall, temperature, rain-recharge, "
        "lagged rainfall, season and tehsil identity (local hydrogeology is a natural, not human, variable). "
        "Stage 2 uses only the three human-pressure features and season. This disjointness is "
        "enforced in code, not left as a convention, because the separation is the central technical claim."))

A(("h2", "3.4 STAGE 1: CLIMATE BASELINE MODEL"))
A(("p", "Stage 1 is a RandomForestRegressor that maps natural features to the observed water table. "
        "Writing x_{nat} for the natural feature vector, the baseline prediction is"))
A(("eq", "ŷ_{base} = f₁(x_{nat}),     y = observed depth to water (m below ground)"))
A(("p", "Monotonic constraints encode physical expectations (Table 3.2). Higher rainfall, recharge and "
        "lagged rainfall can only make the water table shallower (a lower value of depth); higher "
        "temperature can only make it deeper."))
A(("tbl", "Table 3.2: Monotonic constraints (effect on depth to water)",
   ["Stage", "Feature", "Constraint", "Rationale"],
   [["1", "rainfall_mm", "Non-increasing", "More rain → shallower water table"],
    ["1", "rainfall_mm_lag1", "Non-increasing", "Preceding recharge helps"],
    ["1", "recharge_proxy", "Non-increasing", "More recharge → shallower"],
    ["1", "temperature_c", "Non-decreasing", "More evapotranspiration → deeper"],
    ["2", "irrigation_draft_mm", "Non-decreasing", "More irrigation pumping → deeper"],
    ["2", "domestic_industrial_draft_mm", "Non-decreasing", "More other pumping → deeper"],
    ["2", "stage_of_extraction_pct", "Non-decreasing", "Higher extraction ratio → deeper"]],
   [0.55, 2.25, 1.3, 2.15]))
A(("p", "Hyperparameters were set conservatively to limit overfitting on a small panel: 300 trees, maximum "
        "depth 8, minimum 3 samples per leaf, random seed 42. **Evaluation is forward in time:** the model is "
        f"fitted on rounds before {SPLIT['train_before_year']} ({SPLIT['n_train']} rows) and scored on "
        f"{SPLIT['train_before_year']}\u20132024 ({SPLIT['n_test']} rows). A random split would let neighbouring "
        "seasons of the same tehsil appear in both sets and overstate skill. The model that is persisted and "
        "served is then refitted on all rows. In addition, cross-fitted (out-of-fold) predictions are produced "
        "with five folds grouped by year, so that every row is predicted by a model that did not see its year."))

A(("h2", "3.5 STAGE 2: RESIDUAL EXTRACTION MODEL"))
A(("p", "The historical residual for every row is the observed water table minus the cross-fitted Stage 1 "
        "prediction,"))
A(("eq", "r = y − ŷ_{base}^{oof}"))
A(("p", "Using out-of-fold predictions matters: with in-sample predictions the baseline would already have "
        "memorised part of the signal and the residual would be artificially small. The residual is positive "
        "where the water table is deeper than climate alone would predict, interpreted as human-attributable "
        "depletion pressure. Stage 2 is a second RandomForest regressing r on the human-pressure features "
        "x_{hum} and season:"))
A(("eq", "r̂ = f₂(x_{hum})"))
A(("p", "It uses 300 trees, maximum depth 6, minimum 3 samples per leaf and the monotonic constraints of "
        "Table 3.2. The two stages are separate modules (baseline_model.py and residual_model.py) with "
        "separate persisted artefacts, so the separation is demonstrable rather than merely asserted. The "
        "predicted water table under a scenario is the sum ŷ_{base} + r̂. Stage 2 is evaluated on the same "
        "forward split and, because the risk map depends on ranking tehsils against each other, also with a "
        "leave-one-tehsil-out test in which each tehsil's residual is predicted by a model that never saw "
        "that tehsil."))
A(("note", "Framing caution carried into the interface: r also absorbs Stage 1 error and unmeasured natural "
           "variables. It is reported as estimated human-attributable pressure, not as a measured quantity."))

A(("h2", "3.6 NATURAL VERSUS HUMAN ATTRIBUTION"))
A(("p", "For each tehsil, the change between the first three well-covered years and the latest year is split "
        "into a natural and a human part, using the final models' predictions. Let b̄₀ and b̄₁ be the mean "
        "baseline prediction in the early window and the latest year, and r̄₀ and r̄₁ the mean Stage 2 "
        "prediction. Then"))
A(("eq", "Δ_{nat} = b̄₁ − b̄₀,     Δ_{hum} = r̄₁ − r̄₀"))
A(("eq", "natural % = 100 · |Δ_{nat}| / (|Δ_{nat}| + |Δ_{hum}|),     human % = 100 − natural %"))
A(("p", "Yearly means are season-balanced: each value is first expressed relative to its own tehsil-season "
        "average, so that a missing season in one year (pre-monsoon, the deepest season, is absent in 2020 and "
        "2021) cannot masquerade as a recovery. If both changes are negligible the split is reported as 50/50. "
        "The split describes the relative size of two modelled changes over time; it is not a measured "
        "apportionment."))

A(("h2", "3.7 RISK CLASSIFICATION"))
A(("p", "The official categories are defined by stage of extraction, and the assessments report that "
        "directly. GroundVision ML nevertheless ranks tehsils by their latest-year modelled human-attributable residual "
        "and assigns categories by quartile thresholds (25th, 50th, 75th percentile) fitted once on the 14 "
        "tehsils:"))
A(("eq", "Safe if r̂ ≤ q₂₅;  Semi-Critical if ≤ q₅₀;  Critical if ≤ q₇₅;  otherwise Over-Exploited"))
A(("p", "The reason is that the goal is to show how a scenario shifts a tehsil relative to the others, which "
        "the published categories (fixed years, fixed thresholds) cannot do. The thresholds are saved and "
        "reused by the sandbox so that a tehsil's category changes relative to a fixed reference. This is a "
        "relative ranking, not the official stage-of-extraction classification, and Section 5.8 discusses the "
        "consequence."))

A(("h2", "3.8 POLICY SANDBOX AND SIMULATION"))
A(("p", "The sandbox expresses interventions in terms officials understand and maps them onto model inputs "
        "(Table 3.3). A scenario is evaluated on all seasons of each tehsil's latest well-covered year and the "
        "predictions are averaged. This makes the \u201cno change\u201d scenario reproduce the \u201ctoday\u201d "
        "status exactly (verified for all 14 tehsils). No retraining occurs."))
A(("tbl", "Table 3.3: Mapping of slider interventions to model inputs",
   ["Slider", "Range", "Effect on model inputs"],
   [["Rainfall change (%)", "−30 to +30 (UI)", "Scales rainfall_mm, rainfall_mm_lag1 and recharge_proxy by (1 + p/100)"],
    ["Drip-irrigation adoption (%)", "0 to 100", "Scales irrigation_draft_mm by (1 − 0.6·a/100); a state, held constant"],
    ["Borewell growth (%/yr)", "−20 to +50 (UI)", "Scales irrigation_draft_mm by (1 + 0.5·g/100)^t; stage of extraction moves with total draft"]],
   [1.75, 1.2, 3.3]))
A(("p", "A ten-year drawdown curve is generated by evaluating the scenario for horizons t = 1…10, "
        "joining the season-balanced historical annual mean of observed depth with the projected values. "
        "The 0.6 and 0.5 coefficients are assumed elasticities (that full drip adoption saves 60% of irrigation "
        "draft, and that each 1% of annual borewell growth raises irrigation draft by 0.5%), chosen for "
        "plausibility and not calibrated; they are flagged as assumptions in the interface and in Chapter 5. "
        "Because the risk category is computed from the Stage 2 residual only, the rainfall slider changes "
        "the projected depth but never the category."))

A(("h2", "3.9 SYSTEM ARCHITECTURE AND IMPLEMENTATION"))
A(("p", "Figure 3.1 shows the overall pipeline. Natural features feed Stage 1; its residual against the "
        "observed target feeds Stage 2 together with the human-pressure features; both stages feed the risk "
        "classifier; and the results are exposed through the sandbox, a REST service and the dashboard."))
A(("fig", "fig_architecture.png", "Figure 3.1: Two-stage architecture of GroundVision ML", 6.0))
A(("p", "**Back end.** A FastAPI application (src/api.py) loads the persisted models and processed tables "
        "once at start-up and precomputes baseline results for all 14 tehsils. It exposes "
        "GET /api/tehsils (baseline scores, categories, attribution splits and drawdown curves) and "
        "POST /api/simulate (scenario results for a chosen tehsil together with updated scores for every "
        "tehsil so the map can be recoloured in a single call). It also serves the front-end files and the "
        "GeoJSON boundaries. The residual is converted to a 0\u2013100 pressure score by piecewise linear "
        "calibration between the saved quartile thresholds. Inference is single-threaded, which brought a "
        "simulation request to about 0.7 s on a laptop."))
A(("p", "**Front end.** A single-page dashboard (HTML, CSS and JavaScript with Chart.js) provides five views: "
        "Dashboard (risk summary chips, tehsil map, detail panel with drawdown and attribution charts), "
        "Policy Sandbox (sliders, recoloured map, projected drawdown), Reports, Data Sources (which states "
        "plainly which datasets are in use and which are not), and How It Works. Slider changes are debounced "
        "before calling the API. If the API is unreachable, the page falls back to built-in preview constants "
        "with simplified effects so the interface never blocks."))

A(("h2", "3.10 EVALUATION PLAN"))
A(("p", "Four layers of evaluation are used. (i) Predictive accuracy of each stage on the forward holdout "
        "using mean absolute error (MAE), root mean squared error (RMSE) and R². (ii) Comparison of Stage 1 "
        "with two trivial references, a tehsil-by-season average of the training years and persistence of the "
        "previous round, because a model is only useful if it beats them. (iii) For Stage 2, a leave-one-tehsil-"
        "out test and the rank correlation between predicted and observed mean residuals across the 14 "
        "tehsils. (iv) Behavioural checks on the sandbox: the direction and magnitude of change when each "
        "slider is moved, and consistency of the no-change scenario with the dashboard status. No labelled "
        "ground truth exists for the attribution split, so it is not validated directly."))

# =====================================================================================
A(("ch", "SOFTWARE / HARDWARE REQUIRED"))
A(("h2", "4.1 SOFTWARE REQUIREMENTS"))
A(("tbl", "Table 4.1: Software used",
   ["Component", "Purpose", "Version used"],
   [["Python", "Language for data, models, API", "3.13"],
    ["scikit-learn", "RandomForestRegressor with monotonic_cst (needs ≥ 1.4)", "1.9.0"],
    ["pandas / NumPy", "Tabular data and numerics", "3.0.3 / 2.4.4"],
    ["joblib", "Model persistence", "1.5.3"],
    ["PyMuPDF", "Text extraction from the GSDA PDF tables", "1.28.2"],
    ["Shapely", "Point-in-polygon assignment of wells to tehsils", "2.1.2"],
    ["Requests", "NASA POWER download", "2.33.1"],
    ["FastAPI / Uvicorn / Pydantic", "REST service and validation", "0.141.1 / 0.50.0 / 2.13.5"],
    ["Chart.js (CDN) + vanilla JavaScript", "Dashboard charts and interactivity", "Browser-loaded"],
    ["Matplotlib", "Figures for this report", "3.10.8"],
    ["Playwright (system Chrome)", "Dashboard screenshots for this report", "Documentation only"],
    ["Git / GitHub", "Version control and hosting", "—"]],
   [2.1, 2.9, 1.25]))
A(("p", "Runtime requirements are listed in requirements.txt. No GPU, database or external API key is needed. "
        "The data-assembly scripts additionally need the GSDA PDFs on local disk and, for rainfall and "
        "temperature, internet access to NASA POWER."))
A(("h2", "4.2 HARDWARE REQUIREMENTS"))
A(("tbl", "Table 4.2: Hardware",
   ["Item", "Requirement", "Note"],
   [["Processor", "Any modern dual-core CPU", "Training takes well under a minute"],
    ["Memory", "4 GB minimum, 8 GB recommended", "Panel is < 1 MB; models ≈ 3 MB"],
    ["Storage", "≈ 1 GB if the source PDFs are kept", "The PDFs total several hundred MB"],
    ["Network", "Internet for map/chart CDN and NASA POWER", "The trained model runs offline"],
    ["Client", "Any current browser", "Responsive layout"]],
   [1.3, 2.5, 2.45]))
A(("p", "Development and all reported experiments were run on a Windows 11 laptop. Extracting the well "
        "tables from roughly 38,000 PDF pages takes a few minutes."))

# =====================================================================================
A(("ch", "RESULTS AND DISCUSSIONS"))
A(("h2", "5.1 DATA AND EXPERIMENTAL SETUP"))
A(("p", f"The panel contains {N_ROWS} tehsil-season rows from 2010 to 2024. The target (depth to water) has mean "
        f"{f2(DS['target_mean_m'])} m, standard deviation {f2(DS['target_std_m'])} m and range "
        f"{neg(DS['target_min_m'])}\u2013{f2(DS['target_max_m'])} m, typical of shallow unconfined aquifers (the slightly "
        "negative minimum is an artefact of the anomaly adjustment on a near-surface well). Figure 5.1 shows the "
        "season-balanced mean depth against annual rainfall. Depth increased steadily from 2013 to 2019 and "
        "recovered after 2020, which coincides with wetter years (about 1,390 mm a year on average in 2020–2024 against about 1,120 mm in 2014–2019); the data therefore contain a clear "
        "climate-driven episode. Figure 5.2 shows the number of wells behind each tehsil-year and the uneven "
        "coverage across tehsils."))
A(("fig", "fig_data_overview.png", "Figure 5.1: Season-balanced mean depth to water (all tehsils) and annual "
          "rainfall, 2010–2024", 5.6))
A(("fig", "fig_well_coverage.png", "Figure 5.2: Median number of wells behind each tehsil-year value", 5.6))
A(("p", "Models were fitted with the hyperparameters of Sections 3.4 and 3.5. The forward split uses rounds "
        f"before {SPLIT['train_before_year']} for training ({SPLIT['n_train']} rows) and "
        f"{SPLIT['train_before_year']}\u20132024 for testing ({SPLIT['n_test']} rows)."))

A(("h2", "5.2 STAGE 1 RESULTS"))
A(("tbl", f"Table 5.1: Forward holdout performance ({HOLD}, n = {SPLIT['n_test']})",
   ["Model", "MAE (m)", "RMSE (m)", "R²"],
   [["Stage 1: climate baseline", f3(B["mae"]), f3(B["rmse"]), f3(B["r2"])],
    ["Reference: tehsil-season average of training years", f3(CLIM["mae"]), "—", f3(CLIM["r2"])],
    ["Reference: previous round persisted", f3(PERS["mae"]), "—", neg(PERS["r2"])],
    ["Stage 2: residual extraction (forward)", f3(S2["mae"]), f3(S2["rmse"]), neg(S2["r2"])],
    ["Stage 2: leave-one-tehsil-out (rows)", "—", "—", f3(LO["r2_rows"])]],
   [3.1, 0.9, 0.9, 0.9]))
A(("p", f"Stage 1 explains {100 * B['r2']:.0f}% of the variance of the water table on the unseen years, with a "
        f"typical error of {f2(B['mae'])} m against a standard deviation of {f2(DS['target_std_m'])} m "
        "(Figure 5.3). **This must be read against the references.** A model that simply averages each "
        f"tehsil-season over the training years reaches R² = {f2(CLIM['r2'])} and MAE = {f2(CLIM['mae'])} m, "
        "marginally better in MAE than Stage 1. In other words, most of Stage 1's skill is seasonality and "
        "tehsil level; rainfall and temperature add no measurable accuracy over that on this holdout. "
        "Persistence of the previous round is far worse because consecutive rounds fall in different seasons."))
A(("fig", "fig_pred_vs_obs.png", "Figure 5.3: Stage 1 predicted versus observed depth on the forward holdout", 3.9))
A(("p", f"Figure 5.4 (left) gives the importances. Season (pre-monsoon, {f2(IMP1['season_Pre-Monsoon'])}) "
        f"and temperature ({f2(IMP1['temperature_c'])}) dominate; temperature is largely a proxy for season "
        f"here. Rainfall contributes {f2(IMP1['rainfall_mm'])}, and each tehsil indicator a few percent. The "
        "weather effect is nevertheless real and in the right direction, as the sandbox results show "
        "(Section 5.6). Figure 5.5 illustrates the decomposition for one tehsil: the cross-fitted baseline "
        "follows the seasonal cycle, and the gap between observed and baseline is the residual that Stage 2 "
        "tries to explain."))
A(("fig", "fig_importance.png", "Figure 5.4: Feature importance for Stage 1 (left) and Stage 2 (right)", 6.0))
A(("fig", "fig_katol_series.png", "Figure 5.5: Observed depth and out-of-fold climate baseline for Katol; shading marks "
          "positive residuals", 5.7))

A(("h2", "5.3 STAGE 2 RESULTS"))
A(("p", f"Stage 2 has **no useful out-of-time skill**: its forward R² is {neg(S2['r2'])} (Table 5.1), worse than "
        "predicting a constant. The explanation is the nature of its inputs. The three human-pressure features "
        "are linear interpolations between three assessment snapshots, so within a tehsil they change slowly and "
        "smoothly, while the residual on the holdout contains the post-2020 recovery in wet years that "
        "no slowly varying feature can explain. Stage 2 can learn level differences between tehsils but not "
        "year-to-year movement."))
A(("p", f"It does carry between-tehsil information. Predicting each tehsil from a model that never saw it gives "
        f"almost no row-level skill (R² = {f3(LO['r2_rows'])}), but the tehsil mean residuals are rank-correlated "
        f"with the observed ones (Spearman {f2(LO['tehsil_mean_spearman'])}, Pearson {f2(LO['tehsil_mean_pearson'])}, "
        "n = 14; Figure 5.6). With only 14 points this is suggestive, not conclusive. Importance within Stage 2 "
        f"is led by stage of extraction ({f2(IMP2['stage_of_extraction_pct'])}) and irrigation draft "
        f"({f2(IMP2['irrigation_draft_mm'])}), which is consistent with the hydrological expectation that "
        "irrigation is the dominant draft in this district."))
A(("fig", "fig_loto.png", "Figure 5.6: Leave-one-tehsil-out check of Stage 2 on tehsil mean residuals", 4.2))
A(("note", "Consequence for interpretation: the risk map ranks tehsils by a Stage 2 output whose between-tehsil "
           "ordering is moderately supported and whose within-tehsil, over-time behaviour is not. Rankings can "
           "be discussed; the exact residual values should not be quoted as measured extraction."))

A(("h2", "5.4 ATTRIBUTION RESULTS"))
A(("p", "Table 5.2 and Figure 5.7 give the per-tehsil status and natural/human attribution computed as in "
        "Section 3.6. Climate dominates the modelled change in most tehsils, with human extraction taking the "
        "larger share in Katol, Narkhed and Kuhi. Because the human features change only through "
        "interpolation, the human share reflects the size of the change in reported extraction between "
        "assessment years rather than observed pumping."))
A(("tbl", "Table 5.2: Latest-year status and attribution by tehsil",
   ["Tehsil", "Depth to water (m)", "Modelled human residual (m)", "Natural %", "Human %", "Risk category"],
   [[r["tehsil"], f2(r["latest_observed_water_table_m"]), neg(r["latest_human_residual_m"]),
     f"{r['natural_pct']:.1f}", f"{r['human_pct']:.1f}", r["risk_category"]] for r in STATUS],
   [1.15, 0.95, 1.2, 0.75, 0.75, 1.2]))
A(("fig", "fig_attribution.png", "Figure 5.7: Natural versus human share of the modelled change in water table", 5.2))

A(("h2", "5.5 RISK CLASSIFICATION RESULTS"))
A(("p", f"The fitted quartile thresholds on the latest-year modelled residual are q\u2082\u2085 = {neg(TH['q25'])} m, "
        f"q\u2085\u2080 = {neg(TH['q50'])} m and q\u2087\u2085 = {neg(TH['q75'])} m. They yield {COUNTS['Safe']} Safe, "
        f"{COUNTS['Semi-Critical']} Semi-Critical, {COUNTS['Critical']} Critical and {COUNTS['Over-Exploited']} "
        "Over-Exploited tehsils (Figure 5.8). Katol, Narkhed, Savner and Kalmeshwar are the highest-pressure "
        "tehsils; Mouda, Ramtek, Nagpur Urban and Bhiwapur the lowest. The residuals separating adjacent "
        "categories are a few tenths of a metre, and because the thresholds are quartiles of 14 values the "
        "balanced counts are guaranteed by construction. They are therefore a ranking, not evidence of "
        "absolute risk; the official categories derived from stage of extraction (for example Katol "
        "Semi-Critical in the 2023 assessment) should be consulted for statutory purposes."))
A(("fig", "fig_residual_risk.png", "Figure 5.8: Latest-year modelled human-attributable residual by tehsil, coloured by "
          "risk category; dotted lines are the quartile thresholds", 5.2))

A(("h2", "5.6 POLICY SANDBOX RESULTS"))
nu = lambda k: f2(SC[k]["nu_depth_10y"])
cnt = lambda k: [str(SC[k]["counts"][c]) for c in ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]]
A(("p", "Table 5.3 reports the district-wide effect of five scenarios and Figure 5.9 the ten-year projection "
        "for Nagpur Urban. Table 5.4 reports the average change in projected depth when each slider is moved, "
        "relative to no intervention, across all 14 tehsils. The no-intervention counts equal the status "
        "counts of Section 5.5, as designed."))
A(("tbl", "Table 5.3: Scenario outcomes (district risk counts; Nagpur Urban depth at the ten-year horizon)",
   ["Scenario", "Safe", "Semi-Crit.", "Critical", "Over-Expl.", "Nagpur Urban depth (m)"],
   [[k] + cnt(k) + [nu(k)] for k in ["No intervention", "Rainfall −20%", "Borewell growth +10%/yr",
                                      "Drip adoption 50%", "Drip 50% + borewell +10%/yr"]],
   [2.1, 0.55, 0.8, 0.7, 0.85, 1.25]))
A(("fig", "fig_scenarios.png", "Figure 5.9: Ten-year projection for Nagpur Urban under five policy scenarios", 5.4))
A(("tbl", "Table 5.4: Average change in projected depth to water versus no intervention (all tehsils, one-year horizon)",
   ["Slider setting", "Mean Δ depth (m)", "Range across tehsils (m)", "Stage responding"],
   [[k, sgn(SEN[k]["mean"]), f"{sgn(SEN[k]['min'])} to {sgn(SEN[k]['max'])}", "Stage 1" if "Rainfall" in k else "Stage 2"]
    for k in ["Rainfall −20%", "Rainfall −50%", "Rainfall +30%", "Borewell growth +20%/yr",
              "Drip adoption 50%", "Drip adoption 100%"]],
   [1.9, 1.2, 1.8, 1.35]))
A(("p", "The direction of every response is correct, which is the purpose of the monotonic constraints: less "
        "rainfall deepens the water table, more borewell growth deepens it, and more drip adoption makes it "
        f"shallower. A 20% rainfall deficit deepens the district-average water table by about "
        f"{f2(SEN['Rainfall −20%']['mean'])} m and a 50% deficit by about {f2(SEN['Rainfall −50%']['mean'])} m, so "
        "rainfall has a visible effect. Full drip adoption lowers the water table by about "
        f"{f2(-SEN['Drip adoption 100%']['mean'])} m on average and is the strongest lever. Two caveats apply. "
        "First, the magnitudes of the drip and borewell effects rest entirely on the assumed elasticities of "
        "Section 3.8 and carry no empirical validation. Second, risk categories depend only on Stage 2, so the "
        "rainfall slider moves depth but never changes a tehsil's category (Table 5.3, rows 1 and 2); a "
        "decision-maker should read the projected depth and the category together."))

A(("h2", "5.7 THE DASHBOARD"))
A(("p", "Figure 5.10 shows the Dashboard view: four risk-count chips, the tehsil map coloured by category, "
        "and a detail panel for the selected tehsil with a plain-language statement, a drawdown chart "
        "comparing observed or projected depth with the climate-only baseline, and the human/natural "
        "attribution donut. Figure 5.11 shows the Policy Sandbox, where the three sliders recolour the map "
        "and redraw the projection for the focused tehsil. Figure 5.12 shows the Data Sources view, which "
        "states which datasets are in use (the two GSDA/CGWB sources and NASA POWER) and which are not "
        "connected; an earlier version of this page marked every source as connected, which was corrected "
        "once the real data were integrated."))
A(("fig", "ui_dashboard_detail.png", "Figure 5.10: Dashboard view with a selected tehsil (Bhiwapur)", 6.1))
A(("fig", "ui_sandbox.png", "Figure 5.11: Policy Sandbox view with assumption sliders and district response map", 6.1))
A(("fig", "ui_sources.png", "Figure 5.12: Data Sources view", 6.1))

A(("h2", "5.8 DISCUSSION AND LIMITATIONS"))
A(("p", "The results support the architecture and the engineering: real public records can be assembled into "
        "a clean tehsil-season panel, the two stages can be built, constrained, persisted and served, the "
        "sandbox responds in the correct direction, and the interface presents the decomposition in plain "
        "language. They do not yet support strong claims about how much of Nagpur's depletion is human-"
        "driven. The limitations are listed here so that they are not discovered by a reader."))
A(("bul", [
    "**Stage 1 adds little over seasonality.** On the forward holdout it does not beat a tehsil-season average "
    "(Table 5.1). A richer climate representation (gauge rainfall, antecedent wetness, evapotranspiration) "
    "would be needed for the climate stage to earn its place.",
    "**Stage 2 has no out-of-time skill** and only moderate between-tehsil ranking skill, because the "
    "human-pressure features are three interpolated snapshots (Section 5.3).",
    "**Missing human-pressure data.** There are no borewell counts, crop areas, cropping patterns or "
    "agricultural power data in the model. The Minor Irrigation Census, district crop statistics and feeder "
    "consumption were not obtainable in time.",
    "**Shared Nagpur values.** Nagpur Urban and Rural share the 2011-12 and 2023 human-pressure values, so "
    "their difference rests on the 2024 report and on well locations alone.",
    "**Satellite rainfall.** NASA POWER is not a gauge record and runs somewhat wetter than the long-term gauge "
    "mean for Nagpur.",
    "**Changing well network.** The well fixed-effect adjustment reduces but cannot eliminate network effects; "
    "Bhiwapur and Kuhi have few usable wells, and 2020\u20132021 and 2024 have missing rounds.",
    "**Residual is not pure human signal.** It includes Stage 1 error and unmeasured natural variables.",
    "**Relative risk categories.** Quartile thresholds rank tehsils against each other; they are not the "
    "official stage-of-extraction categories and the balanced counts are by construction.",
    "**Assumed policy elasticities.** The 0.6 drip and 0.5 borewell coefficients are assumptions.",
    "**No uncertainty quantification.** Predictions are point estimates without intervals, and the dashboard "
    "draws projected curves without bands.",
]))

A(("h2", "5.9 ACHIEVEMENT OF OBJECTIVES"))
A(("tbl", "Table 5.5: Status of objectives",
   ["Objective", "Status", "Evidence"],
   [["1. Climate baseline model", "Achieved; no gain over seasonal average", "Table 5.1; Section 5.2"],
    ["2. Separate residual stage", "Implemented; weak out-of-time skill", "Table 5.1; Section 5.3"],
    ["3. Monotonic constraints", "Achieved", "Table 3.2; Table 5.4 directions"],
    ["4. Four-class risk map", "Achieved (relative thresholds)", "Table 5.2; Figure 5.8"],
    ["5. Policy Sandbox", "Achieved; effects rest on assumptions", "Tables 5.3\u20135.4; Figure 5.9"],
    ["6. REST service + dashboard", "Achieved", "Figures 5.10\u20135.12"],
    ["7. Real dataset and honest evaluation", "Achieved for water levels, climate and extraction snapshots; "
     "borewell, crop and power data missing", "Sections 3.2, 5.8"]],
   [2.2, 2.3, 1.75]))

# =====================================================================================
A(("ch", "CONCLUSION AND FUTURE SCOPE"))
A(("h2", "6.1 CONCLUSION"))
A(("p", "GroundVision ML demonstrates a practical way to make groundwater governance data more actionable: decompose "
        "the observed water table into a climate baseline and a human-attributable residual with two "
        "separate, constrained and interpretable models, classify tehsils in familiar categories, and let "
        "officials rehearse interventions on a live simulator. The system has been implemented end to end and "
        "now runs on real data: roughly ten thousand GSDA/CGWB well readings, NASA POWER climate, and "
        "taluka-level extraction figures from three official assessments."))
A(("p", f"On a forward-in-time holdout the climate baseline attains R\u00b2 = {f2(B['r2'])}, but no better than "
        "a tehsil-by-season average; the residual stage has no out-of-time skill, though it ranks the 14 tehsils "
        f"moderately well (Spearman {f2(LO['tehsil_mean_spearman'])}). The sandbox responds in the correct "
        "direction to every slider. The study therefore establishes the pipeline and a working decision-support "
        "interface, and it also shows precisely what is missing for the attribution to be trusted: "
        "year-by-year human-pressure data. Reporting this is part of the contribution, since a decision-support "
        "tool that hides its limits is more dangerous than one without a simulator."))
A(("p", "The central claim of the work is architectural: that attribution can be made structural by "
        "withholding human variables from the baseline, and that monotonic constraints make an interactive "
        "tree-based simulator behave sensibly. Whether the decomposition is hydrologically meaningful for "
        "Nagpur can be settled only with richer human-pressure data, which is the principal next step."))

A(("h2", "6.2 FUTURE SCOPE"))
A(("num", [
    "**Time-varying human-pressure data.** Add the 5th/6th Minor Irrigation Census borewell counts, district "
    "crop-area statistics converted to a crop-water index, and agricultural feeder power consumption from "
    "MSEDCL (which needs a direct request). Annual or seasonal extraction series would let Stage 2 learn "
    "year-to-year behaviour.",
    "**Gauge-based rainfall.** Replace or supplement NASA POWER with IMD gridded rainfall, and add "
    "antecedent-wetness and evapotranspiration features so that the climate stage can beat a seasonal average.",
    "**Real risk categories.** Calibrate to the official stage-of-extraction categories and use the "
    "assessments' recharge and extractable-resource figures directly.",
    "**Uncertainty.** Add prediction intervals, for example by quantile forests or conformal prediction, "
    "and show them on the drawdown chart.",
    "**Stronger Stage 2 design.** Test panel models with tehsil fixed effects, gradient-boosted trees "
    "with monotonic constraints, and a within-tehsil target such as the change in residual.",
    "**Calibrated policy effects.** Replace the assumed 0.6 and 0.5 elasticities with values from field trials "
    "or published water-use studies.",
    "**Validation against events.** Compare attributed shares in known drought years with known "
    "high-extraction years, and the risk ranking with the official categories.",
    "**Operations.** Deploy publicly, add authentication and report export, and automate data refresh "
    "and retraining; extend to other Vidarbha districts only after local validation.",
]))

# =====================================================================================
REFERENCES = [
    "S. Dangar, A. Asoka and V. Mishra, \u201cCauses and implications of groundwater depletion in India: A review,\u201d "
    "Journal of Hydrology, vol. 596, art. 126103, 2021.",
    "M. Rodell, I. Velicogna and J. S. Famiglietti, \u201cSatellite-based estimates of groundwater depletion in "
    "India,\u201d Nature, vol. 460, pp. 999\u20131002, 2009.",
    "Central Ground Water Board, Ministry of Jal Shakti, Government of India, Dynamic Ground Water Resource "
    "Assessment reports (national and state compilations), https://cgwb.gov.in.",
    "A. M. MacDonald et al., \u201cGroundwater quality and depletion in the Indo-Gangetic Basin mapped from "
    "in situ observations,\u201d Nature Geoscience, vol. 9, pp. 762\u2013766, 2016.",
    "J. S. Famiglietti, \u201cThe global groundwater crisis,\u201d Nature Climate Change, vol. 4, pp. 945\u2013948, 2014.",
    "A. Asoka, T. Gleeson, Y. Wada and V. Mishra, \u201cRelative contribution of monsoon precipitation and "
    "pumping to changes in groundwater storage in India,\u201d Nature Geoscience, vol. 10, pp. 109\u2013117, 2017.",
    "M. O. Cuthbert et al., \u201cGlobal patterns and dynamics of climate\u2013groundwater interactions,\u201d "
    "Nature Climate Change, vol. 9, pp. 137\u2013141, 2019.",
    "T. Gleeson, Y. Wada, M. F. P. Bierkens and L. P. H. van Beek, \u201cWater balance of global aquifers "
    "revealed by groundwater footprint,\u201d Nature, vol. 488, pp. 197\u2013200, 2012.",
    "T. Rajaee, H. Ebrahimi and V. Nourani, \u201cA review of the artificial intelligence methods in groundwater "
    "level modeling,\u201d Journal of Hydrology, vol. 572, pp. 336\u2013351, 2019.",
    "J. Zhang, Y. Zhu, X. Zhang, M. Ye and J. Yang, \u201cDeveloping a Long Short-Term Memory (LSTM) based model "
    "for predicting water table depth in agricultural areas,\u201d Journal of Hydrology, vol. 561, pp. 918\u2013929, 2018.",
    "S. Hochreiter and J. Schmidhuber, \u201cLong short-term memory,\u201d Neural Computation, vol. 9, no. 8, "
    "pp. 1735\u20131780, 1997.",
    "L. Breiman, \u201cRandom forests,\u201d Machine Learning, vol. 45, no. 1, pp. 5\u201332, 2001.",
    "J. H. Friedman, \u201cGreedy function approximation: a gradient boosting machine,\u201d Annals of Statistics, "
    "vol. 29, no. 5, pp. 1189\u20131232, 2001.",
    "F. Pedregosa et al., \u201cScikit-learn: Machine learning in Python,\u201d Journal of Machine Learning Research, "
    "vol. 12, pp. 2825\u20132830, 2011.",
    "C. Rudin, \u201cStop explaining black box machine learning models for high stakes decisions and use "
    "interpretable models instead,\u201d Nature Machine Intelligence, vol. 1, pp. 206\u2013215, 2019.",
    "D. S. Pai et al., \u201cDevelopment of a new high spatial resolution (0.25\u00b0 \u00d7 0.25\u00b0) long period "
    "(1901\u20132010) daily gridded rainfall data set over India,\u201d Mausam, vol. 65, no. 1, pp. 1\u201318, 2014.",
    "A. H. Sparks, \u201cnasapower: A NASA POWER global meteorology, surface solar energy and climatology data "
    "client for R,\u201d Journal of Open Source Software, vol. 3, no. 30, art. 1035, 2018.",
    "NASA Langley Research Center, POWER Project data service, https://power.larc.nasa.gov.",
    "Groundwater Surveys and Development Agency (Pune) and Central Ground Water Board, Central Region (Nagpur), "
    "Report on the Dynamic Ground Water Resources of Maharashtra (2011\u20132012), February 2014.",
    "Groundwater Surveys and Development Agency (Pune) and Central Ground Water Board, Central Region (Nagpur), "
    "Report on the Dynamic Groundwater Resources of Maharashtra 2023, March 2024.",
    "Groundwater Surveys and Development Agency (Pune) and Central Ground Water Board, Central Region (Nagpur), "
    "Report on Dynamic Ground Water Resources of Maharashtra 2024, 2025.",
    "Groundwater Surveys and Development Agency, Maharashtra, Depth to ground water level tables for the January, "
    "pre-monsoon (May), August and post-monsoon (November) rounds, 1994\u20132025, https://gsda.maharashtra.gov.in.",
]
REF_NOTE = ("Reference check: entries [1], [6] and [9] were confirmed against publisher listings during "
            "preparation; the remaining entries are standard, well-known works cited from memory, and [19]\u2013[22] "
            "are the GSDA/CGWB documents used as data sources. All entries, including page numbers, should be "
            "verified against the originals before submission.")

# =====================================================================================
APPENDIX_A = """git clone https://github.com/aarshnomorecool/GLOWkrishna.git
cd GroundVision MLkrishna
pip install -r requirements.txt

# rebuild the real dataset (source PDFs are read from D:/NAKSHA/research-paper; use --src to change)
python scripts/extract_gsda_waterlevels.py
python scripts/extract_gsda_assessments.py
python scripts/fetch_nasa_power.py
python scripts/build_real_dataset.py

# retrain both stages (writes models/ and models/metrics.json)
python -m src.train

# launch the API and dashboard
uvicorn src.api:app --reload --port 8000
# open http://localhost:8000"""

APPENDIX_LAYOUT = """src/config.py            tehsil registry, schema, risk categories
src/features.py          encodings and monotonic constraint definitions
src/baseline_model.py    Stage 1 climate baseline (forward split + out-of-fold)
src/residual_model.py    Stage 2 residual extraction (+ leave-one-tehsil-out)
src/risk.py              quartile-threshold risk classification
src/simulate.py          Policy Sandbox scenario engine
src/train.py             end-to-end training pipeline, writes metrics.json
src/api.py               FastAPI service
frontend/                index.html, styles.css, app.js
scripts/                 extract_gsda_*.py, fetch_nasa_power.py, build_real_dataset.py
data/ , models/          raw + processed tables, boundaries, joblib artefacts"""

APPENDIX_B = [
    ["GET /", "Serves the dashboard"],
    ["GET /api/tehsils", "Baseline scores, categories, attribution splits and drawdown for 14 tehsils"],
    ["POST /api/simulate", "Body: rainfall, drip, borewell, tehsil_id. Returns scenario score, drawdown "
                           "projection and updated scores for all tehsils"],
    ["GET /assets/nagpur_tehsils.geojson", "Tehsil boundaries"],
]

APPENDIX_C1 = """_BASELINE_DIRECTIONS = {
    "rainfall_mm": -1, "temperature_c": 1,
    "recharge_proxy": -1, "rainfall_mm_lag1": -1,
}
_RESIDUAL_DIRECTIONS = {
    "irrigation_draft_mm": 1,
    "domestic_industrial_draft_mm": 1,
    "stage_of_extraction_pct": 1,
}
model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3,
        monotonic_cst=baseline_monotonic_constraints(), random_state=42, n_jobs=-1)"""

APPENDIX_C2 = """def _apply_scenario_rows(rows, inputs, years_forward):
    df = rows.copy()
    rain_factor = max(0.0, 1 + inputs.rainfall_pct_change / 100.0)
    for col in ("rainfall_mm", "rainfall_mm_lag1", "recharge_proxy"):
        df[col] = rows[col] * rain_factor
    irr_factor = (1 - 0.6 * inputs.drip_irrigation_adoption_pct / 100.0) * \\
                 (1 + 0.5 * inputs.borewell_growth_pct / 100.0) ** years_forward
    old_total = rows["irrigation_draft_mm"] + rows["domestic_industrial_draft_mm"]
    df["irrigation_draft_mm"] = rows["irrigation_draft_mm"] * max(0.0, irr_factor)
    new_total = df["irrigation_draft_mm"] + rows["domestic_industrial_draft_mm"]
    df["stage_of_extraction_pct"] = (rows["stage_of_extraction_pct"] * new_total / old_total).clip(0, 150)
    return df"""

APPENDIX_PLACEHOLDERS = [
    ("D. CERTIFICATES OF PATENT / COPYRIGHT / PAPER PUBLICATION / PAPER PRESENTED",
     "[Insert certificates here, if any.]"),
    ("E. CO-PO AND PSO MAPPING", "[Insert the department's CO-PO and PSO mapping table here.]"),
    ("F. PHOTO WITH DETAILS", "[Insert project photograph with details here.]"),
    ("G. PHOTO WITH GUIDE", "[Insert photograph with the project guide here.]"),
    ("H. PLAGIARISM REPORT", "[Insert plagiarism report here.]"),
]


# ---------------------------------------------------------------------------------------
# Index page: generated from the body so it can never disagree with the headings
def _title(s):
    small = {"and", "of", "the", "to", "for", "in", "a", "versus"}
    words = s.lower().split()
    return " ".join(w if (i and w in small) else w.capitalize() for i, w in enumerate(words)).replace("Ph", "pH")


def _build_index():
    idx, cur, n = [], None, 0
    for it in BODY:
        if it[0] == "ch":
            n += 1
            cur = (f"CHAPTER {n}: {it[1]}", [])
            idx.append(cur)
        elif it[0] == "h2" and cur is not None:
            m = re.match(r"(\d+\.\d+)\s+(.*)", it[1])
            cur[1].append(f"{m.group(1)} {_title(m.group(2))}")
    idx.append(("REFERENCES", []))
    idx.append(("APPENDIX", ["A. Repository Layout and Run Commands", "B. REST API Summary", "C. Key Source Excerpts",
                             "D\u2013H. Certificates, CO-PO and PSO Mapping, Photographs, Plagiarism Report"]))
    return idx

# ---------------------------------------------------------------------------------------
# Gap-filling disclosure and row counts (applied after BODY is built)
N_MEAS, N_ALL, N_GAP = DS["rows"], DS["rows_total_with_gap_filled"], DS["gap_filled_rows"]


def _find(pred):
    for i, it in enumerate(BODY):
        if pred(it):
            return i
    raise AssertionError("anchor not found")


def _sub(prefix, old, new):
    i = _find(lambda it: it[0] == "p" and it[1].startswith(prefix))
    assert old in BODY[i][1], (prefix, old)
    BODY[i] = ("p", BODY[i][1].replace(old, new))


_sub("All downstream code depends", "Every input is real public data; nothing in the table is simulated.",
     "Every input is real public data; the only derived values are interpolated or gap-filled ones, "
     "described in Sections 3.2.3 and 3.2.4.")
_sub("**Adjusting for a changing well network.**",
     "Cells with too few wells, and the rounds that simply were not published (for example May 2021 and November 2024), are left missing rather than filled.",
     "Cells with too few wells, and the rounds that were not published (for example May 2021 and November 2024), "
     "are gap-filled automatically as described in Section 3.2.4.")
i = _find(lambda it: it[0] == "h3" and it[1].startswith("3.2.4"))
BODY[i] = ("h3", "3.2.4 Gap-filling and the assembled panel")
j = _find(lambda it: it[0] == "p" and it[1].startswith("The final table (scripts/build_real_dataset.py)"))
BODY[j] = ("p", (
    f"A tehsil-season with no usable well reading ({N_GAP + 1} of the 840 cells, including the whole May 2021 and "
    "November 2024 rounds) is filled automatically from real neighbouring measurements: the tehsil's own average "
    "for that season plus the district-wide anomaly of that round, taken as the median anomaly of the tehsils that "
    "were measured. Where fewer than three tehsils were measured in a round, the anomaly of that season is "
    "interpolated linearly across years. This keeps the pipeline free of holes, so the dashboard and the sandbox "
    "always return a value. The filled cells are recorded in a gap_filled column and are **never used as training "
    "or test targets**, so every reported score refers to measured data only. Depth is clipped at zero. Gap-filled "
    "values are derived from measured neighbours and cannot capture a purely local event; this is stated as a "
    "limitation in Section 5.8."))
BODY.insert(j + 1, ("p", (
    f"The final table (scripts/build_real_dataset.py) has {N_ALL} tehsil-season rows from {DS['year_min']} to "
    f"{DS['year_max']}: {N_MEAS} with a measured water level and {N_GAP} gap-filled (each tehsil's first round is "
    f"dropped because it has no lag). The median measured cell is based on {D['median_wells_per_cell']:.0f} wells. "
    f"Coverage is uneven: Bhiwapur ({D['rows_per_tehsil']['Bhiwapur']} measured rows) and Kuhi "
    f"({D['rows_per_tehsil']['Kuhi']}) have few wells with long histories, while Nagpur Urban has by far the "
    "densest network (Figure 5.2).")))
_sub("Four layers of evaluation", "so it is not validated directly.",
     "so it is not validated directly. All scores use measured rows only; gap-filled rows are never training "
     "or test targets.")
_sub("The panel contains", f"The panel contains {N_MEAS} tehsil-season rows from 2010 to 2024.",
     f"The panel contains {N_ALL} tehsil-season rows from 2010 to 2024, of which {N_MEAS} have a measured water "
     f"level and {N_GAP} are gap-filled; all statistics and scores below use the {N_MEAS} measured rows.")
_sub("The panel contains", "(the slightly negative minimum is an artefact of the anomaly adjustment on a near-surface well)",
     "(depth is clipped at zero, since it cannot be negative)")
k = _find(lambda it: it[0] == "bul" and any("Changing well network" in b for b in it[1]))
bl = list(BODY[k][1])
for n, b in enumerate(bl):
    if "Changing well network" in b:
        bl[n] = ("**Changing well network.** The well fixed-effect adjustment reduces but cannot eliminate network "
                 "effects; Bhiwapur and Kuhi have few usable wells.")
bl.insert(0, ("**Gap-filled values are not measurements.** About 9% of the rows (including the whole May 2021 and "
              "November 2024 rounds) are filled from measured neighbours. They keep the dashboard complete but "
              "cannot capture a purely local event, and they are excluded from training and scoring."))
BODY[k] = ("bul", bl)
ABSTRACT[2] = ABSTRACT[2] + (" Rounds with no readings are gap-filled automatically from measured neighbours and "
                             "are excluded from training and scoring.")


INDEX = _build_index()
