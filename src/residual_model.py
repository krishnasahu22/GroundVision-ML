"""
Stage 2 — Residual Extraction Model.

  residual = observed_water_table - climate_baseline_prediction

The residual (the part of the water table that climate alone does not
explain) is regressed against anthropogenic proxies only. The baseline
prediction used to form the residual is the cross-fitted (out-of-fold)
prediction from Stage 1, so the residual is not artificially shrunk by
Stage 1 having seen the same rows.

Evaluation is forward-in-time (same split year as Stage 1); the persisted
model is refitted on all rows.
"""

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.baseline_model import TEST_START_YEAR
from src.config import MODELS_DIR
from src.features import add_encodings, residual_feature_columns, residual_monotonic_constraints

RESIDUAL_MODEL_PATH = MODELS_DIR / "residual_model.joblib"


@dataclass
class ResidualFitResult:
    model: RandomForestRegressor
    mae: float
    rmse: float
    r2: float
    n_train: int
    n_test: int
    feature_columns: list


def compute_residual(df: pd.DataFrame, baseline_pred) -> pd.Series:
    observed = df[["observed_water_table_m"]].reset_index(drop=True)["observed_water_table_m"]
    return observed - pd.Series(baseline_pred, index=observed.index)


def _new_model(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        monotonic_cst=residual_monotonic_constraints(),
        random_state=random_state,
        n_jobs=-1,
    )


def train_residual_model(df: pd.DataFrame, baseline_pred, random_state: int = 42) -> ResidualFitResult:
    encoded = add_encodings(df)
    feature_cols = residual_feature_columns()
    X = encoded[feature_cols]
    y = compute_residual(df, baseline_pred)
    years = encoded["year"]

    train_mask = years < TEST_START_YEAR
    holdout = _new_model(random_state).fit(X[train_mask], y[train_mask])
    preds = holdout.predict(X[~train_mask])
    mae = mean_absolute_error(y[~train_mask], preds)
    rmse = mean_squared_error(y[~train_mask], preds) ** 0.5
    r2 = r2_score(y[~train_mask], preds)

    final = _new_model(random_state).fit(X, y)
    return ResidualFitResult(
        model=final, mae=mae, rmse=rmse, r2=r2,
        n_train=int(train_mask.sum()), n_test=int((~train_mask).sum()), feature_columns=feature_cols,
    )


def leave_one_tehsil_out(df: pd.DataFrame, baseline_pred, random_state: int = 42):
    """Predict each tehsil's residual from a model that never saw that tehsil. Tests whether the
    human-pressure proxies explain *between-tehsil* differences (which is what the risk ranking uses)."""
    import numpy as np
    encoded = add_encodings(df)
    X = encoded[residual_feature_columns()]
    y = compute_residual(df, baseline_pred)
    pred = np.zeros(len(df))
    for t in df["tehsil"].unique():
        te = (df["tehsil"] == t).to_numpy()
        m = _new_model(random_state).fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    return pred, y


def predict_residual(model: RandomForestRegressor, feature_cols: list, df_encoded: pd.DataFrame):
    return model.predict(df_encoded[feature_cols])


def save_residual_model(model: RandomForestRegressor, feature_cols: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_cols}, RESIDUAL_MODEL_PATH)


def load_residual_model():
    payload = joblib.load(RESIDUAL_MODEL_PATH)
    return payload["model"], payload["feature_columns"]
