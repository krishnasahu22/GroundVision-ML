"""
Stage 1 — Climate Baseline Model.

Trained on natural features only (rainfall, temperature, recharge proxy,
local hydrogeology via tehsil identity, season). Predicts what the water
table "should" be under natural conditions alone. Never sees any
anthropogenic feature — that separation is the core technical claim of
GroundVision ML and has to hold in code, not just in the pitch (see claude.md's
Coding Conventions).

Evaluation is forward-in-time: the model is fitted on years before
TEST_START_YEAR and scored on the later years. The persisted model is then
refitted on all rows. Cross-fitted (out-of-fold, grouped by year) predictions
are also returned so Stage 2 is trained on honest residuals rather than on
residuals of rows the baseline has already memorised.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.config import MODELS_DIR, TARGET
from src.features import add_encodings, baseline_feature_columns, baseline_monotonic_constraints

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
TEST_START_YEAR = 2021          # train on < 2021, test on 2021-2024
OOF_FOLDS = 5


@dataclass
class BaselineFitResult:
    model: RandomForestRegressor      # refitted on all rows (this is what is persisted)
    mae: float                        # forward-in-time holdout
    rmse: float
    r2: float
    n_train: int
    n_test: int
    feature_columns: list
    oof_pred: np.ndarray              # cross-fitted predictions for every row (aligned with df)


def _new_model(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        monotonic_cst=baseline_monotonic_constraints(),
        random_state=random_state,
        n_jobs=-1,
    )


def train_baseline_model(df: pd.DataFrame, random_state: int = 42) -> BaselineFitResult:
    encoded = add_encodings(df)
    feature_cols = baseline_feature_columns()
    X = encoded[feature_cols]
    y = encoded[TARGET]
    years = encoded["year"]

    train_mask = years < TEST_START_YEAR
    holdout = _new_model(random_state).fit(X[train_mask], y[train_mask])
    preds = holdout.predict(X[~train_mask])
    mae = mean_absolute_error(y[~train_mask], preds)
    rmse = mean_squared_error(y[~train_mask], preds) ** 0.5
    r2 = r2_score(y[~train_mask], preds)

    oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=OOF_FOLDS).split(X, y, groups=years):
        fold = _new_model(random_state).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = fold.predict(X.iloc[te])

    final = _new_model(random_state).fit(X, y)
    return BaselineFitResult(
        model=final, mae=mae, rmse=rmse, r2=r2,
        n_train=int(train_mask.sum()), n_test=int((~train_mask).sum()),
        feature_columns=feature_cols, oof_pred=oof,
    )


def predict_baseline(model: RandomForestRegressor, feature_cols: list, df_encoded: pd.DataFrame):
    return model.predict(df_encoded[feature_cols])


def save_baseline_model(model: RandomForestRegressor, feature_cols: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_cols}, BASELINE_MODEL_PATH)


def load_baseline_model():
    payload = joblib.load(BASELINE_MODEL_PATH)
    return payload["model"], payload["feature_columns"]
