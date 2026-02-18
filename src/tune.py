from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    RAW_DATA_PATH, TARGET_COL, MODEL_OUT_PATH, METRICS_OUT_PATH,
    TEST_SIZE, RANDOM_SEED
)
from src.data import load_data
from src.evaluate import regression_metrics
from src.utils import ensure_dir, save_json


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )


def main() -> None:
    df = load_data(RAW_DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
    ])

    # Good “internship-tier” grid: small enough to run, meaningful enough to matter
    param_grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.7],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    gs.fit(X_train, y_train)

    best_pipe: Pipeline = gs.best_estimator_
    preds = best_pipe.predict(X_test)
    holdout = regression_metrics(y_test, preds)

    # Convert neg RMSE back to positive
    tuned_cv_rmse = float(-gs.best_score_)

    results = {
        "tuned_random_forest": {
            **holdout,
            "cv_rmse_best": tuned_cv_rmse,
            "best_params": gs.best_params_,
        }
    }

    ensure_dir("models")
    joblib.dump(best_pipe, MODEL_OUT_PATH)

    ensure_dir("reports")
    save_json(results, METRICS_OUT_PATH)

    print("Tuning complete.")
    print(results)


if __name__ == "__main__":
    main()
