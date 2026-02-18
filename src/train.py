from __future__ import annotations

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
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


def cv_metrics(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    5-fold CV on the training split only (to avoid leaking the test set).
    Returns mean/std for RMSE and R^2.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    # sklearn returns NEGATIVE RMSE for loss metrics (higher is better),
    # so we flip the sign back to get positive RMSE values.
    rmse_vals = -scores["test_rmse"]
    r2_vals = scores["test_r2"]

    return {
        "cv_rmse_mean": float(rmse_vals.mean()),
        "cv_rmse_std": float(rmse_vals.std()),
        "cv_r2_mean": float(r2_vals.mean()),
        "cv_r2_std": float(r2_vals.std()),
    }


def train_and_eval(df: pd.DataFrame) -> dict:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Hold-out split (keep this for a final “realistic” evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
    }

    results: dict = {}
    best_name = None
    best_holdout_rmse = float("inf")
    best_pipeline = None

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])

        # 1) Cross-validation on training split
        cv_res = cv_metrics(pipe, X_train, y_train)

        # 2) Fit on full training split, evaluate on hold-out test split
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        holdout_res = regression_metrics(y_test, preds)

        # Store everything
        results[name] = {**holdout_res, **cv_res}

        # Pick best model by hold-out RMSE (could also use cv_rmse_mean)
        if holdout_res["rmse"] < best_holdout_rmse:
            best_holdout_rmse = holdout_res["rmse"]
            best_name = name
            best_pipeline = pipe

    ensure_dir("models")
    joblib.dump(best_pipeline, MODEL_OUT_PATH)

    results["best_model"] = best_name
    return results


def main() -> None:
    df = load_data(RAW_DATA_PATH)
    results = train_and_eval(df)
    save_json(results, METRICS_OUT_PATH)
    print("Training complete.")
    print(results)


if __name__ == "__main__":
    main()
