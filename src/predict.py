from __future__ import annotations

import joblib
import pandas as pd
from src.config import MODEL_OUT_PATH

def load_model():
    return joblib.load(MODEL_OUT_PATH)

def predict_from_dict(row: dict) -> float:
    model = load_model()
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    return float(pred)

if __name__ == "__main__":
    # Example usage (replace keys with your dataset columns)
    example = {
        "OverallQual": 7,
        "GrLivArea": 1700,
        "Neighborhood": "NAmes",
        "YearBuilt": 2003,
    }
    print(predict_from_dict(example))
