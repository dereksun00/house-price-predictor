from __future__ import annotations

import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.config import MODEL_OUT_PATH
from src.utils import ensure_dir


def main() -> None:
    pipe = joblib.load(MODEL_OUT_PATH)

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    # Works for OneHotEncoder + numeric passthrough
    feature_names = prep.get_feature_names_out()
    importances = model.feature_importances_

    idx = np.argsort(importances)[::-1][:15]
    top_names = feature_names[idx]
    top_vals = importances[idx]

    ensure_dir("reports")

    plt.figure()
    plt.barh(top_names[::-1], top_vals[::-1])
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=200)
    plt.close()

    print("Saved reports/feature_importance.png")


if __name__ == "__main__":
    main()
