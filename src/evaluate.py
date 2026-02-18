import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}
