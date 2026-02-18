import os
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. Put your CSV at data/raw/housing.csv "
            "or update RAW_DATA_PATH in src/config.py."
        )
    return pd.read_csv(path)
