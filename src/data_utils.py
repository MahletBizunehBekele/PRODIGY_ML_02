import pandas as pd
from pathlib import Path

def smart_read(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(p)
    elif suffix in ('.xlsx', '.xls', '.xlsm', '.excel'):
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file: {path}")

def load_train_test(train_path, test_path):
    train_df = smart_read(train_path)
    test_df = smart_read(test_path)
    return train_df, test_df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))
