from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_clean_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def make_train_test_split(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: list,
    test_size: float,
    seed: int,
    stratify: bool,
) -> SplitData:
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    strat = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=strat
    )

    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
