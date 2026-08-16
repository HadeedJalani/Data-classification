from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_dataset(path):
    df = pd.read_csv(path)
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 150:
        raise ValueError(f"Expected 150 samples, found {len(df)}")
    if df[TARGET_COLUMN].nunique() != 3:
        raise ValueError("Expected exactly 3 classes")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature data contains missing values")
    return df


def split_dataset(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )
