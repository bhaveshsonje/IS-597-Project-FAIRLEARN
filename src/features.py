"""
features.py — Feature selection and encoding for baseline models.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks",
    "num_activities",
    "active_days",
    "avg_early_score",
    "num_assessments_taken",
]

CATEGORICAL_FEATURES = [
    "gender",
    "age_band",
    "highest_education",
    "disability",
]

FAIRNESS_FEATURES = ["gender", "disability", "age_band"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical features in place."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X (feature matrix) and y (target vector)."""
    df_enc = encode_categoricals(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    feature_cols = [c for c in feature_cols if c in df_enc.columns]
    X = df_enc[feature_cols]
    y = df_enc["target"]
    return X, y
