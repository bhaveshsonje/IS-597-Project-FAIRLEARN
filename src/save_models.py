"""
save_models.py — Train all 4 models in binary AND multiclass mode and persist to disk.

The Streamlit dashboard loads these at startup instead of retraining every time.

Outputs saved to models/:
  logistic_regression.pkl      ← binary
  random_forest.pkl
  xgboost.pkl
  mlp.pkl
  logistic_regression_mc.pkl   ← multiclass (4 classes)
  random_forest_mc.pkl
  xgboost_mc.pkl
  mlp_mc.pkl
  encoders.pkl                 ← shared LabelEncoders (same features both modes)
  test_data.pkl                ← binary test split
  test_data_mc.pkl             ← multiclass test split

Usage:
    python src/save_models.py
"""

import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import build_dataset
from features import get_feature_matrix, FAIRNESS_FEATURES, CATEGORICAL_FEATURES
from models import train_all_models, split_data
from sklearn.preprocessing import LabelEncoder

DATA_DIR   = "Dataset"
MODELS_DIR = "models"


def main():
    print("=== Saving models for Streamlit dashboard ===\n")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Shared: fit LabelEncoders on full binary df (same columns both modes) ──
    print("Loading dataset (binary mode) for encoders...")
    df_bin = build_dataset(DATA_DIR, mode="binary")

    print("Fitting and saving category encoders...")
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df_bin.columns:
            le = LabelEncoder()
            le.fit(df_bin[col].astype(str))
            encoders[col] = le
    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")
    joblib.dump(encoders, enc_path)
    print(f"  Saved: {enc_path}")
    for col, le in encoders.items():
        print(f"    {col}: {list(le.classes_)}")

    # ── Binary models ──────────────────────────────────────────────────────────
    print("\nBuilding binary feature matrix...")
    X, y = get_feature_matrix(df_bin)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_test_raw = df_bin.loc[y_test.index, FAIRNESS_FEATURES]

    print("Training all 4 binary models...")
    trained_bin = train_all_models(X_train, y_train, balanced=False)
    for name, model in trained_bin.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved: {path}")

    test_path = os.path.join(MODELS_DIR, "test_data.pkl")
    joblib.dump({
        "X_test":        X_test,
        "y_test":        y_test.reset_index(drop=True),
        "X_test_raw":    X_test_raw.reset_index(drop=True),
        "feature_names": feature_names,
        "df":            df_bin,
    }, test_path)
    print(f"  Saved: {test_path}")

    # ── Multiclass models ──────────────────────────────────────────────────────
    print("\nLoading dataset (multiclass mode)...")
    df_mc = build_dataset(DATA_DIR, mode="multiclass")

    print("Building multiclass feature matrix...")
    X_mc, y_mc = get_feature_matrix(df_mc)
    X_train_mc, X_test_mc, y_train_mc, y_test_mc = split_data(X_mc, y_mc)
    X_test_raw_mc = df_mc.loc[y_test_mc.index, FAIRNESS_FEATURES]

    print("Training all 4 multiclass models (balanced)...")
    trained_mc = train_all_models(X_train_mc, y_train_mc, balanced=True)
    for name, model in trained_mc.items():
        path = os.path.join(MODELS_DIR, f"{name}_mc.pkl")
        joblib.dump(model, path)
        print(f"  Saved: {path}")

    test_path_mc = os.path.join(MODELS_DIR, "test_data_mc.pkl")
    joblib.dump({
        "X_test":        X_test_mc,
        "y_test":        y_test_mc.reset_index(drop=True),
        "X_test_raw":    X_test_raw_mc.reset_index(drop=True),
        "feature_names": feature_names,
        "df":            df_mc,
    }, test_path_mc)
    print(f"  Saved: {test_path_mc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
