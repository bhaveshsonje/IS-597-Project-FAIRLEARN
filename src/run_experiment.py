"""
run_experiment.py — End-to-end pipeline for March 23 milestone.

Usage:
    python src/run_experiment.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import build_dataset
from features import get_feature_matrix, FAIRNESS_FEATURES
from models import run_training, split_data
from fairness import run_fairness_analysis
from mitigation import run_mitigation
from explain import run_shap_explanations

DATA_DIR = "Dataset"
RESULTS_DIR = "results"


def main():
    print("=== IS 597 — March 23 Milestone ===\n")

    # 1. Load and clean data
    print("Loading and merging dataset...")
    df = build_dataset(DATA_DIR)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}\n")

    # 2. Feature engineering
    print("Building feature matrix...")
    X, y = get_feature_matrix(df)
    print(f"Features: {list(X.columns)}\n")

    # 3. Train baseline models + evaluate
    models, metrics, splits = run_training(X, y, RESULTS_DIR)
    X_train, X_test, y_train, y_test = splits

    # Raw demographic columns for fairness / mitigation
    train_idx = y_train.index
    test_idx  = y_test.index
    X_test_raw  = df.loc[test_idx,  FAIRNESS_FEATURES]
    X_train_raw = df.loc[train_idx, FAIRNESS_FEATURES]

    # 4. Fairness analysis
    print()
    run_fairness_analysis(models, X_test_raw, X_test, y_test, RESULTS_DIR)

    # 5. Mitigation (reweighting + thresholding)
    run_mitigation(
        base_models=models,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        df_train=X_train_raw.reset_index(drop=True),
        df_test=X_test_raw.reset_index(drop=True),
        results_dir=RESULTS_DIR,
    )

    # 6. SHAP explanations on Random Forest
    run_shap_explanations(
        rf_model=models["random_forest"],
        X_test=X_test,
        y_test=y_test.reset_index(drop=True),
        feature_names=list(X.columns),
        results_dir=RESULTS_DIR,
        n_students=5,
    )

    print(f"\nDone. All results saved to /{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
