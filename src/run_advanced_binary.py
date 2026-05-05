"""
run_advanced_binary.py — All four models (LR, RF, XGBoost, MLP) in BINARY mode
                         with full fairness analysis, mitigation, and SHAP.

Pairs with run_advanced.py (multiclass). This pipeline gives the binary numbers
needed to run the existing reweighting + thresholding mitigation on XGBoost,
which the original binary pipeline (run_experiment.py) doesn't cover.

Usage:
    python src/run_advanced_binary.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import build_dataset
from features import get_feature_matrix, FAIRNESS_FEATURES
from models import run_training_all
from fairness import run_fairness_analysis
from mitigation import run_mitigation
from explain import run_shap_explanations

DATA_DIR    = "Dataset"
RESULTS_DIR = "results/advanced_binary"


def main():
    print("=== IS 597 — Phase 3: All Models, Binary Mode ===\n")

    # 1. Load data in binary mode
    print("Loading dataset (binary mode)...")
    df = build_dataset(DATA_DIR, mode="binary")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}\n")

    # 2. Feature engineering
    print("Building feature matrix...")
    X, y = get_feature_matrix(df)
    print(f"Features: {list(X.columns)}\n")

    # 3. Train all four models (no class balancing in binary — already 47/53)
    models, _, splits = run_training_all(X, y, RESULTS_DIR, balanced=False)
    X_train, X_test, y_train, y_test = splits

    # 4. Raw demographic columns for fairness / mitigation
    train_idx = y_train.index
    test_idx  = y_test.index
    X_test_raw  = df.loc[test_idx,  FAIRNESS_FEATURES]
    X_train_raw = df.loc[train_idx, FAIRNESS_FEATURES]

    # 5. Fairness analysis on all four
    print()
    run_fairness_analysis(models, X_test_raw, X_test, y_test, RESULTS_DIR)

    # 6. Mitigation: reweighting (LR, RF, XGBoost) + thresholding (all four)
    run_mitigation(
        base_models=models,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        df_train=X_train_raw.reset_index(drop=True),
        df_test=X_test_raw.reset_index(drop=True),
        results_dir=RESULTS_DIR,
        include_xgboost_rw=True,
    )

    # 7. SHAP on XGBoost (binary: explain class 1 = Success)
    run_shap_explanations(
        tree_model=models["xgboost"],
        X_test=X_test,
        y_test=y_test.reset_index(drop=True),
        feature_names=list(X.columns),
        results_dir=RESULTS_DIR,
        n_students=5,
        # class_labels=None → defaults to binary {0:'At-Risk', 1:'Success'}
        explain_class=1,
    )

    print(f"\nDone. All results saved to /{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
