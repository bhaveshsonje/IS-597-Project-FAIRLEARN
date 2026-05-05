"""
run_multiclass.py — Phase 1: 4-class student outcome prediction (LR + RF only).

Classes: Pass=0, Distinction=1, Fail=2, Withdrawn=3

NOTE — superseded by run_advanced.py, which trains the same LR + RF
plus XGBoost and MLP on the same data, producing identical numbers for
LR and RF. Kept here as the original Phase 1 entry point for project
history; new analyses should use run_advanced.py instead.

Usage:
    python src/run_multiclass.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import build_dataset, CLASS_LABELS
from features import get_feature_matrix, FAIRNESS_FEATURES
from models import run_training
from fairness import run_fairness_analysis_multiclass
from explain import run_shap_explanations

DATA_DIR   = "Dataset"
RESULTS_DIR = "results/multiclass"


def main():
    print("=== IS 597 — Phase 1: Multi-Class Classification ===\n")

    # 1. Load data in multiclass mode
    print("Loading dataset (multiclass mode)...")
    df = build_dataset(DATA_DIR, mode="multiclass")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:")
    counts = df["target"].value_counts().sort_index()
    for cls_int, n in counts.items():
        print(f"  {CLASS_LABELS[cls_int]} ({cls_int}): {n} ({n/len(df)*100:.1f}%)")
    print()

    # 2. Feature engineering (same features as binary)
    print("Building feature matrix...")
    X, y = get_feature_matrix(df)
    print(f"Features: {list(X.columns)}\n")

    # 3. Train baseline models + evaluate
    # evaluate() auto-detects multiclass and uses OVR AUC
    models, _, splits = run_training(X, y, RESULTS_DIR)
    _, X_test, _, y_test = splits

    # 4. Raw demographic columns for fairness
    X_test_raw = df.loc[y_test.index, FAIRNESS_FEATURES]

    # 5. Multiclass fairness analysis
    print()
    run_fairness_analysis_multiclass(
        models=models,
        X_test_raw=X_test_raw,
        X_test_enc=X_test,
        y_test=y_test,
        class_labels=CLASS_LABELS,
        results_dir=RESULTS_DIR,
    )

    # 6. SHAP on Random Forest — explain Withdrawn class (highest stakes)
    run_shap_explanations(
        tree_model=models["random_forest"],
        X_test=X_test,
        y_test=y_test.reset_index(drop=True),
        feature_names=list(X.columns),
        results_dir=RESULTS_DIR,
        n_students=5,
        class_labels=CLASS_LABELS,
        explain_class=3,  # Withdrawn
    )

    print(f"\nDone. All results saved to /{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
