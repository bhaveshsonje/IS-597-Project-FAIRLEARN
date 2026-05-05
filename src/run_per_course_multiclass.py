"""
run_per_course_multiclass.py — Per-module multiclass models (Phase 2 extension).

For each of the 4 largest OULAD modules (BBB, FFF, DDD, CCC):
  - Train all 4 models (LR, RF, XGBoost, MLP) with class balancing
  - Run multiclass fairness analysis
  - SHAP on XGBoost (Withdrawn class)

Per-module models are expected to outperform the aggregated multiclass
model because course design and engagement patterns differ substantially.

Usage:
    python src/run_per_course_multiclass.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import build_dataset, CLASS_LABELS
from features import get_feature_matrix, FAIRNESS_FEATURES
from models import run_training_all
from fairness import run_fairness_analysis_multiclass
from explain import run_shap_explanations

DATA_DIR     = "Dataset"
RESULTS_BASE = "results/per_course_mc"

# 4 largest modules by student count
TARGET_MODULES = ["BBB", "FFF", "DDD", "CCC"]


def run_course(df_course, module, results_dir):
    print(f"\n{'='*60}")
    print(f"Module: {module}  |  Students: {len(df_course)}")
    counts = df_course["target"].value_counts().sort_index()
    for cls_int, n in counts.items():
        print(f"  {CLASS_LABELS[cls_int]:12s}: {n:5d} ({n/len(df_course)*100:5.2f}%)")

    if len(df_course) < 100:
        print(f"Skipping {module} — too few samples ({len(df_course)})")
        return

    X, y = get_feature_matrix(df_course)
    models, _, splits = run_training_all(X, y, results_dir, balanced=True,
                                          class_labels=CLASS_LABELS)
    _, X_test, _, y_test = splits

    X_test_raw = df_course.loc[y_test.index, FAIRNESS_FEATURES]

    print()
    run_fairness_analysis_multiclass(
        models=models,
        X_test_raw=X_test_raw,
        X_test_enc=X_test,
        y_test=y_test,
        class_labels=CLASS_LABELS,
        results_dir=results_dir,
    )

    print(f"\n  SHAP on XGBoost (Withdrawn class)...")
    run_shap_explanations(
        tree_model=models["xgboost"],
        X_test=X_test,
        y_test=y_test.reset_index(drop=True),
        feature_names=list(X.columns),
        results_dir=results_dir,
        n_students=5,
        class_labels=CLASS_LABELS,
        explain_class=3,  # Withdrawn
    )


def main():
    print("=== IS 597 — Per-Course Multiclass Analysis (Phase 2) ===\n")

    print("Loading full dataset (multiclass mode)...")
    df = build_dataset(DATA_DIR, mode="multiclass")
    print(f"Full dataset: {df.shape[0]} students across all modules\n")

    for module in TARGET_MODULES:
        df_course = df[df["code_module"] == module].copy().reset_index(drop=True)
        results_dir = os.path.join(RESULTS_BASE, module)
        run_course(df_course, module, results_dir)

    print(f"\n{'='*60}")
    print("Done. Per-module multiclass results saved to:")
    for module in TARGET_MODULES:
        print(f"  results/per_course_mc/{module}/")


if __name__ == "__main__":
    main()
