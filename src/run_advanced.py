"""
run_advanced.py — All four models (LR, RF, XGBoost, MLP) in multiclass mode.

Runs both UNBALANCED and BALANCED variants in one pass so the accuracy/fairness
trade-off from class balancing is directly comparable.

  results/advanced_unbalanced/ → no class weighting
  results/advanced_balanced/   → class_weight='balanced' for LR/RF, sample weights for XGB
                                 (MLP unaffected — sklearn limitation)

SHAP explanations use XGBoost (TreeExplainer, fast and exact) for both variants.

Usage:
    python src/run_advanced.py
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
RESULTS_BASE = "results"

VARIANTS = [
    ("advanced_unbalanced", False),
    ("advanced_balanced",   True),
]


def run_variant(X, y, df, variant_name: str, balanced: bool):
    results_dir = os.path.join(RESULTS_BASE, variant_name)
    print(f"\n{'#' * 70}")
    print(f"# Variant: {variant_name}  (balanced={balanced})")
    print(f"{'#' * 70}")

    models, _, splits = run_training_all(
        X, y, results_dir, balanced=balanced, class_labels=CLASS_LABELS,
    )
    _, X_test, _, y_test = splits

    X_test_raw = df.loc[y_test.index, FAIRNESS_FEATURES]

    print()
    run_fairness_analysis_multiclass(
        models=models,
        X_test_raw=X_test_raw,
        X_test_enc=X_test,
        y_test=y_test,
        class_labels=CLASS_LABELS,
        results_dir=results_dir,
    )

    print("\nRunning SHAP on XGBoost...")
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

    print(f"\nVariant complete. Results: /{results_dir}/")


def main():
    print("=== IS 597 — Phase 3: Multiclass (Balanced vs Unbalanced) ===\n")

    print("Loading dataset (multiclass mode)...")
    df = build_dataset(DATA_DIR, mode="multiclass")
    print(f"Dataset shape: {df.shape}")
    counts = df["target"].value_counts().sort_index()
    for cls_int, n in counts.items():
        print(f"  {CLASS_LABELS[cls_int]} ({cls_int}): {n} ({n/len(df)*100:.1f}%)")
    print()

    print("Building feature matrix...")
    X, y = get_feature_matrix(df)
    print(f"Features: {list(X.columns)}\n")

    for variant_name, balanced in VARIANTS:
        run_variant(X, y, df, variant_name, balanced)

    print(f"\n{'=' * 70}")
    print("Done. Variants saved to:")
    for variant_name, _ in VARIANTS:
        print(f"  {RESULTS_BASE}/{variant_name}/")


if __name__ == "__main__":
    main()
