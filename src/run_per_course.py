"""
run_per_course.py — Run baseline models + fairness analysis per course module.

Runs on the 4 largest OULAD modules: BBB, FFF, DDD, CCC.
Results saved to results/<MODULE>/ for each course.

Usage:
    python src/run_per_course.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import build_dataset
from features import get_feature_matrix, FAIRNESS_FEATURES
from models import run_training, split_data
from fairness import run_fairness_analysis

DATA_DIR = "Dataset"
RESULTS_BASE = "results"

# Modules chosen by size: BBB (~7900), FFF (~7700), DDD (~6300), CCC (~4400)
TARGET_MODULES = ["BBB", "FFF", "DDD", "CCC"]


def run_course(df_course, module, results_dir):
    print(f"\n{'='*50}")
    print(f"Module: {module}  |  Students: {len(df_course)}")
    print(f"Target distribution:\n{df_course['target'].value_counts().to_dict()}")

    X, y = get_feature_matrix(df_course)

    # Need enough samples for a meaningful split
    if len(df_course) < 100:
        print(f"Skipping {module} — too few samples ({len(df_course)})")
        return

    models, metrics, splits = run_training(X, y, results_dir)
    X_train, X_test, y_train, y_test = splits

    test_idx = y_test.index
    train_idx = y_train.index
    X_test_raw = df_course.loc[test_idx, FAIRNESS_FEATURES]

    run_fairness_analysis(models, X_test_raw, X_test, y_test, results_dir)

    print(f"Results saved to {results_dir}/")


def main():
    print("=== IS 597 — Per-Course Baseline Analysis ===\n")

    print("Loading full dataset...")
    df = build_dataset(DATA_DIR)
    print(f"Full dataset: {df.shape[0]} students across all modules\n")

    for module in TARGET_MODULES:
        df_course = df[df["code_module"] == module].copy().reset_index(drop=True)
        results_dir = os.path.join(RESULTS_BASE, module)
        run_course(df_course, module, results_dir)

    print(f"\n{'='*50}")
    print("Done. Per-course results saved to:")
    for module in TARGET_MODULES:
        print(f"  results/{module}/")


if __name__ == "__main__":
    main()
