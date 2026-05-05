"""
fairness.py — Basic fairness analysis across demographic groups.
Measures: accuracy per group, true positive rate (TPR) per group.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAIRNESS_GROUPS = ["gender", "disability", "age_band"]


def compute_group_metrics(model, X_test_raw: pd.DataFrame, X_test_enc, y_test: pd.Series) -> dict:
    """
    Compute accuracy and TPR per demographic group.

    Parameters
    ----------
    X_test_raw : original (non-encoded) test dataframe with demographic columns
    X_test_enc : encoded feature matrix used for prediction
    y_test     : true labels
    """
    y_pred = model.predict(X_test_enc)
    results = {}

    for group_col in FAIRNESS_GROUPS:
        if group_col not in X_test_raw.columns:
            continue

        group_results = {}
        for group_val in X_test_raw[group_col].unique():
            mask = X_test_raw[group_col] == group_val
            if mask.sum() < 10:
                continue

            y_true_g = y_test[mask]
            y_pred_g = y_pred[mask]

            accuracy = (y_true_g == y_pred_g).mean()

            # TPR = TP / (TP + FN)
            tp = ((y_true_g == 1) & (y_pred_g == 1)).sum()
            fn = ((y_true_g == 1) & (y_pred_g == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

            group_results[str(group_val)] = {
                "n": int(mask.sum()),
                "accuracy": round(float(accuracy), 4),
                "tpr": round(float(tpr), 4),
            }

        results[group_col] = group_results

    return results


def plot_fairness(fairness_results: dict, model_name: str, results_dir: str):
    """Save bar charts for accuracy and TPR per demographic group."""
    os.makedirs(results_dir, exist_ok=True)

    for group_col, group_data in fairness_results.items():
        labels = list(group_data.keys())
        accuracies = [group_data[k]["accuracy"] for k in labels]
        tprs = [group_data[k]["tpr"] for k in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, accuracies, width, label="Accuracy")
        ax.bar(x + width / 2, tprs, width, label="TPR")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_name} — Fairness by {group_col}")
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fairness_{model_name}_{group_col}.png", dpi=100)
        plt.close()


def compute_group_metrics_multiclass(model, X_test_raw: pd.DataFrame,
                                     X_test_enc, y_test: pd.Series,
                                     class_labels: dict) -> dict:
    """
    Compute accuracy and per-class recall per demographic group for multiclass targets.

    class_labels : dict mapping int → class name, e.g. {0:'Pass', 1:'Distinction', ...}
    """
    y_pred = model.predict(X_test_enc)
    results = {}

    for group_col in FAIRNESS_GROUPS:
        if group_col not in X_test_raw.columns:
            continue

        group_results = {}
        for group_val in X_test_raw[group_col].unique():
            mask = X_test_raw[group_col] == group_val
            if mask.sum() < 10:
                continue

            y_true_g = y_test[mask]
            y_pred_g = y_pred[mask]
            accuracy = float((y_true_g == y_pred_g).mean())

            per_class_recall = {}
            for cls_int, cls_name in class_labels.items():
                tp = ((y_true_g == cls_int) & (y_pred_g == cls_int)).sum()
                fn = ((y_true_g == cls_int) & (y_pred_g != cls_int)).sum()
                recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                per_class_recall[cls_name] = round(float(recall), 4)

            group_results[str(group_val)] = {
                "n": int(mask.sum()),
                "accuracy": round(accuracy, 4),
                "per_class_recall": per_class_recall,
            }

        results[group_col] = group_results

    return results


def plot_fairness_multiclass(fairness_results: dict, model_name: str,
                             results_dir: str, focus_classes: list = None):
    """
    Bar charts showing accuracy and recall for key classes per demographic group.

    focus_classes : class names to plot recall for (e.g. ['Fail', 'Withdrawn']).
                    Defaults to all classes if None.
    """
    os.makedirs(results_dir, exist_ok=True)

    for group_col, group_data in fairness_results.items():
        labels = list(group_data.keys())
        accuracies = [group_data[k]["accuracy"] for k in labels]

        all_classes = list(next(iter(group_data.values()))["per_class_recall"].keys())
        plot_classes = focus_classes if focus_classes else all_classes

        x = np.arange(len(labels))
        n_bars = 1 + len(plot_classes)
        width = 0.8 / n_bars

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - (n_bars / 2 - 0.5) * width, accuracies, width, label="Accuracy")

        for i, cls_name in enumerate(plot_classes):
            recalls = [group_data[k]["per_class_recall"].get(cls_name, float("nan"))
                       for k in labels]
            ax.bar(x - (n_bars / 2 - 1.5 - i) * width, recalls, width,
                   label=f"Recall ({cls_name})")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_name} — Multiclass Fairness by {group_col}")
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fairness_mc_{model_name}_{group_col}.png", dpi=100)
        plt.close()


def run_fairness_analysis_multiclass(models: dict, X_test_raw: pd.DataFrame,
                                     X_test_enc, y_test: pd.Series,
                                     class_labels: dict, results_dir: str):
    all_fairness = {}

    for model_name, model in models.items():
        fairness = compute_group_metrics_multiclass(
            model, X_test_raw, X_test_enc, y_test, class_labels)
        all_fairness[model_name] = fairness
        plot_fairness_multiclass(fairness, model_name, results_dir,
                                 focus_classes=["Fail", "Withdrawn"])

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/fairness_metrics.json", "w") as f:
        json.dump(all_fairness, f, indent=2)

    print("\n=== Multiclass Fairness Analysis ===")
    for model_name, fairness in all_fairness.items():
        print(f"\n-- {model_name} --")
        for group_col, groups in fairness.items():
            print(f"  {group_col}:")
            for gval, m in groups.items():
                recalls = ", ".join(
                    f"{c}={v}" for c, v in m["per_class_recall"].items())
                print(f"    {gval}: n={m['n']}, accuracy={m['accuracy']}, recalls=[{recalls}]")

    return all_fairness


def run_fairness_analysis(models: dict, X_test_raw: pd.DataFrame,
                          X_test_enc, y_test: pd.Series, results_dir: str):
    all_fairness = {}

    for model_name, model in models.items():
        fairness = compute_group_metrics(model, X_test_raw, X_test_enc, y_test)
        all_fairness[model_name] = fairness
        plot_fairness(fairness, model_name, results_dir)

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/fairness_metrics.json", "w") as f:
        json.dump(all_fairness, f, indent=2)

    print("\n=== Fairness Analysis ===")
    for model_name, fairness in all_fairness.items():
        print(f"\n-- {model_name} --")
        for group_col, groups in fairness.items():
            print(f"  {group_col}:")
            for gval, m in groups.items():
                print(f"    {gval}: n={m['n']}, accuracy={m['accuracy']}, tpr={m['tpr']}")

    return all_fairness
