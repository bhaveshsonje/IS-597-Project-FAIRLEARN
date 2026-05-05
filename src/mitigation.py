"""
mitigation.py — Fairness mitigation methods.

1. Reweighting  (pre-processing): upweight underrepresented group×outcome combinations
2. Thresholding (post-processing): find per-group thresholds that equalise TPR
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# 1. REWEIGHTING
# ---------------------------------------------------------------------------

def compute_reweights(y_train: pd.Series, group_train: pd.Series) -> np.ndarray:
    """
    Compute sample weights so every group×label cell has equal total weight.
    Weight for sample i = (N / (n_groups * n_labels)) / count(group_i, label_i)
    """
    df = pd.DataFrame({"y": y_train.values, "group": group_train.values})
    counts = df.groupby(["group", "y"])["y"].transform("count")
    n_groups = df["group"].nunique()
    n_labels = df["y"].nunique()
    weights = len(df) / (n_groups * n_labels * counts)
    return weights.values


def train_reweighted(X_train, y_train, group_train: pd.Series,
                     include_xgboost: bool = False):
    """Train LR + RF (and optionally XGBoost) with reweighting on the given sensitive attribute.

    MLP is excluded because sklearn MLPClassifier.fit() does not accept sample_weight.
    """
    weights = compute_reweights(y_train, group_train)

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=42))
    lr.fit(X_train, y_train, logisticregression__sample_weight=weights)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=weights)

    out = {"logistic_regression_rw": lr, "random_forest_rw": rf}

    if include_xgboost:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train, sample_weight=weights)
        out["xgboost_rw"] = xgb

    return out


# ---------------------------------------------------------------------------
# 2. THRESHOLDING
# ---------------------------------------------------------------------------

def find_equal_tpr_thresholds(model, X_test, y_test: pd.Series,
                               group_test: pd.Series) -> dict:
    """
    For each group, find the decision threshold that achieves the same TPR
    as the overall model at threshold=0.5 (equalised-odds style).
    Returns a dict mapping group value → threshold.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    # Overall TPR at 0.5
    y_pred_default = (y_prob >= 0.5).astype(int)
    tp = ((y_test == 1) & (y_pred_default == 1)).sum()
    fn = ((y_test == 1) & (y_pred_default == 0)).sum()
    target_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.5

    thresholds = {}
    for gval in group_test.unique():
        mask = (group_test == gval).values
        probs_g = y_prob[mask]
        labels_g = y_test.values[mask]

        best_thresh = 0.5
        best_diff = float("inf")
        for t in np.linspace(0.1, 0.9, 81):
            pred_g = (probs_g >= t).astype(int)
            tp_g = ((labels_g == 1) & (pred_g == 1)).sum()
            fn_g = ((labels_g == 1) & (pred_g == 0)).sum()
            tpr_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
            diff = abs(tpr_g - target_tpr)
            if diff < best_diff:
                best_diff = diff
                best_thresh = t

        thresholds[str(gval)] = round(float(best_thresh), 2)

    return {"target_tpr": round(target_tpr, 4), "thresholds": thresholds}


def apply_thresholds(model, X_test, group_test: pd.Series,
                     thresholds: dict) -> np.ndarray:
    """Apply per-group thresholds to produce final predictions."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = np.zeros(len(y_prob), dtype=int)
    for gval, thresh in thresholds.items():
        mask = (group_test == gval).values
        y_pred[mask] = (y_prob[mask] >= thresh).astype(int)
    return y_pred


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def eval_with_thresholds(model, X_test, y_test: pd.Series,
                         group_test: pd.Series) -> dict:
    thresh_info = find_equal_tpr_thresholds(model, X_test, y_test, group_test)
    y_pred = apply_thresholds(model, X_test, group_test, thresh_info["thresholds"])
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
        "target_tpr": thresh_info["target_tpr"],
        "group_thresholds": thresh_info["thresholds"],
    }


def group_tpr(y_true, y_pred, group_series: pd.Series) -> dict:
    result = {}
    for gval in group_series.unique():
        mask = (group_series == gval).values
        yt, yp = y_true.values[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        result[str(gval)] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else float("nan")
    return result


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_tpr_comparison(before: dict, after: dict, group_col: str,
                        model_name: str, results_dir: str):
    labels = sorted(set(before) | set(after))
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [before.get(k, 0) for k in labels], width, label="Before mitigation")
    ax.bar(x + width / 2, [after.get(k, 0) for k in labels], width, label="After thresholding")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} — TPR by {group_col} (thresholding)")
    ax.set_ylabel("TPR")
    ax.legend()
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/mitigation_threshold_{model_name}_{group_col}.png", dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------

def run_mitigation(base_models: dict, X_train, X_test, y_train, y_test,
                   df_train: pd.DataFrame, df_test: pd.DataFrame,
                   results_dir: str, include_xgboost_rw: bool = False):
    """
    Run both mitigation methods and save results.

    df_train / df_test  : original (non-encoded) dataframes with demographic cols.
    include_xgboost_rw  : if True, also produce a reweighted XGBoost model.
    """
    SENSITIVE = "gender"   # primary attribute for reweighting demo

    print("\n=== Mitigation: Reweighting ===")
    group_train = df_train[SENSITIVE].reset_index(drop=True)
    rw_models = train_reweighted(X_train, y_train, group_train,
                                 include_xgboost=include_xgboost_rw)

    rw_metrics = {}
    for name, model in rw_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rw_metrics[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "auc": round(roc_auc_score(y_test, y_prob), 4),
        }
        print(f"  {name}: Accuracy={rw_metrics[name]['accuracy']} | "
              f"F1={rw_metrics[name]['f1']} | AUC={rw_metrics[name]['auc']}")

    print("\n=== Mitigation: Per-group Thresholding ===")
    threshold_results = {}
    for name, model in base_models.items():
        for group_col in ["gender", "disability", "age_band"]:
            group_test = df_test[group_col].reset_index(drop=True)

            # TPR before
            y_pred_default = model.predict(X_test)
            before_tpr = group_tpr(y_test.reset_index(drop=True),
                                   y_pred_default, group_test)

            # TPR after thresholding
            thresh_info = find_equal_tpr_thresholds(
                model, X_test, y_test.reset_index(drop=True), group_test)
            y_pred_thresh = apply_thresholds(
                model, X_test, group_test, thresh_info["thresholds"])
            after_tpr = group_tpr(y_test.reset_index(drop=True),
                                  y_pred_thresh, group_test)

            tpr_gap_before = round(max(v for v in before_tpr.values()
                                       if not np.isnan(v))
                                   - min(v for v in before_tpr.values()
                                         if not np.isnan(v)), 4)
            tpr_gap_after  = round(max(v for v in after_tpr.values()
                                       if not np.isnan(v))
                                   - min(v for v in after_tpr.values()
                                         if not np.isnan(v)), 4)

            print(f"  {name} | {group_col}: TPR gap {tpr_gap_before} -> {tpr_gap_after}")
            plot_tpr_comparison(before_tpr, after_tpr, group_col, name, results_dir)

            threshold_results[f"{name}_{group_col}"] = {
                "thresholds": thresh_info["thresholds"],
                "tpr_before": before_tpr,
                "tpr_after": after_tpr,
                "tpr_gap_before": tpr_gap_before,
                "tpr_gap_after": tpr_gap_after,
            }

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/mitigation_reweighting.json", "w") as f:
        json.dump(rw_metrics, f, indent=2)
    with open(f"{results_dir}/mitigation_thresholding.json", "w") as f:
        json.dump(threshold_results, f, indent=2)

    return rw_models, threshold_results
