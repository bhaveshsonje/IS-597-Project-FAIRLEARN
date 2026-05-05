"""
explain.py — SHAP-based explanations for flagged (at-risk) students.

Uses TreeExplainer for tree-based models (Random Forest, XGBoost) — fast and exact.
Produces:
  - summary plot: global feature importance
  - waterfall plots: per-student explanation for top N flagged students
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# SHAP internally strips DataFrame feature names when computing predictions,
# which triggers a noisy sklearn warning. Suppress it at the SHAP boundary.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


def run_shap_explanations(tree_model, X_test, y_test: pd.Series,
                          feature_names: list, results_dir: str,
                          n_students: int = 5,
                          class_labels: dict = None,
                          explain_class: int = 1):
    """
    tree_model    : trained tree-based model (RandomForest or XGBoost)
    X_test        : numpy array or DataFrame (encoded)
    y_test        : true labels (aligned with X_test)
    n_students    : number of flagged students to explain individually
    class_labels  : dict mapping int → name (e.g. {0:'Pass', 1:'Distinction', ...}).
                    If None, assumes binary with labels {0:'At-Risk', 1:'Success'}.
    explain_class : which class index to use for SHAP values and probability display.
                    Binary default = 1 (success). Multiclass recommended = Withdrawn index.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Build label lookup
    if class_labels is None:
        class_labels = {0: "At-Risk", 1: "Success"}

    is_binary = len(class_labels) == 2

    # At-risk classes: in binary = 0; in multiclass = Fail and Withdrawn
    if is_binary:
        at_risk_classes = {0}
    else:
        at_risk_classes = {k for k, v in class_labels.items()
                           if v in ("Fail", "Withdrawn")}

    X_arr = X_test.values if hasattr(X_test, "values") else X_test

    # Subsample for speed — SHAP on full test set is slow
    rng = np.random.default_rng(42)
    sample_size = min(500, len(X_arr))
    sample_idx = rng.choice(len(X_arr), size=sample_size, replace=False)
    X_sample = X_arr[sample_idx]

    explain_class_name = class_labels.get(explain_class, str(explain_class))
    model_name = type(tree_model).__name__
    print(f"\n=== SHAP Explanations ({model_name}) ===")
    print(f"Computing SHAP values on {sample_size} samples (explaining class: {explain_class_name})...")

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_sample)

    # Newer SHAP returns (n_samples, n_features, n_classes); older returns a list
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, explain_class]
    elif isinstance(shap_values, list):
        sv = shap_values[explain_class]
    else:
        sv = shap_values

    # --- Summary plot (global feature importance) ---
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — Random Forest (class: {explain_class_name})")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary.png")

    # --- Bar plot (mean |SHAP|) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs = np.abs(sv).mean(axis=0).flatten()
    order = np.argsort(mean_abs).tolist()
    sorted_names = [feature_names[i] for i in order]
    ax.barh(sorted_names, [float(mean_abs[i]) for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance (SHAP — class: {explain_class_name})")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/shap_importance.png", dpi=100)
    plt.close()
    print("  Saved: shap_importance.png")

    # --- Per-student waterfall plots for flagged (at-risk) students ---
    y_prob_sample = tree_model.predict_proba(X_sample)[:, explain_class]
    y_pred_sample = tree_model.predict(X_sample)
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test

    # Flagged = predicted as any at-risk class
    flagged_idx = np.where(np.isin(y_pred_sample, list(at_risk_classes)))[0]
    flagged_probs = y_prob_sample[flagged_idx]
    sorted_by_prob = flagged_idx[np.argsort(flagged_probs)[::-1]]  # highest risk first
    sample_positions = np.linspace(0, len(sorted_by_prob) - 1, n_students, dtype=int)
    selected = sorted_by_prob[sample_positions]

    for rank, idx in enumerate(selected):
        student_shap = sv[idx]
        true_cls = int(y_test_arr[sample_idx[idx]])
        pred_cls = int(y_pred_sample[idx])
        true_name = class_labels.get(true_cls, str(true_cls))
        pred_name = class_labels.get(pred_cls, str(pred_cls))

        fig, ax = plt.subplots(figsize=(9, 5))
        shap_flat = student_shap.flatten()
        order = np.argsort(np.abs(shap_flat))[::-1][:10].tolist()
        top_names = [feature_names[i] for i in order][::-1]
        top_vals  = [float(shap_flat[i]) for i in order][::-1]
        colors = ["#d73027" if s < 0 else "#4575b4" for s in top_vals]
        ax.barh(top_names, top_vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"SHAP value (positive = pushes toward {explain_class_name})")
        ax.set_title(
            f"Student #{rank + 1} — Predicted: {pred_name} | "
            f"True: {true_name} | "
            f"P({explain_class_name})={y_prob_sample[idx]:.2f}"
        )
        plt.tight_layout()
        fname = f"{results_dir}/shap_student_{rank + 1}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"  Saved: shap_student_{rank + 1}.png  "
              f"[P({explain_class_name})={y_prob_sample[idx]:.2f}, "
              f"pred={pred_name}, true={true_name}]")

    print("\nTop global features by mean |SHAP|:")
    for i in np.argsort(mean_abs)[::-1][:5].tolist():
        print(f"  {feature_names[i]}: {float(mean_abs[i]):.4f}")
