"""
explain.py — SHAP-based explanations for flagged (at-risk) students.

Uses TreeExplainer for Random Forest (fast, exact).
Produces:
  - summary plot: global feature importance
  - waterfall plots: per-student explanation for top N flagged students
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def run_shap_explanations(rf_model, X_test, y_test: pd.Series,
                          feature_names: list, results_dir: str,
                          n_students: int = 5):
    """
    rf_model    : trained RandomForestClassifier
    X_test      : numpy array or DataFrame (encoded)
    y_test      : true labels (aligned with X_test)
    n_students  : number of flagged students to explain individually
    """
    os.makedirs(results_dir, exist_ok=True)

    X_arr = X_test.values if hasattr(X_test, "values") else X_test

    # Subsample for speed — SHAP on full test set is slow
    rng = np.random.default_rng(42)
    sample_size = min(500, len(X_arr))
    sample_idx = rng.choice(len(X_arr), size=sample_size, replace=False)
    X_sample = X_arr[sample_idx]

    print("\n=== SHAP Explanations (Random Forest) ===")
    print(f"Computing SHAP values on {sample_size} samples...")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    # Newer SHAP returns (n_samples, n_features, n_classes); older returns a list
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]   # class 1 (success)
    elif isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # --- Summary plot (global feature importance) ---
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary — Random Forest (class: success)")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_summary.png")

    # --- Bar plot (mean |SHAP|) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs = np.abs(sv).mean(axis=0).flatten()
    order = np.argsort(mean_abs).tolist()  # ascending so most important is at top
    sorted_names = [feature_names[i] for i in order]
    ax.barh(sorted_names, [float(mean_abs[i]) for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/shap_importance.png", dpi=100)
    plt.close()
    print("  Saved: shap_importance.png")

    # --- Per-student waterfall plots for flagged students ---
    # "Flagged" = model predicts at-risk (class 0), i.e. low P(success)
    y_prob_sample = rf_model.predict_proba(X_sample)[:, 1]
    y_pred_sample = (y_prob_sample >= 0.5).astype(int)
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test

    flagged_idx = np.where(y_pred_sample == 0)[0]  # predicted at-risk within sample
    flagged_probs = y_prob_sample[flagged_idx]
    sorted_by_prob = flagged_idx[np.argsort(flagged_probs)]
    sample_positions = np.linspace(0, len(sorted_by_prob) - 1, n_students, dtype=int)
    selected = sorted_by_prob[sample_positions]

    for rank, idx in enumerate(selected):
        student_shap = sv[idx]
        true_label = y_test_arr[sample_idx[idx]]

        fig, ax = plt.subplots(figsize=(9, 5))
        # Manual waterfall: sort features by |shap|
        shap_flat = student_shap.flatten()
        order = np.argsort(np.abs(shap_flat))[::-1][:10].tolist()
        top_names = [feature_names[i] for i in order][::-1]
        top_vals  = [float(shap_flat[i]) for i in order][::-1]
        colors = ["#d73027" if s < 0 else "#4575b4" for s in top_vals]
        ax.barh(top_names, top_vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (positive = pushes toward success)")
        ax.set_title(
            f"Student #{rank + 1} — Predicted: {'Success' if y_pred_sample[idx] else 'At-Risk'} | "
            f"True: {'Success' if true_label else 'At-Risk'} | "
            f"P(success)={y_prob_sample[idx]:.2f}"
        )
        plt.tight_layout()
        fname = f"{results_dir}/shap_student_{rank + 1}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"  Saved: shap_student_{rank + 1}.png  "
              f"[P(success)={y_prob_sample[idx]:.2f}, true={'pass' if true_label else 'fail'}]")

    print("\nTop global features by mean |SHAP|:")
    for i in np.argsort(mean_abs)[::-1][:5].tolist():
        print(f"  {feature_names[i]}: {float(mean_abs[i]):.4f}")
