"""
tradeoff_plot.py — Accuracy vs Fairness scatter plot.

For every binary pipeline+model:
  - Plots accuracy vs max TPR gap (over gender/disability/age_band)
  - Connects "before mitigation" → "after thresholding" with an arrow

The headline figure for the report's Discussion section.

Usage:
    python src/tradeoff_plot.py
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "results"
OUT_PNG     = os.path.join(RESULTS_DIR, "tradeoff_plot.png")
OUT_CSV     = os.path.join(RESULTS_DIR, "tradeoff_data.csv")

# Pipelines we treat as binary (have well-defined binary TPR gap)
BINARY_PIPELINES = [
    "binary_aggregated", "advanced_binary",
    "BBB", "CCC", "DDD", "FFF",
]

MODEL_COLORS = {
    "logistic_regression": "#1f77b4",  # blue
    "random_forest":       "#2ca02c",  # green
    "xgboost":             "#d62728",  # red
    "mlp":                 "#9467bd",  # purple
}

PIPELINE_MARKERS = {
    "binary_aggregated":   "o",   # circle
    "advanced_binary":     "s",   # square
    "BBB": "^", "CCC": "v", "DDD": "D", "FFF": "P",  # per-module shapes
}


def pipeline_dir(pipeline: str) -> str:
    """Map pipeline name to its results directory."""
    return os.path.join(RESULTS_DIR, pipeline)


def gaps_from_thresholding(thresh_dict: dict, model_name: str) -> tuple:
    """Return (max gap_before, max gap_after) for this model across all group_cols.

    Both pulled from the SAME file so the group sets are consistent (thresholding
    doesn't filter small groups, fairness analysis does — using thresholding for
    both keeps the comparison apples-to-apples).
    """
    befores, afters = [], []
    for key, info in thresh_dict.items():
        if key.startswith(model_name + "_"):
            b = info.get("tpr_gap_before")
            a = info.get("tpr_gap_after")
            if b is not None and not np.isnan(b): befores.append(b)
            if a is not None and not np.isnan(a): afters.append(a)
    return (
        max(befores) if befores else float("nan"),
        max(afters)  if afters  else float("nan"),
    )


def collect_rows() -> list:
    rows = []
    for pipeline in BINARY_PIPELINES:
        pdir = pipeline_dir(pipeline)
        metrics_path = os.path.join(pdir, "model_metrics.json")
        thresh_path  = os.path.join(pdir, "mitigation_thresholding.json")

        if not os.path.exists(metrics_path):
            print(f"  Skipping {pipeline} — no model_metrics.json")
            continue

        with open(metrics_path) as f: metrics = json.load(f)
        thresh = {}
        if os.path.exists(thresh_path):
            with open(thresh_path) as f: thresh = json.load(f)

        for model_name, m in metrics.items():
            accuracy = m["accuracy"]
            gap_before, gap_after = gaps_from_thresholding(thresh, model_name)
            rows.append({
                "pipeline":   pipeline,
                "model":      model_name,
                "accuracy":   accuracy,
                "gap_before": round(gap_before, 4) if not np.isnan(gap_before) else None,
                "gap_after":  round(gap_after,  4) if not np.isnan(gap_after)  else None,
            })
    return rows


def plot(rows: list, out_path: str):
    _, ax = plt.subplots(figsize=(11, 7))

    # Draw arrows + endpoints
    for r in rows:
        if r["gap_before"] is None:
            continue
        color  = MODEL_COLORS.get(r["model"], "gray")
        marker = PIPELINE_MARKERS.get(r["pipeline"], "X")
        x_before = r["gap_before"]
        y        = r["accuracy"]
        x_after  = r["gap_after"]

        # "Before" point — hollow marker
        ax.scatter(x_before, y, color=color, marker=marker, s=120,
                   facecolors="none", edgecolors=color, linewidths=1.5, zorder=3)

        # If we have post-thresholding data, draw arrow + solid endpoint
        if x_after is not None:
            ax.annotate(
                "", xy=(x_after, y), xytext=(x_before, y),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.45, lw=1.2),
                zorder=2,
            )
            ax.scatter(x_after, y, color=color, marker=marker, s=120,
                       edgecolors="black", linewidths=0.5, zorder=4)

    # Axes
    ax.set_xlabel("Max TPR gap across demographic groups (lower = more fair)")
    ax.set_ylabel("Accuracy (higher = better)")
    ax.set_title("Accuracy–Fairness Trade-off  (hollow = before mitigation, "
                 "solid = after thresholding, arrows show mitigation effect)")
    ax.set_xlim(left=-0.005)
    ax.grid(alpha=0.3)

    # Build a clean legend in two parts
    model_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name,
                   markerfacecolor=col, markersize=10)
        for name, col in MODEL_COLORS.items()
    ]
    pipeline_handles = [
        plt.Line2D([0], [0], marker=mk, color="gray", label=name,
                   linestyle="None", markersize=10)
        for name, mk in PIPELINE_MARKERS.items()
    ]
    leg1 = ax.legend(handles=model_handles, title="Model",
                     loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=pipeline_handles, title="Pipeline",
              loc="lower right", bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_csv(rows: list, out_path: str):
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pipeline", "model", "accuracy",
                                          "gap_before", "gap_after"])
        w.writeheader()
        w.writerows(rows)


def main():
    print(f"Collecting binary pipeline results from {RESULTS_DIR}/...")
    rows = collect_rows()
    if not rows:
        print("No rows collected — nothing to plot.")
        return

    print(f"Collected {len(rows)} (pipeline, model) points\n")
    print(f"{'pipeline':22s} {'model':22s} {'accuracy':>9s} "
          f"{'gap_before':>11s} {'gap_after':>10s}")
    print("-" * 80)
    for r in rows:
        ga = "—" if r["gap_after"] is None else f"{r['gap_after']:.4f}"
        print(f"{r['pipeline']:22s} {r['model']:22s} "
              f"{r['accuracy']:9.4f} {r['gap_before']:11.4f} {ga:>10s}")

    plot(rows, OUT_PNG)
    save_csv(rows, OUT_CSV)
    print(f"\nSaved: {OUT_PNG}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
