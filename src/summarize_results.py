"""
summarize_results.py — Aggregate metrics from all pipelines into one comparison table.

Reads every model_metrics.json under results/ and prints a clean table
plus saves results/summary.json and results/summary.csv.

Usage:
    python src/summarize_results.py
"""

import csv
import json
import os
from glob import glob


RESULTS_DIR = "results"


def collect_metrics(root: str) -> list:
    """Walk results/ and return one row per (pipeline, model)."""
    rows = []
    for path in sorted(glob(os.path.join(root, "**", "model_metrics.json"), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), root)
        if rel == ".":
            continue  # results/ root no longer holds metrics — every pipeline writes to its own subdir
        if rel.startswith("report_assets" + os.sep) or rel == "report_assets":
            continue  # report_assets/ mirrors other pipelines — would double-count
        pipeline = rel.replace(os.sep, "/")
        with open(path) as f:
            metrics = json.load(f)
        for model_name, m in metrics.items():
            rows.append({
                "pipeline": pipeline,
                "model":    model_name,
                "accuracy": m.get("accuracy"),
                "f1":       m.get("f1"),
                "auc":      m.get("auc"),
            })
    return rows


def print_table(rows: list):
    headers = ["pipeline", "model", "accuracy", "f1", "auc"]
    widths  = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep  = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    last_pipeline = None
    for r in rows:
        if last_pipeline and r["pipeline"] != last_pipeline:
            print(sep)
        print(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))
        last_pipeline = r["pipeline"]


def best_per_pipeline(rows: list) -> dict:
    """For each pipeline, return the model with the highest AUC."""
    by_pipe = {}
    for r in rows:
        p = r["pipeline"]
        if p not in by_pipe or r["auc"] > by_pipe[p]["auc"]:
            by_pipe[p] = r
    return by_pipe


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"No results directory at {RESULTS_DIR}")
        return

    rows = collect_metrics(RESULTS_DIR)
    if not rows:
        print(f"No model_metrics.json files found under {RESULTS_DIR}")
        return

    print(f"=== Results Summary ({len(rows)} model runs across "
          f"{len(set(r['pipeline'] for r in rows))} pipelines) ===\n")
    print_table(rows)

    best = best_per_pipeline(rows)
    print("\n=== Best model per pipeline (by AUC) ===")
    for p in sorted(best):
        r = best[p]
        print(f"  {p:30s} → {r['model']:22s} AUC={r['auc']}, "
              f"Accuracy={r['accuracy']}, F1={r['f1']}")

    out_json = os.path.join(RESULTS_DIR, "summary.json")
    out_csv  = os.path.join(RESULTS_DIR, "summary.csv")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pipeline", "model", "accuracy", "f1", "auc"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
