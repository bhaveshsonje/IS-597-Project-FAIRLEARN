"""
build_report_assets.py — Consolidate every figure + table the IEEE report needs
                         into one self-contained folder: results/report_assets/.

Layout produced:
    results/report_assets/
    ├── INDEX.md                       describes every folder + which report section uses it
    ├── overview/                      summary table + the tradeoff scatter
    ├── aggregated/
    │   ├── binary_advanced/           4 models, binary mode, with mitigation + SHAP
    │   ├── multiclass_unbalanced/     4 models, multiclass, no class weights
    │   └── multiclass_balanced/       4 models, multiclass, with class weights
    ├── per_module_binary/             BBB CCC DDD FFF (LR + RF + mitigation)
    └── per_module_multiclass/         BBB CCC DDD FFF (4 models + class weights)

Re-run any time inputs change. Safe to delete the folder entirely.

Usage:
    python src/build_report_assets.py
"""

import json
import os
import shutil

ROOT       = "results"
ASSETS_DIR = os.path.join(ROOT, "report_assets")


# ─── source → destination layout ───────────────────────────────────────────
LAYOUT = {
    "overview": {
        "files": [
            "summary.json", "summary.csv",
            "tradeoff_plot.png", "tradeoff_data.csv",
        ],
        "from_root": True,
    },
    "aggregated/binary_advanced":      {"copy_dir": "advanced_binary"},
    "aggregated/multiclass_unbalanced":{"copy_dir": "advanced_unbalanced"},
    "aggregated/multiclass_balanced":  {"copy_dir": "advanced_balanced"},
    "per_module_binary/BBB":           {"copy_dir": "BBB"},
    "per_module_binary/CCC":           {"copy_dir": "CCC"},
    "per_module_binary/DDD":           {"copy_dir": "DDD"},
    "per_module_binary/FFF":           {"copy_dir": "FFF"},
    "per_module_multiclass/BBB":       {"copy_dir": "per_course_mc/BBB"},
    "per_module_multiclass/CCC":       {"copy_dir": "per_course_mc/CCC"},
    "per_module_multiclass/DDD":       {"copy_dir": "per_course_mc/DDD"},
    "per_module_multiclass/FFF":       {"copy_dir": "per_course_mc/FFF"},
}


def fresh_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def copy_loose_files(dest: str, files: list):
    """Copy a list of files from results/ root into dest, skipping missing ones."""
    copied = []
    for fname in files:
        src = os.path.join(ROOT, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, fname))
            copied.append(fname)
    return copied


def copy_dir(src_rel: str, dest: str):
    """Copy an entire results/ subdirectory (recursively) into dest."""
    src = os.path.join(ROOT, src_rel)
    if not os.path.isdir(src):
        print(f"  Warning: source missing, skipping → {src}")
        return False
    shutil.copytree(src, dest)
    return True


def best_model(metrics_path: str) -> tuple:
    """Return (best_model_name, auc) from a model_metrics.json."""
    if not os.path.exists(metrics_path):
        return (None, None)
    with open(metrics_path) as f:
        m = json.load(f)
    best = max(m.items(), key=lambda kv: kv[1].get("auc", 0))
    return (best[0], best[1].get("auc"))


def write_index(asset_root: str, status: dict):
    """Generate INDEX.md — a navigation guide for each folder + report-section mapping."""
    lines = []
    lines.append("# Report Assets — IS 597 Final Report\n")
    lines.append("This folder is self-contained: every figure and table the IEEE report ")
    lines.append("references lives here. Regenerate with `python src/build_report_assets.py`.\n\n")

    lines.append("## Folder map\n")
    lines.append("| Folder | What's inside | Report section |\n")
    lines.append("|--------|---------------|----------------|\n")
    lines.append("| `overview/` | Master summary table (all 36 model runs) + accuracy-fairness tradeoff plot | §11 Discussion |\n")
    lines.append("| `aggregated/binary_advanced/` | 4 models in binary mode, full mitigation, SHAP on XGBoost | §6 Models, §7 Fairness, §8 Mitigation, §9 Explainability |\n")
    lines.append("| `aggregated/multiclass_unbalanced/` | 4 models, multiclass, no class weights — accuracy-optimized baseline | §6 Models |\n")
    lines.append("| `aggregated/multiclass_balanced/` | 4 models, multiclass, with class weights — fairness-optimized variant | §6 Models, §7 Fairness |\n")
    lines.append("| `per_module_binary/<MODULE>/` | LR + RF + mitigation, per OULAD module | §10 Per-Course Analysis |\n")
    lines.append("| `per_module_multiclass/<MODULE>/` | 4 models + class weights, per OULAD module | §10 Per-Course Analysis |\n")
    lines.append("\n")

    lines.append("## Headline numbers (best AUC per pipeline)\n")
    lines.append("| Pipeline | Best model | AUC | Accuracy |\n")
    lines.append("|----------|-----------|-----|----------|\n")
    for label, info in status["winners"].items():
        if info["model"] is None:
            continue
        lines.append(f"| {label} | {info['model']} | {info['auc']} | {info['accuracy']} |\n")
    lines.append("\n")

    lines.append("## Headline figures\n")
    lines.append("- `overview/tradeoff_plot.png` — the report's main visualization. ")
    lines.append("Shows every model's accuracy vs max TPR gap, with arrows indicating mitigation effect.\n")
    lines.append("- `aggregated/binary_advanced/model_comparison.png` — bar chart of all 4 binary models on Accuracy/F1/AUC.\n")
    lines.append("- `aggregated/binary_advanced/confusion_matrix_xgboost.png` — confusion matrix for the best binary model.\n")
    lines.append("- `aggregated/multiclass_balanced/confusion_matrix_*.png` — multiclass confusion matrices showing Pass/Distinction confusion.\n")
    lines.append("- `aggregated/binary_advanced/shap_summary.png` — global feature importance for XGBoost.\n")
    lines.append("- `aggregated/binary_advanced/mitigation_threshold_*.png` — TPR before vs after thresholding.\n")
    lines.append("\n")

    lines.append("## File-naming conventions\n")
    lines.append("- `model_metrics.json` — accuracy/F1/AUC per model\n")
    lines.append("- `fairness_metrics.json` — per-group TPR (binary) or per-class recall (multiclass)\n")
    lines.append("- `mitigation_reweighting.json` — accuracy of reweighted models\n")
    lines.append("- `mitigation_thresholding.json` — TPR gap before/after per-group thresholding\n")
    lines.append("- `confusion_matrix_<model>.png` — row-normalized heatmap (multiclass only)\n")
    lines.append("- `fairness_<model>_<group>.png` — binary fairness bar chart\n")
    lines.append("- `fairness_mc_<model>_<group>.png` — multiclass fairness bar chart (Fail/Withdrawn recall)\n")
    lines.append("- `shap_summary.png`, `shap_importance.png` — global SHAP plots\n")
    lines.append("- `shap_student_<n>.png` — per-student SHAP waterfall (top 5 flagged)\n")

    with open(os.path.join(asset_root, "INDEX.md"), "w") as f:
        f.writelines(lines)


def main():
    print(f"Building report assets in {ASSETS_DIR}/ ...")
    fresh_dir(ASSETS_DIR)

    status = {"copied": [], "missing": [], "winners": {}}

    for dest_rel, spec in LAYOUT.items():
        dest = os.path.join(ASSETS_DIR, dest_rel)

        if spec.get("from_root"):
            os.makedirs(dest, exist_ok=True)
            copied = copy_loose_files(dest, spec["files"])
            print(f"  {dest_rel}/  ← {len(copied)}/{len(spec['files'])} files")
            for f in spec["files"]:
                if f not in copied:
                    status["missing"].append(f"{dest_rel}/{f}")
        else:
            ok = copy_dir(spec["copy_dir"], dest)
            if ok:
                status["copied"].append(dest_rel)
                print(f"  {dest_rel}/  ← results/{spec['copy_dir']}/")
            else:
                status["missing"].append(dest_rel)

        # Track the winning model from each copied pipeline
        metrics_path = os.path.join(dest, "model_metrics.json")
        if os.path.exists(metrics_path):
            model, auc = best_model(metrics_path)
            with open(metrics_path) as f:
                m = json.load(f)
            status["winners"][dest_rel] = {
                "model": model,
                "auc": auc,
                "accuracy": m.get(model, {}).get("accuracy") if model else None,
            }

    write_index(ASSETS_DIR, status)
    print(f"  INDEX.md written")

    print(f"\nDone. Asset folder: {ASSETS_DIR}/")
    if status["missing"]:
        print(f"\nMissing inputs ({len(status['missing'])}):")
        for m in status["missing"]:
            print(f"  - {m}")
    print(f"\nTip: zip up {ASSETS_DIR}/ to share the full deliverable.")


if __name__ == "__main__":
    main()
