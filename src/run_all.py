"""
run_all.py — Master script: regenerate every result + figure + asset folder.

Runs the full pipeline end-to-end (~5–10 minutes depending on hardware):
  Experiments:
    1. run_experiment.py            binary, LR + RF, with mitigation
    2. run_advanced_binary.py       binary, all 4 models, with mitigation
    3. run_multiclass.py            multiclass, LR + RF
    4. run_advanced.py              multiclass, all 4 models, balanced + unbalanced
    5. run_per_course.py            per-module binary
    6. run_per_course_multiclass.py per-module multiclass
  Tools:
    7. summarize_results.py         master comparison table
    8. tradeoff_plot.py             accuracy-fairness scatter
    9. build_report_assets.py       consolidated report folder

Usage:
    python src/run_all.py
"""

import subprocess
import sys
import time
import os


SCRIPTS = [
    ("Binary baseline (LR + RF)",                 "src/run_experiment.py"),
    ("Binary all-models + mitigation",            "src/run_advanced_binary.py"),
    ("Multiclass baseline (LR + RF)",             "src/run_multiclass.py"),
    ("Multiclass all-models (balanced + unbalanced)", "src/run_advanced.py"),
    ("Per-module binary",                         "src/run_per_course.py"),
    ("Per-module multiclass",                     "src/run_per_course_multiclass.py"),
    ("Summary table",                             "src/summarize_results.py"),
    ("Accuracy-fairness tradeoff plot",           "src/tradeoff_plot.py"),
    ("Report assets folder",                      "src/build_report_assets.py"),
]


def main():
    python_bin = sys.executable
    print(f"=== run_all.py — regenerating all results ===")
    print(f"Python: {python_bin}\n")

    overall_start = time.time()
    results = []

    for i, (label, script) in enumerate(SCRIPTS, 1):
        if not os.path.exists(script):
            print(f"[{i}/{len(SCRIPTS)}] SKIP — {script} not found")
            results.append((label, "missing", 0))
            continue

        print(f"[{i}/{len(SCRIPTS)}] {label}  ({script})")
        t0 = time.time()
        proc = subprocess.run([python_bin, script], capture_output=True, text=True)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            print(f"        OK  ({elapsed:.1f}s)")
            results.append((label, "ok", elapsed))
        else:
            print(f"        FAIL ({elapsed:.1f}s) — exit {proc.returncode}")
            print(f"        Last 10 lines of stderr:")
            for line in proc.stderr.splitlines()[-10:]:
                print(f"          {line}")
            results.append((label, "fail", elapsed))

    total = time.time() - overall_start
    print(f"\n=== Done in {total:.1f}s ===")
    n_ok   = sum(1 for _, s, _ in results if s == "ok")
    n_fail = sum(1 for _, s, _ in results if s == "fail")
    print(f"OK: {n_ok} / {len(SCRIPTS)}, FAIL: {n_fail}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
