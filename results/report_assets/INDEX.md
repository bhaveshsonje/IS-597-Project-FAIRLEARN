# Report Assets — IS 597 Final Report
This folder is self-contained: every figure and table the IEEE report references lives here. Regenerate with `python src/build_report_assets.py`.

## Folder map
| Folder | What's inside | Report section |
|--------|---------------|----------------|
| `overview/` | Master summary table (all 36 model runs) + accuracy-fairness tradeoff plot | §11 Discussion |
| `aggregated/binary_advanced/` | 4 models in binary mode, full mitigation, SHAP on XGBoost | §6 Models, §7 Fairness, §8 Mitigation, §9 Explainability |
| `aggregated/multiclass_unbalanced/` | 4 models, multiclass, no class weights — accuracy-optimized baseline | §6 Models |
| `aggregated/multiclass_balanced/` | 4 models, multiclass, with class weights — fairness-optimized variant | §6 Models, §7 Fairness |
| `per_module_binary/<MODULE>/` | LR + RF + mitigation, per OULAD module | §10 Per-Course Analysis |
| `per_module_multiclass/<MODULE>/` | 4 models + class weights, per OULAD module | §10 Per-Course Analysis |

## Headline numbers (best AUC per pipeline)
| Pipeline | Best model | AUC | Accuracy |
|----------|-----------|-----|----------|
| aggregated/binary_advanced | xgboost | 0.8133 | 0.732 |
| aggregated/multiclass_unbalanced | xgboost | 0.7546 | 0.5429 |
| aggregated/multiclass_balanced | xgboost | 0.7492 | 0.4869 |
| per_module_binary/BBB | random_forest | 0.8217 | 0.7345 |
| per_module_binary/CCC | logistic_regression | 0.825 | 0.7452 |
| per_module_binary/DDD | logistic_regression | 0.8537 | 0.7769 |
| per_module_binary/FFF | logistic_regression | 0.8284 | 0.7392 |
| per_module_multiclass/BBB | xgboost | 0.766 | 0.5411 |
| per_module_multiclass/CCC | logistic_regression | 0.7675 | 0.4927 |
| per_module_multiclass/DDD | xgboost | 0.7595 | 0.5219 |
| per_module_multiclass/FFF | xgboost | 0.7609 | 0.5319 |

## Headline figures
- `overview/tradeoff_plot.png` — the report's main visualization. Shows every model's accuracy vs max TPR gap, with arrows indicating mitigation effect.
- `aggregated/binary_advanced/model_comparison.png` — bar chart of all 4 binary models on Accuracy/F1/AUC.
- `aggregated/binary_advanced/confusion_matrix_xgboost.png` — confusion matrix for the best binary model.
- `aggregated/multiclass_balanced/confusion_matrix_*.png` — multiclass confusion matrices showing Pass/Distinction confusion.
- `aggregated/binary_advanced/shap_summary.png` — global feature importance for XGBoost.
- `aggregated/binary_advanced/mitigation_threshold_*.png` — TPR before vs after thresholding.

## File-naming conventions
- `model_metrics.json` — accuracy/F1/AUC per model
- `fairness_metrics.json` — per-group TPR (binary) or per-class recall (multiclass)
- `mitigation_reweighting.json` — accuracy of reweighted models
- `mitigation_thresholding.json` — TPR gap before/after per-group thresholding
- `confusion_matrix_<model>.png` — row-normalized heatmap (multiclass only)
- `fairness_<model>_<group>.png` — binary fairness bar chart
- `fairness_mc_<model>_<group>.png` — multiclass fairness bar chart (Fail/Withdrawn recall)
- `shap_summary.png`, `shap_importance.png` — global SHAP plots
- `shap_student_<n>.png` — per-student SHAP waterfall (top 5 flagged)
