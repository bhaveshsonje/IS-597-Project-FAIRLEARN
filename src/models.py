"""
models.py — Train and evaluate models: Logistic Regression, Random Forest, XGBoost, MLP.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def upsample_minorities(X_train, y_train, random_state: int = 42):
    """Random-oversample minority classes so all classes have equal counts.

    Used as a pre-processing step for MLP since sklearn MLPClassifier.fit()
    does not accept sample_weight or class_weight.
    """
    df = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    df["_target"] = y_train.values if hasattr(y_train, "values") else y_train

    counts = df["_target"].value_counts()
    max_n  = counts.max()

    parts = []
    for cls in counts.index:
        cls_df = df[df["_target"] == cls]
        if len(cls_df) < max_n:
            cls_df = resample(cls_df, replace=True, n_samples=max_n,
                              random_state=random_state)
        parts.append(cls_df)

    balanced = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_bal = balanced["_target"]
    X_bal = balanced.drop(columns=["_target"])
    return X_bal, y_bal


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    n_classes = len(set(y_test))
    if n_classes == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "auc": round(auc, 4),
    }


def train_models(X_train, y_train, balanced: bool = False):
    """Train baseline models: Logistic Regression + Random Forest.

    balanced : if True, use class_weight='balanced' to upweight minority classes.
    """
    cw = "balanced" if balanced else None
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight=cw),
    )
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight=cw,
    )
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return {"logistic_regression": lr, "random_forest": rf}


def train_xgboost(X_train, y_train, balanced: bool = False):
    """Train XGBoost — auto-configures for binary or multiclass.

    balanced : if True, compute per-sample weights inversely proportional to class frequency.
    """
    n_classes = len(set(y_train))
    params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"
    else:
        params["eval_metric"] = "logloss"

    xgb = XGBClassifier(**params)
    if balanced:
        sample_weights = compute_sample_weight("balanced", y_train)
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        xgb.fit(X_train, y_train)
    return xgb


def train_mlp(X_train, y_train, balanced: bool = False):
    """Train MLP with two hidden layers inside a StandardScaler pipeline.

    balanced : if True, random-oversample minority classes BEFORE training.
               Workaround because sklearn MLPClassifier.fit() doesn't accept
               sample_weight or class_weight.
    """
    if balanced:
        X_train, y_train = upsample_minorities(X_train, y_train)

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    )
    mlp.fit(X_train, y_train)
    return mlp


def train_all_models(X_train, y_train, balanced: bool = False):
    """Train all four models: LR, RF, XGBoost, MLP.

    balanced : LR + RF use class_weight='balanced', XGBoost uses sample_weight,
               MLP uses random oversampling of minorities.
    """
    bal_str = " (balanced)" if balanced else ""
    print(f"  Training Logistic Regression{bal_str}...")
    print(f"  Training Random Forest{bal_str}...")
    models = train_models(X_train, y_train, balanced=balanced)
    print(f"  Training XGBoost{bal_str}...")
    models["xgboost"] = train_xgboost(X_train, y_train, balanced=balanced)
    if balanced:
        print("  Training MLP (random-oversampled minorities)...")
    else:
        print("  Training MLP...")
    models["mlp"] = train_mlp(X_train, y_train, balanced=balanced)
    return models


def plot_confusion_matrices(models: dict, X_test, y_test, class_labels: dict,
                            results_dir: str):
    """Save confusion matrix heatmap for each model."""
    os.makedirs(results_dir, exist_ok=True)
    label_ints  = sorted(class_labels.keys())
    label_names = [class_labels[i] for i in label_ints]

    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=label_ints)
        # Row-normalize so each row sums to 1 (recall per true class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        _, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=30, ha="right")
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} — confusion matrix (row-normalized)")
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                txt = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/confusion_matrix_{name}.png", dpi=100)
        plt.close()


def plot_model_comparison(metrics: dict, results_dir: str):
    """Bar chart comparing all models on Accuracy, F1, AUC."""
    model_names = list(metrics.keys())
    metric_keys = ["accuracy", "f1", "auc"]
    x = np.arange(len(model_names))
    width = 0.25

    _, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(metric_keys):
        vals = [metrics[m][key] for m in model_names]
        ax.bar(x + i * width, vals, width, label=key.upper())

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy / F1 / AUC")
    ax.legend()
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/model_comparison.png", dpi=100)
    plt.close()


def run_training(X, y, results_dir: str):
    """Train baseline models (LR + RF) and save metrics."""
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_models(X_train, y_train)

    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluate(model, X_test, y_test)

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Model Performance ===")
    for name, m in metrics.items():
        print(f"{name}: Accuracy={m['accuracy']} | F1={m['f1']} | AUC={m['auc']}")

    return models, metrics, (X_train, X_test, y_train, y_test)


def run_training_all(X, y, results_dir: str, balanced: bool = False,
                     class_labels: dict = None):
    """Train all four models (LR, RF, XGBoost, MLP) and save metrics + plots.

    class_labels : if provided, also save confusion-matrix heatmaps per model.
    """
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_all_models(X_train, y_train, balanced=balanced)

    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluate(model, X_test, y_test)

    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Model Performance (All Models) ===")
    for name, m in metrics.items():
        print(f"{name:25s}: Accuracy={m['accuracy']} | F1={m['f1']} | AUC={m['auc']}")

    plot_model_comparison(metrics, results_dir)
    if class_labels is not None:
        plot_confusion_matrices(models, X_test, y_test, class_labels, results_dir)

    return models, metrics, (X_train, X_test, y_train, y_test)
