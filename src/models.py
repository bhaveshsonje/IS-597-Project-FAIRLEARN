"""
models.py — Train and evaluate baseline models (Logistic Regression, Random Forest).
"""

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def train_models(X_train, y_train):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return {"logistic_regression": lr, "random_forest": rf}


def run_training(X, y, results_dir: str):
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
