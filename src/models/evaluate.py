# src/models/evaluate.py
# Stage 4 — Evaluate the trained model on the test set.
#
# Reads:  models/svm_pipeline.pkl
#         data/splits/X_test.csv, y_test.csv
# Writes: reports/metrics.json
#         reports/confusion_matrix.png
#
# Usage:
#   python src/models/evaluate.py
#   python src/models/evaluate.py --params params.yaml

import os
import json
import pickle
import argparse

import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm, classes: list, output_path: str):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def main(params_path: str = "params.yaml"):
    params = load_params(params_path)

    model_path  = params["model"]["output_path"]
    metrics_path = params["reports"]["metrics_path"]
    cm_path      = params["reports"]["confusion_matrix_path"]

    # ── Load model ─────────────────────────────────────────────────
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Model loaded from: {model_path}")

    # ── Load test set ──────────────────────────────────────────────
    X_test = pd.read_csv("data/splits/X_test.csv").squeeze()
    y_test = pd.read_csv("data/splits/y_test.csv").squeeze()
    print(f"Test samples: {len(X_test)}")

    # ── Predict ────────────────────────────────────────────────────
    # notebook 05: y_test_pred = final_model.predict(X_test)
    y_pred  = pipeline.predict(X_test)
    classes = sorted(y_test.unique().tolist())

    # ── Metrics ────────────────────────────────────────────────────
    # notebook 05: classification_report(y_test, y_test_pred)
    #              accuracy_score(y_test, y_test_pred)
    accuracy   = accuracy_score(y_test, y_pred)
    f1_macro   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall     = recall_score(y_test, y_pred, average="macro",    zero_division=0)

    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics = {
        "accuracy":    round(accuracy,    4),
        "f1_macro":    round(f1_macro,    4),
        "f1_weighted": round(f1_weighted, 4),
        "precision":   round(precision,   4),
        "recall":      round(recall,      4),
        "test_samples": len(X_test),
    }

    # ── Save metrics.json ──────────────────────────────────────────
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # ── Confusion matrix ───────────────────────────────────────────
    # notebook 05: confusion_matrix(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plot_confusion_matrix(cm, classes, cm_path)

    # ── MLflow tracking ────────────────────────────────────────────
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="svm_evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(cm_path)
        print(f"MLflow run id: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Evaluate the model")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
