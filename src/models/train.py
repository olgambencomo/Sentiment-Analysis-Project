# src/models/train.py
# Stage 3 — Build and train the final SVM pipeline.
#
# Reads:  data/splits/X_train.csv, y_train.csv
#         data/splits/X_val.csv,   y_val.csv
# Writes: models/svm_pipeline.pkl
#
# The best hyperparameters (C=0.1, max_features=5000, ngram_range=(1,2))
# were already found by GridSearch in notebook 04 and are stored in
# params.yaml — no GridSearch is run here.
#
# Notebook 05 logic: combine train + val, then fit the final model.
#
# Usage:
#   python src/models/train.py
#   python src/models/train.py --params params.yaml

import os
import argparse
import pickle

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(params: dict) -> Pipeline:
    """
    Builds the sklearn Pipeline with the best params from notebook 04:
        TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        + LinearSVC(C=0.1, class_weight='balanced')
    """
    tfidf = TfidfVectorizer(
        max_features=params["model"]["tfidf"]["max_features"],
        ngram_range=tuple(params["model"]["tfidf"]["ngram_range"]),
    )
    svm = LinearSVC(
        C=params["model"]["svm"]["C"],
        class_weight=params["model"]["svm"]["class_weight"],
    )
    return Pipeline([("tfidf", tfidf), ("classifier", svm)])


def main(params_path: str = "params.yaml"):
    params = load_params(params_path)

    text_col  = params["split"]["text_col"]
    label_col = params["split"]["label_col"]

    # ── Load train and val splits ──────────────────────────────────
    X_train = pd.read_csv("data/splits/X_train.csv").squeeze()
    y_train = pd.read_csv("data/splits/y_train.csv").squeeze()
    X_val   = pd.read_csv("data/splits/X_val.csv").squeeze()
    y_val   = pd.read_csv("data/splits/y_val.csv").squeeze()

    # ── Combine train + val for final training ─────────────────────
    # notebook 05: X_final = pd.concat([train['text'], val['text']])
    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train, y_val])
    print(f"Training on {len(X_final)} samples (train + val)")
    print(f"  Label distribution:\n{y_final.value_counts().to_string()}")

    # ── Build pipeline ─────────────────────────────────────────────
    pipeline = build_pipeline(params)
    print(f"\nPipeline: {pipeline}")

    # ── MLflow tracking ────────────────────────────────────────────
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="svm_train"):

        # Log all params to MLflow so every experiment is traceable
        mlflow.log_param("tfidf_max_features", params["model"]["tfidf"]["max_features"])
        mlflow.log_param("tfidf_ngram_range",  str(params["model"]["tfidf"]["ngram_range"]))
        mlflow.log_param("svm_C",              params["model"]["svm"]["C"])
        mlflow.log_param("svm_class_weight",   params["model"]["svm"]["class_weight"])
        mlflow.log_param("train_samples",      len(X_final))

        # ── Train ──────────────────────────────────────────────────
        print("\nFitting model...")
        pipeline.fit(X_final, y_final)
        print("Done.")

        # Log the model artifact to MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print(f"MLflow run id: {mlflow.active_run().info.run_id}")

    # ── Save pipeline as .pkl ──────────────────────────────────────
    output_path = params["model"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Train the SVM pipeline")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
