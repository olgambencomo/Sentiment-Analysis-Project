# src/data/features.py
# Stage 2 — Split clean data into train / val / test sets.
#
# Reads:  data/processed/clean_data.csv
# Writes: data/splits/X_train.csv, y_train.csv
#         data/splits/X_val.csv,   y_val.csv
#         data/splits/X_test.csv,  y_test.csv
#
# Usage:
#   python src/data/features.py
#   python src/data/features.py --params params.yaml

import os
import argparse

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(params_path: str = "params.yaml"):
    params = load_params(params_path)

    clean_path    = params["data"]["clean_path"]
    val_test_size = params["split"]["val_test_size"]   # 0.3
    test_size     = params["split"]["test_size"]       # 0.5
    random_state  = params["split"]["random_state"]    # 42
    stratify      = params["split"]["stratify"]        # true
    text_col      = params["split"]["text_col"]        # "text"
    label_col     = params["split"]["label_col"]       # "label"

    print(f"Loading clean data from: {clean_path}")
    df = pd.read_csv(clean_path)
    print(f"  Total rows: {len(df)}")

    X = df["Clean_Review"]
    y = df["sentiment"]

    # First split: 70% train, 30% temp  (notebook 03, first train_test_split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_size,
        stratify=y if stratify else None,
        random_state=random_state,
    )

    # Second split: 50% val, 50% test from the 30% temp  (notebook 03, second train_test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size,
        stratify=y_temp if stratify else None,
        random_state=random_state,
    )

    print(f"  Train: {len(X_train)} rows ({len(X_train)/len(df)*100:.0f}%)")
    print(f"  Val:   {len(X_val)} rows ({len(X_val)/len(df)*100:.0f}%)")
    print(f"  Test:  {len(X_test)} rows ({len(X_test)/len(df)*100:.0f}%)")

    splits_dir = "data/splits"
    os.makedirs(splits_dir, exist_ok=True)

    X_train.to_csv(f"{splits_dir}/X_train.csv", index=False, header=[text_col])
    y_train.to_csv(f"{splits_dir}/y_train.csv", index=False, header=[label_col])
    X_val.to_csv(f"{splits_dir}/X_val.csv",     index=False, header=[text_col])
    y_val.to_csv(f"{splits_dir}/y_val.csv",     index=False, header=[label_col])
    X_test.to_csv(f"{splits_dir}/X_test.csv",   index=False, header=[text_col])
    y_test.to_csv(f"{splits_dir}/y_test.csv",   index=False, header=[label_col])

    print(f"\nSplits saved to: {splits_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Split data into train/val/test")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
