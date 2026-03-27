# src/data/load.py
# Stage 1 — Load raw data, clean text, create sentiment labels.
#
# Usage:
#   python src/data/load.py
#   python src/data/load.py --params params.yaml

import os
import re
import argparse

import pandas as pd
import yaml
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_clean_text(params: dict):
    stop_words = set(stopwords.words(params["preprocessing"]["language"]))

    def clean_text(text: str) -> str:
        text = str(text).lower()
        if params["preprocessing"]["remove_urls"]:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        if params["preprocessing"]["remove_non_alpha"]:
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
        if params["preprocessing"]["remove_stopwords"]:
            text = " ".join(w for w in text.split() if w not in stop_words)
        return text.strip()

    return clean_text


def sentiment_label(rating: int, pos_threshold: int, neg_threshold: int) -> str:
    if rating >= pos_threshold:
        return "Positive"
    elif rating <= neg_threshold:
        return "Negative"
    else:
        return "Neutral"


def main(params_path: str = "params.yaml"):
    params     = load_params(params_path)
    raw_path   = params["data"]["raw_path"]
    text_col   = params["data"]["text_column"]
    rating_col = params["data"]["rating_column"]
    clean_path = params["data"]["clean_path"]
    pos_thr    = params["labeling"]["positive_threshold"]
    neg_thr    = params["labeling"]["negative_threshold"]

    print(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  Rows loaded: {len(df)}")

    df = df[[text_col, rating_col]].dropna()
    print(f"  Rows after dropping nulls: {len(df)}")

    print("  Cleaning text...")
    clean_text       = build_clean_text(params)
    df["Clean_Review"] = df[text_col].apply(clean_text)

    df["sentiment"] = df[rating_col].apply(
        lambda r: sentiment_label(int(r), pos_thr, neg_thr)
    )
    print(f"  Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}")

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f"\nSaved to: {clean_path}  |  shape: {df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Load and clean data")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)
