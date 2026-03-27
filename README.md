# Sentiment Analysis — Women's Fashion E-Commerce

Supervised NLP model that classifies customer reviews as Positive, Neutral, or Negative using a LinearSVC pipeline trained on 22,000+ real e-commerce reviews. Structured as a reproducible MLOps pipeline with experiment tracking and a live Streamlit deployment.

**[Live demo →](https://app-customer-insights-h6uezhauf6v4zfzt5xpykf.streamlit.app/)**

---

## Overview

Customer reviews contain valuable signals about product performance, but reading them at scale is not feasible. This project builds an end-to-end NLP system that automates sentiment classification, from raw CSV data to a deployed web application, following MLOps best practices for reproducibility and experiment tracking.

**Dataset:** Women's Clothing E-Commerce Reviews (23,486 reviews, Kaggle)  
**Task:** 3-class text classification — Positive / Neutral / Negative  
**Label source:** Customer ratings used as sentiment proxy (rating ≥ 4 → Positive, rating = 3 → Neutral, rating ≤ 2 → Negative)

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 81.7% |
| F1 macro | 0.61 |
| F1 Positive | 0.92 |
| F1 Neutral | 0.36 |
| F1 Negative | 0.55 |

The model performs strongly on the Positive class (77% of training data) and shows expected difficulty on Neutral, a known challenge when sentiment labels are derived from ratings rather than manual annotation.

---

## Why SVM over other models

Two models were evaluated via GridSearchCV with Stratified K-Fold cross-validation (k=3), using F1 macro as the optimization metric to handle class imbalance:

| Model | CV F1 macro |
|---|---|
| LinearSVC (C=0.1) | **0.613** |
| Logistic Regression (C=1) | 0.608 |

LinearSVC was selected for three reasons:
- Marginally better F1 macro on cross-validation
- Faster inference — relevant for real-time prediction in the app
- Well-suited for high-dimensional sparse feature spaces (TF-IDF)

---

## Class imbalance

The dataset has a significant imbalance that directly impacts model performance:

```
Positive    17,448   (77%)
Neutral      2,823   (12%)
Negative     2,370   (11%)
```

Two strategies were applied to address this:
1. **`class_weight='balanced'`** in LinearSVC — adjusts the penalty for each class inversely proportional to its frequency, preventing the model from defaulting to always predicting Positive
2. **Stratified splits** — ensures each train/val/test partition preserves the original class distribution

Despite these measures, the Neutral class remains challenging. Its lower F1 (0.36) reflects genuine label noise: a 3-star review can contain both positive and negative language, making it inherently ambiguous regardless of the model used.

---

## MLOps pipeline

Notebooks were refactored into a reproducible pipeline managed by DVC, with experiment tracking via MLflow.

```
data/raw/reviews.csv
        │
        ▼
[Stage 1] src/data/load.py          Clean text · create sentiment labels
        │
        ▼
[Stage 2] src/data/features.py      Train / val / test split (70 / 15 / 15)
        │
        ▼
[Stage 3] src/models/train.py       TF-IDF + LinearSVC · log to MLflow
        │
        ▼
[Stage 4] src/models/evaluate.py    Metrics · confusion matrix · log to MLflow
        │
        ▼
        models/svm_pipeline.pkl
```

`dvc repro` re-runs only the stages affected by a change. Modifying `model.svm.C` in `params.yaml` triggers only `train` and `evaluate` — not data processing.

---

## Project structure

```
├── src/
│   ├── data/
│   │   ├── load.py           Stage 1: load, clean, label
│   │   └── features.py       Stage 2: train/val/test split
│   └── models/
│       ├── train.py          Stage 3: train SVM pipeline
│       └── evaluate.py       Stage 4: evaluate on test set
├── app/
│   └── app.py                Streamlit application
├── notebooks/                Original exploratory notebooks
├── params.yaml               Central configuration
├── dvc.yaml                  Pipeline definition
├── Makefile                  Automation commands
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/olgambencomo/Sentiment-Analysis-Project.git
cd Sentiment-Analysis-Project

make setup                  # install dependencies + initialize DVC
# place your CSV in data/raw/
make run                    # run the full pipeline
make app                    # launch Streamlit locally
make mlflow                 # open MLflow UI at localhost:5000
```

---

## Tech stack

`Python` `scikit-learn` `NLTK` `pandas` `DVC` `MLflow` `Streamlit` `matplotlib`

---

## Limitations and next steps

- **Label noise** — rating-based labels don't always reflect the textual sentiment. Manual annotation or a pre-trained sentiment model for pseudo-labeling could improve Neutral/Negative performance.
- **Neutral class** — collecting more 3-star reviews or reframing as a binary task (Positive vs. Not Positive) would yield more actionable results for business use.
- **Transformer fine-tuning** — a fine-tuned BERT or DistilBERT would likely improve F1 macro by 10-15 points at the cost of inference speed and compute.
