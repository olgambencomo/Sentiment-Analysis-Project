# Makefile
# Automation commands for the sentiment analysis MLOps project.
#
# Usage:
#   make setup      → install dependencies and initialize DVC
#   make run        → execute the full pipeline with DVC
#   make train      → run only the training stage
#   make evaluate   → run only the evaluation stage
#   make metrics    → print current metrics
#   make mlflow     → open the MLflow UI in the browser
#   make app        → launch the Streamlit app
#   make clean      → remove all generated files (keeps raw data)
#   make help       → show this menu

PYTHON = python
PIP    = pip
PARAMS = params.yaml

.PHONY: all setup install init-dvc run load features train evaluate metrics mlflow app clean help

# ── Default target ─────────────────────────────────────────────────
all: help

# ── Setup ──────────────────────────────────────────────────────────
# Runs install + init-dvc in one command.
# This is the first thing you run on a new machine.
setup: install init-dvc
	@echo ""
	@echo "Setup complete. Place your CSV in data/raw/ and run: make run"

install:
	$(PIP) install -r requirements.txt

init-dvc:
	dvc init --no-scm 2>/dev/null || true
	mkdir -p data/raw data/processed data/splits models reports src/data src/models

# ── Full pipeline (DVC) ────────────────────────────────────────────
# DVC runs only the stages affected by any change.
# First time: runs all 4 stages.
# After changing model.svm.C in params.yaml: runs only train + evaluate.
run:
	dvc repro
	@$(MAKE) metrics

# ── Individual stages ──────────────────────────────────────────────
# Useful during development when you want to re-run a single stage
# without going through DVC.
load:
	$(PYTHON) src/data/load.py --params $(PARAMS)

features:
	$(PYTHON) src/data/features.py --params $(PARAMS)

train:
	$(PYTHON) src/models/train.py --params $(PARAMS)

evaluate:
	$(PYTHON) src/models/evaluate.py --params $(PARAMS)

# ── Metrics ────────────────────────────────────────────────────────
# Shows the current metrics.json.
# After multiple runs, dvc metrics diff compares against the last git commit.
metrics:
	@echo ""
	@echo "── Current metrics ───────────────────────────────────────"
	@cat reports/metrics.json 2>/dev/null || echo "No metrics yet. Run: make run"
	@echo ""

metrics-diff:
	dvc metrics diff

# ── MLflow UI ──────────────────────────────────────────────────────
# Opens the experiment tracker at http://localhost:5000
mlflow:
	mlflow ui --backend-store-uri mlruns

# ── Streamlit app ──────────────────────────────────────────────────
app:
	streamlit run app/app.py

# ── DAG ────────────────────────────────────────────────────────────
dag:
	dvc dag

# ── Clean ──────────────────────────────────────────────────────────
# Removes all generated files. Raw data in data/raw/ is preserved.
clean:
	rm -rf data/processed data/splits models reports mlruns
	@echo "Cleaned. Raw data preserved in data/raw/"

# ── Help ───────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Sentiment Fashion — MLOps Pipeline"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make setup        Install dependencies and initialize DVC"
	@echo "  make run          Run the full pipeline (DVC)"
	@echo "  make load         Stage 1: load and clean data"
	@echo "  make features     Stage 2: build train/val/test splits"
	@echo "  make train        Stage 3: train the SVM model"
	@echo "  make evaluate     Stage 4: evaluate on test set"
	@echo "  make metrics      Show current metrics"
	@echo "  make metrics-diff Compare metrics with last git commit"
	@echo "  make mlflow       Open MLflow UI (localhost:5000)"
	@echo "  make app          Launch Streamlit app"
	@echo "  make dag          Show the DVC pipeline graph"
	@echo "  make clean        Remove all generated files"
	@echo ""
