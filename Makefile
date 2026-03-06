PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

RAW_DIR ?= data/raw/chb-mit
FEATURES_NPZ ?= data/processed/chbmit_features.npz
TENSORS_NPZ ?= data/processed/chbmit_tensors.npz
WINDOWS_CSV ?= data/processed/chbmit_windows.csv
RUNS_DIR ?= reports/runs
RUNS_DL_DIR ?= reports/runs_dl
FIGURES_DIR ?= reports/figures
DOWNLOAD_PATIENTS ?= chb01 chb02 chb03

WINDOW_SEC ?= 4
STEP_SEC ?= 2
L_FREQ ?= 0.5
H_FREQ ?= 40
NOTCH ?= 60
RESAMPLE ?= 128

CV ?= logo
N_SPLITS ?= 5
INCLUDE_RANDOM_BASELINE ?= 1
MODEL ?= soft_voting_ensemble
DL_MODEL ?= eegcnn1d
RUN ?=

.PHONY: help venv install setup download-subset prepare prepare-dl train train-dl latest-run latest-run-dl eval eval-dl all all-dl clean

# Auto venv setup: create venv if missing, then install
$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

setup: $(VENV)/bin/python
	$(ACTIVATE) && pip install -q -e .

help:
	@echo "Targets:"
	@echo "  make setup                Create venv (if needed) and install project"
	@echo "  make venv                 Create virtual environment only"
	@echo "  make install              Install project in editable mode (requires active venv)"
	@echo "  make download-subset      Download CHB-MIT subset (default: chb01 chb02 chb03)"
	@echo "  make prepare              Build processed features from CHB-MIT raw data"
	@echo "  make prepare-dl           Build tensor windows for deep learning"
	@echo "  make train                Run grouped CV training and save artifacts"
	@echo "  make train-dl             Run grouped CV deep learning training"
	@echo "  make latest-run           Print latest run directory path"
	@echo "  make latest-run-dl        Print latest DL run directory path"
	@echo "  make eval RUN=<dir>       Generate ROC/PR/CM + per-patient metrics"
	@echo "  make eval-dl RUN=<dir>    Generate ROC/PR/CM from DL predictions"
	@echo "  make all                  Run prepare + train + eval on MODEL"
	@echo "  make all-dl               Run prepare-dl + train-dl + eval-dl"
	@echo ""
	@echo "Variables you can override:"
	@echo "  RAW_DIR=$(RAW_DIR)"
	@echo "  DOWNLOAD_PATIENTS=$(DOWNLOAD_PATIENTS)"
	@echo "  FEATURES_NPZ=$(FEATURES_NPZ)  TENSORS_NPZ=$(TENSORS_NPZ)"
	@echo "  CV=$(CV)  N_SPLITS=$(N_SPLITS)  MODEL=$(MODEL)  DL_MODEL=$(DL_MODEL)"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && pip install -e .

download-subset:
	bash scripts/download_subset.sh $(RAW_DIR) $(DOWNLOAD_PATIENTS)

prepare: setup
	$(ACTIVATE) && python scripts/prepare_data.py \
		--raw-dir $(RAW_DIR) \
		--out $(FEATURES_NPZ) \
		--summary-csv $(WINDOWS_CSV) \
		--window-sec $(WINDOW_SEC) \
		--step-sec $(STEP_SEC) \
		--l-freq $(L_FREQ) \
		--h-freq $(H_FREQ) \
		--notch $(NOTCH) \
		--resample $(RESAMPLE)

prepare-dl: setup
	$(ACTIVATE) && python scripts/prepare_tensor_data.py \
		--raw-dir $(RAW_DIR) \
		--out $(TENSORS_NPZ) \
		--window-sec $(WINDOW_SEC) \
		--step-sec $(STEP_SEC) \
		--l-freq $(L_FREQ) \
		--h-freq $(H_FREQ) \
		--notch $(NOTCH) \
		--resample $(RESAMPLE)

train: setup
	$(ACTIVATE) && python scripts/train.py \
		--data $(FEATURES_NPZ) \
		--out-dir $(RUNS_DIR) \
		--cv $(CV) \
		--n-splits $(N_SPLITS) \
		$(if $(filter 0 false FALSE no NO,$(INCLUDE_RANDOM_BASELINE)),--disable-random-baseline,)

train-dl: setup
	$(ACTIVATE) && python scripts/train_dl.py \
		--data $(TENSORS_NPZ) \
		--out-dir $(RUNS_DL_DIR) \
		--cv $(CV) \
		--n-splits $(N_SPLITS)

latest-run:
	@ls -1dt $(RUNS_DIR)/* 2>/dev/null | head -n 1

latest-run-dl:
	@ls -1dt $(RUNS_DL_DIR)/* 2>/dev/null | head -n 1

eval: setup
	@if [ -z "$(RUN)" ]; then \
		echo "RUN is required. Example: make eval RUN=reports/runs/20260304_123000"; \
		exit 1; \
	fi
	$(ACTIVATE) && python scripts/evaluate.py \
		--predictions $(RUN)/predictions_$(MODEL).csv \
		--out-dir $(FIGURES_DIR)

eval-dl: setup
	@if [ -z "$(RUN)" ]; then \
		echo "RUN is required. Example: make eval-dl RUN=reports/runs_dl/20260304_123000"; \
		exit 1; \
	fi
	$(ACTIVATE) && python scripts/evaluate.py \
		--predictions $(RUN)/predictions_$(DL_MODEL).csv \
		--out-dir $(FIGURES_DIR)

all: prepare train
	$(eval RUN := $(shell ls -1dt $(RUNS_DIR)/* 2>/dev/null | head -n 1))
	@if [ -z "$(RUN)" ]; then \
		echo "No run directory found under $(RUNS_DIR)."; \
		exit 1; \
	fi
	$(MAKE) eval RUN=$(RUN) MODEL=$(MODEL)

all-dl: prepare-dl train-dl
	$(eval RUN := $(shell ls -1dt $(RUNS_DL_DIR)/* 2>/dev/null | head -n 1))
	@if [ -z "$(RUN)" ]; then \
		echo "No run directory found under $(RUNS_DL_DIR)."; \
		exit 1; \
	fi
	$(MAKE) eval-dl RUN=$(RUN) DL_MODEL=$(DL_MODEL)

clean:
	rm -rf scripts/__pycache__ src/eeg_seizure/__pycache__
