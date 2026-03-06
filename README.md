# EEG Seizure Classification (CHB-MIT)

**NOTICE**: This project is meant purely for learning and discovery.

End-to-end seizure vs non-seizure classification pipeline with patient-level validation.

## What This Project Includes
- CHB-MIT dataset parsing (`*-summary.txt` + `.edf`)
- EEG preprocessing (bandpass, notch, optional resampling)
- Windowing + seizure overlap labeling
- Engineered feature extraction (time-domain + spectral + Hjorth)
- Baseline models: Logistic Regression, kNN, Random Forest, MLP
- Soft-voting ensemble
- Deep learning baseline: compact 1D CNN on raw EEG windows (PyTorch)
- Grouped evaluation (Leave-One-Patient-Out or GroupKFold)
- ROC/PR/Confusion Matrix plots and per-patient metrics

## Folder Layout

```text
eeg_seizure_classification/
  data/
    raw/
    processed/
  reports/
    figures/
    runs/
  scripts/
    prepare_data.py
    prepare_tensor_data.py
    train.py
    train_dl.py
    evaluate.py
  src/eeg_seizure/
    chbmit.py
    dl.py
    features.py
    modeling.py
    utils.py
  pyproject.toml
  requirements.txt
```

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If needed:

```bash
pip install -r requirements.txt
```

## 2) Put CHB-MIT Data

Quick download for starter subset (default `chb01 chb02 chb03`):

```bash
make download-subset
```

Custom subset:

```bash
make download-subset DOWNLOAD_PATIENTS="chb01 chb02 chb03 chb04"
```

Expected structure:

```text
data/raw/chb-mit/
  chb01/
    chb01-summary.txt
    chb01_01.edf
    ...
  chb02/
    chb02-summary.txt
    chb02_01.edf
    ...
```

## 3) Prepare Features

```bash
python scripts/prepare_data.py \
  --raw-dir data/raw/chb-mit \
  --out data/processed/chbmit_features.npz \
  --summary-csv data/processed/chbmit_windows.csv \
  --window-sec 4 \
  --step-sec 2 \
  --l-freq 0.5 \
  --h-freq 40 \
  --notch 60 \
  --resample 128
```

## 4) Train + Cross-Validate

Leave-one-patient-out:

```bash
python scripts/train.py \
  --data data/processed/chbmit_features.npz \
  --out-dir reports/runs \
  --cv logo
```

GroupKFold:

```bash
python scripts/train.py \
  --data data/processed/chbmit_features.npz \
  --out-dir reports/runs \
  --cv groupkfold \
  --n-splits 5

# Optional: disable the random stratified baseline comparison
python scripts/train.py \
  --data data/processed/chbmit_features.npz \
  --out-dir reports/runs \
  --cv logo \
  --disable-random-baseline
```

Artifacts are created in `reports/runs/<timestamp>/`:
- `metrics_summary.csv`
- `metrics_*.json`
- `predictions_*.csv`
- `model_*.joblib`

## 5) Generate Evaluation Plots

```bash
python scripts/evaluate.py \
  --predictions reports/runs/<timestamp>/predictions_soft_voting_ensemble.csv \
  --out-dir reports/figures
```

Outputs:
- `roc_<model>.png`
- `pr_<model>.png`
- `cm_<model>.png`
- `per_patient_<model>.csv`

## 6) Deep Learning Pipeline 

Prepare tensor windows:

```bash
python scripts/prepare_tensor_data.py \
  --raw-dir data/raw/chb-mit \
  --out data/processed/chbmit_tensors.npz \
  --window-sec 4 \
  --step-sec 2 \
  --resample 128 \
  --max-windows-per-record 1200 \
  --patients chb01 chb02 chb03
```

Train 1D CNN with patient-level CV:

```bash
python scripts/train_dl.py \
  --data data/processed/chbmit_tensors.npz \
  --out-dir reports/runs_dl \
  --cv logo \
  --max-epochs 20 \
  --patience 4 \
  --batch-size 64 \
  --device auto
```

Or using Make:

```bash
make prepare-dl RAW_DIR=data/raw/chb-mit
make train-dl
make eval-dl RUN=$(make latest-run-dl)
```

DL artifacts are created in `reports/runs_dl/<timestamp>/`:
- `metrics_summary.csv`
- `metrics_eegcnn.json`
- `predictions_eegcnn1d.csv`
- `model_eegcnn_fold*.pt`

## Results (CHB-MIT subset: chb01, chb02, chb03)

Leave-one-patient-out (grouped LOGO) on 3 patients. Heavy class imbalance (seizure windows under 1% of total).

### Classical ML (engineered features)

| Model                 | ROC AUC | PR AUC | F1   | Recall (sensitivity) | Specificity |
|-----------------------|--------:|-------:|-----:|----------------------|------------:|
| Soft voting ensemble  |   0.89  |  0.38  | 0.43 | 0.32                 |        1.00 |
| MLP                   |   0.93  |  0.38  | 0.46 | 0.50                 |        1.00 |
| Logistic regression   |   0.86  |  0.23  | 0.06 | 0.72                 |        0.94 |
| Random forest         |   0.91  |  0.19  | 0.07 | 0.04                 |        1.00 |
| kNN                   |   0.78  |  0.33  | 0.39 | 0.26                 |        1.00 |

The soft voting ensemble and MLP give the best trade-off on this subset. Random split baselines (not shown) score much higher and indicate leakage if used without grouped CV.

### Deep learning (1D CNN on raw windows)

| Model     | ROC AUC | PR AUC | F1   | Recall | Specificity |
|-----------|--------:|-------:|-----:|-------:|------------:|
| EEGCNN1D  |   0.61  |  0.006 | 0.02 | 0.12   |        0.97 |

The CNN underperforms the classical pipeline on this small subset, likely due to limited data and no class weighting for the minority (seizure) class.

### Next steps

- **Class imbalance:** Add class weights (or focal loss) in the DL loss and consider oversampling/undersampling for both pipelines.
- **More data:** Download more patients (e.g. `make download-subset DOWNLOAD_PATIENTS="chb01 chb02 chb03 chb04 chb05"`) and re-run prepare + train + eval.
- **DL tuning:** Try more epochs, learning rate, or a slightly larger 1D CNN; add early stopping and class-weighted cross-entropy.
- **Per-model figures:** Run `make eval RUN=reports/runs/<timestamp> MODEL=<name>` for each classical model to generate ROC/PR/CM plots (e.g. `MODEL=soft_voting_ensemble`, `MODEL=mlp`).

## Notes

- Grouped CV keeps each patient fully in either train or test, preventing leakage.
- Random split baseline is included to quantify the optimism gap.
- Feature extraction aggregates per-channel descriptors using mean/std/max to support variable channel availability.
