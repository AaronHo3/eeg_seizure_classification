#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_seizure.dl import EEGCNN1D, TrainConfig, evaluate_binary, predict_probs, train_one_fold
from eeg_seizure.modeling import CVConfig, get_splitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG 1D CNN with patient-level CV.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/chbmit_tensors.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/runs_dl"))
    parser.add_argument("--cv", choices=["logo", "groupkfold"], default="logo")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    torch.manual_seed(args.random_state)

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    groups = blob["groups"].astype(str)

    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )

    cv_cfg = CVConfig(strategy=args.cv, n_splits=args.n_splits, random_state=args.random_state)
    splitter = get_splitter(cv_cfg, groups)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Training device: {device}")

    y_prob = np.zeros(len(y), dtype=np.float32)
    fold_ids = np.zeros(len(y), dtype=int)

    folds = list(splitter.split(X, y, groups))
    for fold_num, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)

        perm = rng.permutation(tr_idx)
        val_size = max(1, int(0.15 * len(perm)))
        dev_idx = perm[:val_size]
        fit_idx = perm[val_size:]
        if len(fit_idx) == 0:
            fit_idx = dev_idx

        model = EEGCNN1D(n_channels=X.shape[1], n_timesteps=X.shape[2])
        model, _ = train_one_fold(
            model=model,
            X_train=X[fit_idx],
            y_train=y[fit_idx],
            X_dev=X[dev_idx],
            y_dev=y[dev_idx],
            cfg=cfg,
            device=device,
        )

        fold_prob = predict_probs(
            model=model,
            X=X[te_idx],
            y=y[te_idx],
            batch_size=cfg.batch_size,
            device=device,
        )
        y_prob[te_idx] = fold_prob
        fold_ids[te_idx] = fold_num

        torch.save(model.state_dict(), run_dir / f"model_eegcnn_fold{fold_num}.pt")
        print(f"Finished fold {fold_num}/{len(folds)} | test windows: {len(te_idx)}")

    metrics = evaluate_binary(y, y_prob)
    save_json(run_dir / "metrics_eegcnn.json", metrics)

    y_pred = (y_prob >= 0.5).astype(int)
    pred_df = pd.DataFrame(
        {
            "model": "eegcnn1d",
            "y_true": y,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "group": groups,
            "fold": fold_ids,
        }
    )
    pred_df.to_csv(run_dir / "predictions_eegcnn1d.csv", index=False)

    summary_df = pd.DataFrame([{"model": "eegcnn1d", "evaluation": f"grouped_{args.cv}", **metrics}])
    summary_df.to_csv(run_dir / "metrics_summary.csv", index=False)

    config_out = {
        "train_config": cfg.__dict__,
        "cv_config": cv_cfg.__dict__,
        "data_shape": tuple(map(int, X.shape)),
        "device": str(device),
    }
    save_json(run_dir / "run_config.json", config_out)

    print("\n=== DL Metrics Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nArtifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
