#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and per-patient metrics from predictions CSV.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def per_patient_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_col = "group" if "group" in df.columns else "patient_id"
    rows = []
    for group, part in df.groupby(group_col):
        y_true = part["y_true"].to_numpy()
        y_pred = part["y_pred"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows.append(
            {
                "patient_id": group,
                "n_windows": len(part),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("patient_id")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(df["model"].iloc[0])
    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    roc_path = args.out_dir / f"roc_{model_name}.png"
    plt.savefig(roc_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(f"PR Curve - {model_name}")
    plt.tight_layout()
    pr_path = args.out_dir / f"pr_{model_name}.png"
    plt.savefig(pr_path, dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    cm_path = args.out_dir / f"cm_{model_name}.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    per_patient_df = per_patient_metrics(df)
    per_patient_path = args.out_dir / f"per_patient_{model_name}.csv"
    per_patient_df.to_csv(per_patient_path, index=False)

    print(f"Saved: {roc_path}")
    print(f"Saved: {pr_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {per_patient_path}")


if __name__ == "__main__":
    main()
