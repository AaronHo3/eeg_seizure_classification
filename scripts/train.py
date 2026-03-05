#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_seizure.modeling import (
    CVConfig,
    cross_val_predict_grouped,
    make_models,
    random_split_baseline,
    save_metrics_json,
    save_model,
    summarize_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate seizure classifiers.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/chbmit_features.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/runs"))
    parser.add_argument("--cv", choices=["logo", "groupkfold"], default="logo")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--disable-random-baseline",
        action="store_true",
        help="Disable random stratified split baseline (enabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blob = np.load(args.data, allow_pickle=True)
    required_keys = {"X", "y", "groups"}
    missing = required_keys.difference(blob.files)
    if missing:
        raise ValueError(
            f"Input dataset is missing required arrays: {sorted(missing)}. "
            f"Found: {sorted(blob.files)}"
        )

    X = blob["X"]
    y = blob["y"]
    groups = blob["groups"].astype(str)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = CVConfig(strategy=args.cv, n_splits=args.n_splits, random_state=args.random_state)
    models = make_models(random_state=args.random_state)

    all_metrics = []

    for name, model in models.items():
        print(f"Evaluating: {name}")
        y_prob, fold_ids = cross_val_predict_grouped(model, X, y, groups, config)
        pred_df, metrics = summarize_predictions(name, y, y_prob, groups, fold_ids)

        pred_df.to_csv(run_dir / f"predictions_{name}.csv", index=False)
        save_metrics_json(metrics, run_dir / f"metrics_{name}.json")

        row = {"model": name, "evaluation": f"grouped_{args.cv}", **metrics}
        all_metrics.append(row)

        final_model = clone(model).fit(X, y)
        save_model(final_model, run_dir / f"model_{name}.joblib")

        if not args.disable_random_baseline:
            random_prob = random_split_baseline(
                model=model,
                X=X,
                y=y,
                random_state=args.random_state,
                n_splits=args.n_splits,
            )
            random_fold_ids = np.zeros_like(y, dtype=int)
            random_pred_df, random_metrics = summarize_predictions(
                f"{name}_random_split",
                y,
                random_prob,
                groups,
                random_fold_ids,
            )
            random_pred_df.to_csv(run_dir / f"predictions_{name}_random_split.csv", index=False)
            save_metrics_json(random_metrics, run_dir / f"metrics_{name}_random_split.json")
            all_metrics.append(
                {
                    "model": name,
                    "evaluation": "stratified_random_split",
                    **random_metrics,
                }
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(by=["evaluation", "roc_auc"], ascending=[True, False])
    metrics_df.to_csv(run_dir / "metrics_summary.csv", index=False)

    print("\nMetrics Summary")
    print(metrics_df.to_string(index=False))
    for eval_name in metrics_df["evaluation"].drop_duplicates():
        subset = metrics_df[metrics_df["evaluation"] == eval_name].sort_values(
            by="roc_auc", ascending=False
        )
        best = subset.iloc[0]
        print(
            f"Best model ({eval_name}): {best['model']} | "
            f"ROC-AUC={best['roc_auc']:.4f} | F1={best['f1']:.4f} | MCC={best['mcc']:.4f}"
        )
    print(f"\nArtifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
