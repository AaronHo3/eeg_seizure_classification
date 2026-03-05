#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_seizure.chbmit import build_window_labels, discover_chbmit_records, load_record_data
from eeg_seizure.features import extract_window_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CHB-MIT windows and engineered features.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/chb-mit"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chbmit_features.npz"))
    parser.add_argument("--summary-csv", type=Path, default=Path("data/processed/chbmit_windows.csv"))
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument("--l-freq", type=float, default=0.5)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch", type=float, default=60.0)
    parser.add_argument("--resample", type=float, default=128.0)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_chbmit_records(args.raw_dir)
    if args.max_records is not None:
        records = records[: args.max_records]

    if not records:
        raise RuntimeError(
            "No EDF records found. Place CHB-MIT data under data/raw/chb-mit/chbXX/*.edf"
        )

    all_X = []
    all_y = []
    all_group = []
    all_file = []
    all_start_sec = []
    feature_names = None

    for rec in tqdm(records, desc="Processing EDF records"):
        data, fs, channels = load_record_data(
            rec.edf_path,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freq=args.notch,
            resample_hz=args.resample,
        )

        starts, labels = build_window_labels(
            n_samples=data.shape[1],
            fs=fs,
            seizures=rec.seizures,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )
        if len(starts) == 0:
            continue

        win_samples = int(round(args.window_sec * fs))
        for s, label in zip(starts, labels):
            window = data[:, s : s + win_samples]
            feats, names = extract_window_features(window, fs)
            feature_names = names

            all_X.append(feats)
            all_y.append(int(label))
            all_group.append(rec.patient_id)
            all_file.append(rec.edf_path.name)
            all_start_sec.append(float(s / fs))

    X = np.vstack(all_X).astype(np.float32)
    y = np.asarray(all_y, dtype=np.int8)
    groups = np.asarray(all_group)
    files = np.asarray(all_file)
    starts = np.asarray(all_start_sec, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        groups=groups,
        files=files,
        window_start_sec=starts,
        feature_names=np.asarray(feature_names, dtype=object),
        channels=np.asarray(channels, dtype=object),
        window_sec=np.asarray([args.window_sec], dtype=np.float32),
        step_sec=np.asarray([args.step_sec], dtype=np.float32),
    )

    df = pd.DataFrame(
        {
            "file": files,
            "patient_id": groups,
            "window_start_sec": starts,
            "label": y,
        }
    )
    df.to_csv(args.summary_csv, index=False)

    seizure_rate = float(np.mean(y)) if len(y) > 0 else 0.0
    print(f"Saved features: {args.out}")
    print(f"Saved summary: {args.summary_csv}")
    print(f"Samples: {len(y)} | Features: {X.shape[1]} | Seizure ratio: {seizure_rate:.4f}")
    print(f"Patients: {len(np.unique(groups))}")


if __name__ == "__main__":
    main()
