#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_seizure.chbmit import build_window_labels, discover_chbmit_records, load_record_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare window tensors for DL training.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/chb-mit"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chbmit_tensors.npz"))
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument("--l-freq", type=float, default=0.5)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--notch", type=float, default=60.0)
    parser.add_argument("--resample", type=float, default=128.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--patients", nargs="*", default=None, help="Optional patient IDs, e.g., chb01 chb02")
    parser.add_argument(
        "--max-windows-per-record",
        type=int,
        default=1200,
        help="Cap windows per EDF record to limit memory/cost.",
    )
    return parser.parse_args()


def zscore_per_channel(window: np.ndarray) -> np.ndarray:
    mu = np.mean(window, axis=1, keepdims=True)
    sd = np.std(window, axis=1, keepdims=True) + 1e-6
    return (window - mu) / sd


def main() -> None:
    args = parse_args()
    records = discover_chbmit_records(args.raw_dir)

    if args.patients:
        keep = set(args.patients)
        records = [r for r in records if r.patient_id in keep]

    if args.max_records is not None:
        records = records[: args.max_records]

    if not records:
        raise RuntimeError("No records matched. Check --raw-dir or --patients.")

    raw_data_by_record = []
    meta = []
    channel_sets = []

    for rec in tqdm(records, desc="Loading EDF"):
        data, fs, channels = load_record_data(
            rec.edf_path,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch_freq=args.notch,
            resample_hz=args.resample,
        )
        raw_data_by_record.append((rec, data, fs, channels))
        meta.append(rec)
        channel_sets.append(set(channels))

    common_channels = sorted(set.intersection(*channel_sets))
    if not common_channels:
        raise RuntimeError("No common channels across selected records.")

    X_list = []
    y_list = []
    group_list = []
    file_list = []
    start_sec_list = []

    for rec, data, fs, channels in tqdm(raw_data_by_record, desc="Windowing"):
        idx = [channels.index(ch) for ch in common_channels]
        data = data[idx]

        starts, labels = build_window_labels(
            n_samples=data.shape[1],
            fs=fs,
            seizures=rec.seizures,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )

        if len(starts) == 0:
            continue

        if args.max_windows_per_record and len(starts) > args.max_windows_per_record:
            select = np.linspace(0, len(starts) - 1, args.max_windows_per_record, dtype=int)
            starts = starts[select]
            labels = labels[select]

        win_samples = int(round(args.window_sec * fs))
        for s, label in zip(starts, labels):
            window = data[:, s : s + win_samples]
            window = zscore_per_channel(window)
            X_list.append(window.astype(np.float32))
            y_list.append(int(label))
            group_list.append(rec.patient_id)
            file_list.append(rec.edf_path.name)
            start_sec_list.append(float(s / fs))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int8)
    groups = np.asarray(group_list)
    files = np.asarray(file_list)
    starts = np.asarray(start_sec_list, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        groups=groups,
        files=files,
        window_start_sec=starts,
        channels=np.asarray(common_channels, dtype=object),
        fs=np.asarray([args.resample], dtype=np.float32),
        window_sec=np.asarray([args.window_sec], dtype=np.float32),
        step_sec=np.asarray([args.step_sec], dtype=np.float32),
    )

    seizure_rate = float(np.mean(y)) if len(y) else 0.0
    print(f"Saved tensor dataset: {args.out}")
    print(f"Shape X: {X.shape} | seizure ratio: {seizure_rate:.4f}")
    print(f"Patients: {len(np.unique(groups))} | Channels: {len(common_channels)}")


if __name__ == "__main__":
    main()
