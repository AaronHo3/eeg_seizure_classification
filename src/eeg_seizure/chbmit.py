from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np


@dataclass(frozen=True)
class ChbmitRecord:
    patient_id: str
    edf_path: Path
    seizures: tuple[tuple[float, float], ...]


_FILE_RE = re.compile(r"^File Name:\s*(.+)$", re.IGNORECASE)
_START_RE = re.compile(r"^Seizure\s+\d+\s+Start Time:\s*(\d+(?:\.\d+)?)\s*seconds", re.IGNORECASE)
_END_RE = re.compile(r"^Seizure\s+\d+\s+End Time:\s*(\d+(?:\.\d+)?)\s*seconds", re.IGNORECASE)


def _normalize_file_key(name: str) -> str:
    return Path(name.strip()).name.lower()


def parse_summary_file(summary_path: Path) -> dict[str, tuple[tuple[float, float], ...]]:
    """Parse CHB-MIT summary file into filename -> seizure intervals in seconds."""
    results: dict[str, list[tuple[float, float]]] = {}
    current_file: str | None = None
    pending_start: float | None = None

    with summary_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            file_match = _FILE_RE.match(line)
            if file_match:
                current_file = _normalize_file_key(file_match.group(1))
                results.setdefault(current_file, [])
                pending_start = None
                continue

            if current_file is None:
                continue

            start_match = _START_RE.match(line)
            if start_match:
                pending_start = float(start_match.group(1))
                continue

            end_match = _END_RE.match(line)
            if end_match and pending_start is not None:
                end = float(end_match.group(1))
                start = min(pending_start, end)
                stop = max(pending_start, end)
                results[current_file].append((start, stop))
                pending_start = None

    return {k: tuple(v) for k, v in results.items()}


def discover_chbmit_records(raw_root: Path) -> list[ChbmitRecord]:
    """Discover EDF files and seizure annotations from CHB-MIT folder structure."""
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_root}")

    records: list[ChbmitRecord] = []
    patient_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        summary_files = sorted(patient_dir.glob("*-summary.txt"))

        annotations: dict[str, tuple[tuple[float, float], ...]] = {}
        for summary_file in summary_files:
            annotations.update(parse_summary_file(summary_file))

        for edf_path in sorted(patient_dir.glob("*.edf")):
            key = _normalize_file_key(edf_path.name)
            seizures = annotations.get(key, tuple())
            records.append(
                ChbmitRecord(
                    patient_id=patient_id,
                    edf_path=edf_path,
                    seizures=seizures,
                )
            )

    return records


def load_record_data(
    edf_path: Path,
    channel_whitelist: Iterable[str] | None = None,
    l_freq: float | None = 0.5,
    h_freq: float | None = 40.0,
    notch_freq: float | None = 60.0,
    resample_hz: float | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Load EDF into channels x samples."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    # Keep EEG channels if channel types are present.
    try:
        raw.pick("eeg")
    except Exception:
        pass

    if channel_whitelist is not None:
        channel_whitelist_set = {c.strip() for c in channel_whitelist}
        present = [ch for ch in raw.ch_names if ch in channel_whitelist_set]
        if present:
            raw.pick(present)

    if notch_freq is not None:
        raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    if resample_hz is not None:
        raw.resample(resample_hz, verbose="ERROR")

    data = raw.get_data()
    fs = float(raw.info["sfreq"])
    channels = list(raw.ch_names)
    return data, fs, channels


def build_window_labels(
    n_samples: int,
    fs: float,
    seizures: tuple[tuple[float, float], ...],
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create window start sample indices and binary labels."""
    win = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 0 or step <= 0:
        raise ValueError("window_sec and step_sec must produce positive sample counts")
    if n_samples < win:
        return np.array([], dtype=int), np.array([], dtype=int)

    starts = np.arange(0, n_samples - win + 1, step, dtype=int)
    labels = np.zeros(len(starts), dtype=int)

    for i, s in enumerate(starts):
        win_start = s / fs
        win_end = (s + win) / fs
        for seiz_start, seiz_end in seizures:
            if win_end > seiz_start and win_start < seiz_end:
                labels[i] = 1
                break

    return starts, labels
