from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _line_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def _zero_crossing_rate(x: np.ndarray) -> float:
    zc = np.where(np.diff(np.signbit(x)))[0]
    return float(len(zc) / max(1, len(x) - 1))


def _hjorth_params(x: np.ndarray) -> tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)

    var0 = np.var(x)
    var1 = np.var(dx) if len(dx) > 1 else 0.0
    var2 = np.var(ddx) if len(ddx) > 1 else 0.0

    activity = float(var0)
    mobility = float(np.sqrt(var1 / var0)) if var0 > 0 else 0.0
    complexity = float(np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0.0
    return activity, mobility, complexity


def _bandpowers(x: np.ndarray, fs: float) -> dict[str, float]:
    freqs, pxx = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    total_power = np.trapz(pxx, freqs) + 1e-12
    out: dict[str, float] = {}
    for name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = np.trapz(pxx[mask], freqs[mask]) if np.any(mask) else 0.0
        out[name] = float(bp / total_power)
    return out


def _spectral_entropy(x: np.ndarray, fs: float) -> float:
    freqs, pxx = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    pxx = pxx + 1e-12
    p = pxx / np.sum(pxx)
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


def extract_channel_features(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    ll = _line_length(x)
    zcr = _zero_crossing_rate(x)

    hj_activity, hj_mobility, hj_complexity = _hjorth_params(x)
    spec_entropy = _spectral_entropy(x, fs)
    skw = float(skew(x, bias=False, nan_policy="omit"))
    krt = float(kurtosis(x, bias=False, nan_policy="omit"))
    bps = _bandpowers(x, fs)

    names = [
        "mean",
        "std",
        "rms",
        "line_length",
        "zcr",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "spectral_entropy",
        "skew",
        "kurtosis",
    ] + [f"bandpower_{k}" for k in BANDS]

    vals = [
        mean,
        std,
        rms,
        ll,
        zcr,
        hj_activity,
        hj_mobility,
        hj_complexity,
        spec_entropy,
        skw,
        krt,
    ] + [bps[k] for k in BANDS]

    arr = np.asarray(vals, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, names


def extract_window_features(window: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    """
    window: channels x samples

    Returns aggregated feature vector:
    for each channel feature -> mean/std/max across channels.
    """
    if window.ndim != 2:
        raise ValueError("Expected window with shape channels x samples")

    per_channel = []
    base_names: list[str] | None = None
    for ch in range(window.shape[0]):
        feats, names = extract_channel_features(window[ch], fs)
        base_names = names
        per_channel.append(feats)

    feat_mat = np.vstack(per_channel)

    means = np.mean(feat_mat, axis=0)
    stds = np.std(feat_mat, axis=0)
    maxs = np.max(feat_mat, axis=0)

    agg = np.concatenate([means, stds, maxs], axis=0)
    assert base_names is not None
    out_names = (
        [f"mean_{n}" for n in base_names]
        + [f"std_{n}" for n in base_names]
        + [f"max_{n}" for n in base_names]
    )
    return agg.astype(float), out_names
