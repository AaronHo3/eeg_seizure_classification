from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CVConfig:
    strategy: str = "logo"  # logo or groupkfold
    n_splits: int = 5
    random_state: int = 42


def make_models(random_state: int = 42) -> dict[str, object]:
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )

    knn = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    voting = VotingClassifier(
        estimators=[("lr", clone(lr)), ("knn", clone(knn)), ("rf", clone(rf))],
        voting="soft",
        n_jobs=-1,
    )

    return {
        "logistic_regression": lr,
        "knn": knn,
        "random_forest": rf,
        "mlp": mlp,
        "soft_voting_ensemble": voting,
    }


def get_splitter(config: CVConfig, groups: np.ndarray):
    if config.strategy == "logo":
        return LeaveOneGroupOut()
    if config.strategy == "groupkfold":
        unique_groups = np.unique(groups)
        n_splits = min(config.n_splits, len(unique_groups))
        if n_splits < 2:
            raise ValueError("Need at least 2 groups for GroupKFold")
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unsupported CV strategy: {config.strategy}")


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def cross_val_predict_grouped(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: CVConfig,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = get_splitter(config, groups)

    y_prob = np.zeros_like(y, dtype=float)
    fold_ids = np.zeros_like(y, dtype=int)

    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(X, y, groups), start=1):
        m = clone(model)
        m.fit(X[tr_idx], y[tr_idx])
        y_prob[te_idx] = m.predict_proba(X[te_idx])[:, 1]
        fold_ids[te_idx] = fold_idx

    return y_prob, fold_ids


def random_split_baseline(
    model,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    n_splits: int = 5,
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_prob = np.zeros_like(y, dtype=float)

    for tr_idx, te_idx in skf.split(X, y):
        m = clone(model)
        m.fit(X[tr_idx], y[tr_idx])
        y_prob[te_idx] = m.predict_proba(X[te_idx])[:, 1]

    return y_prob


def save_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def save_metrics_json(metrics: dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def summarize_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    metrics = evaluate_binary(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    pred_df = pd.DataFrame(
        {
            "model": model_name,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "group": groups,
            "fold": fold_ids,
        }
    )
    return pred_df, metrics
