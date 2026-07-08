"""Privacy-safe shared data preparation utilities for centralized, FL, and SplitFed pipelines."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.data.expansion import _detect_categorical_feature_indices, expand_dataset
from src.data.hospital_split import create_hospital_splits
from src.data.preprocessing import load_and_preprocess

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRIVACY_EXPANDED_PATH = PROCESSED_DIR / "privacy_expanded_12000.csv"
PRIVACY_CLIENT_DIR = PROJECT_ROOT / "clients" / "privacy"


def set_global_determinism(seed: int) -> None:
    """Set deterministic seeds for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_binary_labels(y: np.ndarray) -> np.ndarray:
    """Convert target labels to binary form."""
    y = np.asarray(y)
    unique = np.unique(y)
    if set(unique.tolist()).issubset({0, 1}):
        return y.astype(np.int32)
    return (y > 0).astype(np.int32)


def safe_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split with stratification when feasible, falling back safely for tiny class counts."""
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) > 1 and counts.min() >= 2
    stratify = y if can_stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _augment_partition(
    X: np.ndarray,
    y: np.ndarray,
    target_size: int,
    noise_std: float,
    seed: int,
    categorical_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap-resample with Gaussian noise on continuous features (no file IO).

    Identical augmentation method used for the train-origin and test-origin partitions
    separately, so the two never share source rows -> no train/test leakage.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=target_size, replace=True)
    X_aug = X[idx].astype(np.float64, copy=True)
    y_aug = y[idx].copy()

    continuous_idx = np.where(~categorical_mask)[0]
    if continuous_idx.size > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=(target_size, continuous_idx.size))
        X_aug[:, continuous_idx] += noise
        col_mins = X[:, continuous_idx].min(axis=0)
        col_maxs = X[:, continuous_idx].max(axis=0)
        X_aug[:, continuous_idx] = np.clip(X_aug[:, continuous_idx], col_mins, col_maxs)

    return X_aug.astype(np.float32), y_aug.astype(np.int32)


def _write_hospital_frame(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _ensure_privacy_hospital_csvs(
    target_size: int,
    random_state: int,
    csv_path: str,
    force_rebuild: bool,
) -> None:
    """Build leakage-free hospital corpora.

    The original (un-augmented) records are split into disjoint train-origin and
    test-origin sets FIRST. Only after that split is each side bootstrap-augmented, so a
    held-out patient's near-duplicates can never appear in any hospital's training data.
    Writes per-hospital train CSVs (non-IID) and per-hospital held-out test CSVs.
    """
    expected_train = [PRIVACY_CLIENT_DIR / f"hospital_{i}.csv" for i in range(1, 6)]
    expected_test = [PRIVACY_CLIENT_DIR / f"hospital_{i}_test.csv" for i in range(1, 6)]
    if not force_rebuild and all(p.exists() for p in expected_train + expected_test):
        return

    X_raw, y_raw = load_and_preprocess(csv_path=csv_path, use_global_scaling=False)
    y_binary = to_binary_labels(y_raw)

    # 1) Split the ORIGINAL rows before any augmentation (this is the leakage guard).
    X_tr_o, X_te_o, y_tr_o, y_te_o = safe_train_test_split(
        X=X_raw, y=y_binary, test_size=0.2, random_state=random_state
    )

    categorical_mask = _detect_categorical_feature_indices(X_raw)

    # 2) Augment each side independently from its own disjoint source rows.
    X_train_corpus, y_train_corpus = _augment_partition(
        X_tr_o, y_tr_o, target_size=target_size, noise_std=0.02,
        seed=random_state, categorical_mask=categorical_mask,
    )
    test_corpus_size = max(2000, target_size // 5)
    X_test_corpus, y_test_corpus = _augment_partition(
        X_te_o, y_te_o, target_size=test_corpus_size, noise_std=0.02,
        seed=random_state + 7, categorical_mask=categorical_mask,
    )

    # 3) Non-IID hospital TRAIN partitions come from the train corpus only.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _write_hospital_frame(X_train_corpus, y_train_corpus, PRIVACY_EXPANDED_PATH)
    PRIVACY_CLIENT_DIR.mkdir(parents=True, exist_ok=True)
    create_hospital_splits(
        input_path=str(PRIVACY_EXPANDED_PATH),
        output_dir=str(PRIVACY_CLIENT_DIR),
        total_samples=target_size,
        random_state=random_state,
    )

    # 4) Held-out TEST corpus is partitioned across hospitals (shuffled, disjoint origin).
    rng = np.random.default_rng(random_state + 99)
    perm = rng.permutation(X_test_corpus.shape[0])
    for i, chunk in enumerate(np.array_split(perm, 5), start=1):
        _write_hospital_frame(
            X_test_corpus[chunk], y_test_corpus[chunk],
            PRIVACY_CLIENT_DIR / f"hospital_{i}_test.csv",
        )


def build_privacy_preserving_splits(
    csv_path: str = str(PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"),
    target_size: int = 12000,
    test_size: float = 0.2,
    val_size_from_train: float = 0.1,
    random_state: int = 42,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    """Create consistent client-wise scaled splits for fair and leakage-free comparison.

    Returns a dictionary with per-client train/val/test arrays and concatenated global sets.
    """
    set_global_determinism(random_state)
    _ensure_privacy_hospital_csvs(
        target_size=target_size,
        random_state=random_state,
        csv_path=csv_path,
        force_rebuild=force_rebuild,
    )

    clients: Dict[int, Dict[str, np.ndarray]] = {}

    def _read_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Hospital file is empty: {path}")
        target_col = "target" if "target" in df.columns else "num" if "num" in df.columns else None
        if target_col is None:
            raise ValueError(f"Target column missing in {path}")
        y = to_binary_labels(df[target_col].to_numpy())
        X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
        return X, y

    for cid in range(1, 6):
        # Train rows come from the non-IID train corpus; test rows are the disjoint,
        # held-out corpus built from original patients never seen in any training row.
        X_train_full, y_train_full = _read_xy(PRIVACY_CLIENT_DIR / f"hospital_{cid}.csv")
        X_test, y_test = _read_xy(PRIVACY_CLIENT_DIR / f"hospital_{cid}_test.csv")

        X_train, X_val, y_train, y_val = safe_train_test_split(
            X=X_train_full,
            y=y_train_full,
            test_size=val_size_from_train,
            random_state=random_state + 100 + cid,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        clients[cid - 1] = {
            "X_train": X_train_scaled,
            "y_train": y_train.astype(np.int32),
            "X_val": X_val_scaled,
            "y_val": y_val.astype(np.int32),
            "X_test": X_test_scaled,
            "y_test": y_test.astype(np.int32),
        }

    all_train_X = np.concatenate([clients[cid]["X_train"] for cid in sorted(clients)], axis=0)
    all_train_y = np.concatenate([clients[cid]["y_train"] for cid in sorted(clients)], axis=0)
    all_val_X = np.concatenate([clients[cid]["X_val"] for cid in sorted(clients)], axis=0)
    all_val_y = np.concatenate([clients[cid]["y_val"] for cid in sorted(clients)], axis=0)
    all_test_X = np.concatenate([clients[cid]["X_test"] for cid in sorted(clients)], axis=0)
    all_test_y = np.concatenate([clients[cid]["y_test"] for cid in sorted(clients)], axis=0)

    return {
        "clients": clients,
        "feature_dim": int(all_train_X.shape[1]),
        "global_train": (all_train_X.astype(np.float32), all_train_y.astype(np.int32)),
        "global_val": (all_val_X.astype(np.float32), all_val_y.astype(np.int32)),
        "global_test": (all_test_X.astype(np.float32), all_test_y.astype(np.int32)),
    }
