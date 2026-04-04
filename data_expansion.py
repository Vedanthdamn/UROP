"""Dataset expansion utilities for tabular healthcare ML workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_PATH = "data/processed/expanded_12000.csv"


def _detect_categorical_feature_indices(X: np.ndarray, max_unique: int = 10) -> np.ndarray:
    """Detect likely categorical/binary columns using unique-value heuristics."""
    n_samples, n_features = X.shape
    categorical_mask = np.zeros(n_features, dtype=bool)

    for col_idx in range(n_features):
        col = X[:, col_idx]
        unique_vals = np.unique(col)

        # Likely categorical if low-cardinality or integer-coded with few values.
        is_low_cardinality = unique_vals.size <= max_unique
        is_integer_like = np.allclose(unique_vals, np.round(unique_vals), rtol=0.0, atol=1e-8)
        is_small_ratio = unique_vals.size <= max(2, int(0.05 * n_samples))

        if is_low_cardinality or (is_integer_like and is_small_ratio):
            categorical_mask[col_idx] = True

    return categorical_mask


def _clip_continuous_columns(X_aug: np.ndarray, X_ref: np.ndarray, continuous_idx: np.ndarray) -> None:
    """Clip augmented continuous values to the original observed ranges."""
    if continuous_idx.size == 0:
        return

    col_mins = X_ref[:, continuous_idx].min(axis=0)
    col_maxs = X_ref[:, continuous_idx].max(axis=0)
    X_aug[:, continuous_idx] = np.clip(X_aug[:, continuous_idx], col_mins, col_maxs)

    # If the first continuous feature is strictly positive in source data
    # (commonly age), keep it non-negative after clipping.
    first_cont_col = continuous_idx[0]
    if np.nanmin(X_ref[:, first_cont_col]) > 0:
        X_aug[:, first_cont_col] = np.maximum(X_aug[:, first_cont_col], 1e-6)


def expand_dataset(
    X: np.ndarray,
    y: np.ndarray,
    target_size: int = 12000,
    noise_std: float = 0.02,
    random_state: int = 42,
    categorical_indices: np.ndarray | list[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand a dataset using bootstrap sampling with controlled noise injection.

    Steps:
    1. Bootstrap sample rows with replacement until target_size
    2. Add Gaussian noise only on likely continuous features
    3. Keep categorical/binary columns unchanged
    4. Clip continuous features to valid source ranges
    5. Save expanded dataset to data/processed/expanded_12000.csv

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        target_size: Number of samples in the expanded dataset.
        noise_std: Std dev for Gaussian noise; must be between 0.01 and 0.05.
        random_state: Seed for reproducibility.
        categorical_indices: Optional explicit categorical/binary column indices.
            If provided, these columns are excluded from noise injection.

    Returns:
        Tuple of (X_aug, y_aug) as numpy arrays.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of shape (n_samples,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if target_size < X.shape[0]:
        raise ValueError("target_size must be >= original sample count.")
    if not (0.01 <= noise_std <= 0.05):
        raise ValueError("noise_std must be between 0.01 and 0.05.")

    rng = np.random.default_rng(seed=random_state)

    sample_idx = rng.choice(X.shape[0], size=target_size, replace=True)
    X_aug = X[sample_idx].astype(np.float64, copy=True)
    y_aug = y[sample_idx].copy()

    if categorical_indices is None:
        categorical_mask = _detect_categorical_feature_indices(X)
    else:
        categorical_mask = np.zeros(X.shape[1], dtype=bool)
        categorical_indices = np.asarray(categorical_indices, dtype=int)
        if categorical_indices.size > 0:
            if np.any(categorical_indices < 0) or np.any(categorical_indices >= X.shape[1]):
                raise ValueError("categorical_indices contains out-of-range column index.")
            categorical_mask[categorical_indices] = True

    continuous_idx = np.where(~categorical_mask)[0]

    if continuous_idx.size > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=(target_size, continuous_idx.size))
        X_aug[:, continuous_idx] += noise
        _clip_continuous_columns(X_aug, X, continuous_idx)

    output_path = Path(DEFAULT_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [f"feature_{i}" for i in range(X_aug.shape[1])]
    out_df = pd.DataFrame(X_aug, columns=feature_cols)
    out_df["target"] = y_aug
    out_df.to_csv(output_path, index=False)

    return X_aug, y_aug
