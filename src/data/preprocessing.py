"""Utilities for loading and preprocessing the UCI heart disease dataset."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


REQUIRED_COLUMNS = {"cp", "restecg", "thal", "sex", "fbs"}
DROP_COLUMNS = ["id", "dataset"]
LABEL_ENCODE_COLUMNS = ["cp", "restecg", "thal"]
BINARY_MAP_COLUMNS = {
    "sex": {"male": 1, "female": 0},
    "fbs": {"true": 1, "false": 0},
}


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure expected columns are present before preprocessing."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")


def _resolve_target_column(df: pd.DataFrame) -> str:
    """Resolve a target column name commonly used in heart disease datasets."""
    if "num" in df.columns:
        return "num"
    if "target" in df.columns:
        return "target"
    raise ValueError("Could not find target column. Expected one of: 'num', 'target'.")


def _apply_binary_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Map specified binary categorical columns to numeric values."""
    for col, mapping in BINARY_MAP_COLUMNS.items():
        if col in df.columns:
            normalized = df[col].astype(str).str.strip().str.lower()
            df[col] = normalized.map(mapping)
    return df


def _fill_missing_values(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Fill missing values by mode for categorical and mean for numeric columns."""
    existing_categorical = [col for col in categorical_cols if col in df.columns]

    for col in existing_categorical:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in existing_categorical
    ]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    object_cols = [
        col
        for col in df.select_dtypes(include=["object", "category", "bool"]).columns
        if col not in existing_categorical
    ]
    for col in object_cols:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    return df


def _label_encode_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply label encoding to selected categorical columns."""
    for col in columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    return df


def _encode_remaining_categoricals(df: pd.DataFrame, exclude: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Label-encode any remaining non-numeric columns not already handled."""
    remaining = [
        col
        for col in df.columns
        if col not in exclude and df[col].dtype in ["object", "category", "bool"]
    ]

    for col in remaining:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df, remaining


def _scale_numerical_features(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Standardize continuous numeric feature columns."""
    X = X.copy()
    categorical_set = set(categorical_cols)
    numeric_cols = [
        col for col in X.select_dtypes(include=[np.number]).columns if col not in categorical_set
    ]

    if numeric_cols:
        X = X.astype({col: float for col in numeric_cols})
        scaler = StandardScaler()
        X.loc[:, numeric_cols] = scaler.fit_transform(X[numeric_cols].astype(float))

    return X


def load_and_preprocess(
    csv_path: str = "data/raw/heart_disease_uci.csv",
    use_global_scaling: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the heart disease dataset.

    Steps:
    1. Load CSV using pandas
    2. Drop unnecessary columns
    3. Map binary categorical columns (sex, fbs)
    4. Fill missing values (mean for numeric, mode for categorical)
    5. Label encode selected categorical columns (cp, restecg, thal)
    6. Split into features (X) and target (y)
    7. Optionally scale numerical features using StandardScaler

    Args:
        csv_path: Path to the raw CSV dataset.
        use_global_scaling: Whether to apply global StandardScaler normalization.
            Set to False when using client-wise scaling to avoid global normalization.

    Returns:
        Tuple containing:
        - X as a numpy array of processed features
        - y as a numpy array target
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input dataset is empty.")
    _validate_columns(df)

    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])
    df = _apply_binary_mappings(df)

    categorical_cols = ["sex", "fbs", *LABEL_ENCODE_COLUMNS]
    df = _fill_missing_values(df, categorical_cols=categorical_cols)
    df = _label_encode_columns(df, columns=LABEL_ENCODE_COLUMNS)
    df, extra_categoricals = _encode_remaining_categoricals(df, exclude=[])
    categorical_cols.extend(extra_categoricals)

    target_col = _resolve_target_column(df)
    y = df[target_col].to_numpy()
    if y.size == 0:
        raise ValueError("Target column is empty after preprocessing.")
    X = df.drop(columns=[target_col])

    if use_global_scaling:
        X = _scale_numerical_features(X, categorical_cols=categorical_cols)

    return X.to_numpy(dtype=np.float32), y
