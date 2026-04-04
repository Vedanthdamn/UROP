"""Create non-IID hospital client datasets from the expanded healthcare dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/expanded_12000.csv"
OUTPUT_DIR = "clients"
N_HOSPITALS = 5
DEFAULT_TOTAL_SAMPLES = 12000


def _resolve_target_column(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    if "num" in df.columns:
        return "num"
    raise ValueError("Could not find target column. Expected one of: 'target', 'num'.")


def _resolve_feature_column(df: pd.DataFrame, primary: str, fallback_idx: int) -> str:
    """Resolve a feature column by semantic name, then by index fallback."""
    if primary in df.columns:
        return primary

    feature_cols = [c for c in df.columns if c != _resolve_target_column(df)]
    if fallback_idx >= len(feature_cols):
        raise ValueError(f"Could not resolve feature '{primary}' using fallback index {fallback_idx}.")
    return feature_cols[fallback_idx]


def _class_counts_for_sample(labels: pd.Series, n: int) -> Dict[int, int]:
    """Compute class counts preserving the source ratio as closely as possible."""
    value_counts = labels.value_counts(normalize=True).sort_index()
    if value_counts.empty:
        raise ValueError("Cannot compute class distribution from empty labels.")

    base_counts = {int(cls): int(np.floor(p * n)) for cls, p in value_counts.items()}
    assigned = sum(base_counts.values())

    # Distribute any remaining slots to classes with highest fractional remainder.
    remainders = sorted(
        [(int(cls), float((p * n) - base_counts[int(cls)])) for cls, p in value_counts.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    i = 0
    while assigned < n:
        cls = remainders[i % len(remainders)][0]
        base_counts[cls] += 1
        assigned += 1
        i += 1

    # Ensure at least one sample per class whenever feasible.
    if n >= len(base_counts):
        zero_classes = [cls for cls, count in base_counts.items() if count == 0]
        for cls in zero_classes:
            largest_cls = max(base_counts, key=base_counts.get)
            if base_counts[largest_cls] > 1:
                base_counts[largest_cls] -= 1
                base_counts[cls] = 1

    return base_counts


def _stratified_sample(df: pd.DataFrame, target_col: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows from df while keeping class balance reasonably close to source."""
    if df.empty:
        raise ValueError("Cannot sample from an empty dataframe.")

    desired_counts = _class_counts_for_sample(df[target_col], n)
    sampled_parts = []

    for cls, needed in desired_counts.items():
        cls_df = df[df[target_col] == cls]
        if cls_df.empty:
            continue
        replace = needed > len(cls_df)
        sampled = cls_df.sample(n=needed, replace=replace, random_state=int(rng.integers(0, 1_000_000)))
        sampled_parts.append(sampled)

    if not sampled_parts:
        raise ValueError("Failed to sample because no class partitions were available.")

    sampled_df = pd.concat(sampled_parts, axis=0)
    if len(sampled_df) < n:
        needed = n - len(sampled_df)
        extra = df.sample(n=needed, replace=(needed > len(df)), random_state=int(rng.integers(0, 1_000_000)))
        sampled_df = pd.concat([sampled_df, extra], axis=0)

    sampled_df = sampled_df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return sampled_df.iloc[:n].copy()


def _biased_pool(df: pd.DataFrame, feature_col: str, mode: str) -> pd.DataFrame:
    """Create biased candidate pools from semantic or scaled feature values."""
    series = df[feature_col]

    # If values are in original units (e.g., age years), use domain thresholds.
    # Otherwise (scaled values), use quantile thresholds to preserve intent.
    if mode == "younger":
        if series.max() > 10:
            return df[series < 45]
        q = series.quantile(0.40)
        return df[series <= q]

    if mode == "older":
        if series.max() > 10:
            return df[series > 50]
        q = series.quantile(0.60)
        return df[series >= q]

    if mode == "high_chol":
        if series.max() > 40:
            return df[series >= 240]
        q = series.quantile(0.65)
        return df[series >= q]

    if mode == "high_bp":
        if series.max() > 40:
            return df[series >= 140]
        q = series.quantile(0.65)
        return df[series >= q]

    raise ValueError(f"Unsupported bias mode: {mode}")


def create_hospital_splits(
    input_path: str = INPUT_PATH,
    output_dir: str = OUTPUT_DIR,
    n_hospitals: int = N_HOSPITALS,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 5 non-IID hospital datasets with approximately equal sample sizes.

    Hospital biases:
    1. Younger patients
    2. Older patients
    3. High cholesterol
    4. High blood pressure
    5. Mixed distribution

    The function writes:
    - clients/hospital_1.csv
    - clients/hospital_2.csv
    - clients/hospital_3.csv
    - clients/hospital_4.csv
    - clients/hospital_5.csv

    Returns:
        Tuple of five pandas DataFrames in hospital order.
    """
    if n_hospitals != 5:
        raise ValueError("This splitter is designed for exactly 5 hospital datasets.")

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input dataset is empty.")

    target_col = _resolve_target_column(df)
    age_col = _resolve_feature_column(df, primary="age", fallback_idx=0)
    bp_col = _resolve_feature_column(df, primary="trestbps", fallback_idx=3)
    chol_col = _resolve_feature_column(df, primary="chol", fallback_idx=4)

    per_hospital = total_samples // n_hospitals
    if per_hospital <= 0:
        raise ValueError("total_samples must be at least the number of hospitals.")

    rng = np.random.default_rng(seed=random_state)

    hosp1_pool = _biased_pool(df, age_col, mode="younger")
    hosp2_pool = _biased_pool(df, age_col, mode="older")
    hosp3_pool = _biased_pool(df, chol_col, mode="high_chol")
    hosp4_pool = _biased_pool(df, bp_col, mode="high_bp")

    # Mixed distribution comes from the full set.
    hosp5_pool = df

    # If any bias pool is too small, fall back to full dataset while keeping class-aware sampling.
    pools = [hosp1_pool, hosp2_pool, hosp3_pool, hosp4_pool, hosp5_pool]
    hospitals = []
    for pool in pools:
        source = pool if len(pool) >= max(200, per_hospital // 5) else df
        sampled = _stratified_sample(source, target_col=target_col, n=per_hospital, rng=rng)
        sampled = sampled.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        hospitals.append(sampled)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, hospital_df in enumerate(hospitals, start=1):
        hospital_df.to_csv(output_path / f"hospital_{i}.csv", index=False)

    return tuple(hospitals)  # type: ignore[return-value]
