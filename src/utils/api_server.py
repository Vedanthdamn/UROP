"""FastAPI backend for serving healthcare model training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI


app = FastAPI(title="Healthcare FL Metrics API", version="1.0.0")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_ROOT / "data" / "processed" / "metrics.json"
HOSPITAL_METRICS_FILE = PROJECT_ROOT / "data" / "processed" / "hospital_metrics.json"
GLOBAL_METRICS_FILE = PROJECT_ROOT / "data" / "processed" / "global_metrics.json"
CLIENTS_DIR = PROJECT_ROOT / "clients"


def _load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_hospital_stats_from_clients() -> Dict[str, Dict[str, float | int]]:
    """Build per-hospital stats from local client CSV files when no metrics file exists."""
    stats: Dict[str, Dict[str, float | int]] = {}

    for i in range(1, 6):
        path = CLIENTS_DIR / f"hospital_{i}.csv"
        if not path.exists():
            continue

        import pandas as pd

        df = pd.read_csv(path)
        if df.empty:
            stats[f"hospital_{i}"] = {
                "samples": 0,
                "positive_rate": 0.0,
            }
            continue

        target_col = "target" if "target" in df.columns else "num" if "num" in df.columns else None
        if target_col is None:
            stats[f"hospital_{i}"] = {
                "samples": int(len(df)),
                "positive_rate": 0.0,
            }
            continue

        y = df[target_col].to_numpy()
        y_binary = (y > 0).astype(int)

        stats[f"hospital_{i}"] = {
            "samples": int(len(df)),
            "positive_rate": float(y_binary.mean()),
        }

    return stats


@app.get("/metrics")
def get_metrics() -> Dict[str, float]:
    """Return overall training metrics (accuracy and loss)."""
    data = _load_json_if_exists(METRICS_FILE)
    if data is not None:
        return {
            "accuracy": float(data.get("accuracy", 0.0)),
            "loss": float(data.get("loss", 0.0)),
        }

    return {
        "accuracy": 0.0,
        "loss": 0.0,
    }


@app.get("/hospital-metrics")
def get_hospital_metrics() -> Dict[str, Dict[str, float | int]]:
    """Return per-hospital statistics in JSON format."""
    data = _load_json_if_exists(HOSPITAL_METRICS_FILE)
    if data is not None:
        return data

    return _infer_hospital_stats_from_clients()


@app.get("/global-model")
def get_global_model_metrics() -> Dict[str, float]:
    """Return global model accuracy in JSON format."""
    data = _load_json_if_exists(GLOBAL_METRICS_FILE)
    if data is not None:
        return {
            "global_accuracy": float(data.get("global_accuracy", 0.0)),
        }

    return {
        "global_accuracy": 0.0,
    }
