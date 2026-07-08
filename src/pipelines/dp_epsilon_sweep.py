"""Privacy-utility tradeoff sweep: run DP-SGD federated training at several noise
multipliers, recording the formal epsilon and the resulting test accuracy at each
setting. This produces the epsilon-vs-accuracy curve referenced in the paper's DP
evaluation section.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.dp_federated_pipeline import run_dp_federated_pipeline

SWEEP_PATH = PROJECT_ROOT / "data" / "processed" / "dp_epsilon_sweep.json"

# 0.0 = no noise (clipping-only reference point, epsilon = infinity / no privacy guarantee).
NOISE_MULTIPLIERS = [0.0, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]


def run_sweep(
    noise_multipliers: List[float] = NOISE_MULTIPLIERS,
    rounds: int = 15,
    local_epochs: int = 2,
    clip_norm: float = 1.0,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for nm in noise_multipliers:
        print(f"\n=== DP-SGD sweep: noise_multiplier={nm} ===")
        res = run_dp_federated_pipeline(
            rounds=rounds,
            local_epochs=local_epochs,
            clip_norm=clip_norm,
            noise_multiplier=nm,
            random_state=random_state,
            save_metrics=False,
        )
        results.append({
            "noise_multiplier": nm,
            "epsilon": res["epsilon"] if res["epsilon"] != float("inf") else None,
            "delta": res["delta"],
            "accuracy": res["accuracy"],
            "f1_score": res["f1_score"],
            "roc_auc": res["roc_auc"],
            "loss": res["loss"],
            "total_local_steps": res["total_local_steps"],
        })

    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved sweep results to {SWEEP_PATH}")
    return results


if __name__ == "__main__":
    run_sweep()
