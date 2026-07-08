"""Generate the DP privacy-utility tradeoff figure from the epsilon sweep results.

Run after `src/pipelines/dp_epsilon_sweep.py`. Produces `plots/dp_epsilon_tradeoff.png`:
test accuracy vs. formal privacy budget (epsilon), on a log-scale x-axis since epsilon
spans several orders of magnitude across noise multipliers.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SWEEP_PATH = PROJECT_ROOT / "data" / "processed" / "dp_epsilon_sweep.json"
PLOTS = PROJECT_ROOT / "plots"


def main() -> None:
    results = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))

    finite = [r for r in results if r["epsilon"] is not None]
    finite.sort(key=lambda r: r["epsilon"])
    no_dp = [r for r in results if r["epsilon"] is None]

    eps = [r["epsilon"] for r in finite]
    acc = [r["accuracy"] for r in finite]

    plt.figure(figsize=(8, 5))
    plt.plot(eps, acc, marker="o", linewidth=2, color="#7c3aed", label="DP-SGD (per-example clip + noise)")
    for r in finite:
        plt.annotate(f"$\\sigma$={r['noise_multiplier']}", (r["epsilon"], r["accuracy"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    if no_dp:
        plt.axhline(no_dp[0]["accuracy"], color="#9ca3af", linestyle="--", linewidth=1.5,
                    label=f"No noise / clip-only reference ({no_dp[0]['accuracy']:.3f})")
    plt.xscale("log")
    plt.xlabel("Privacy budget $\\varepsilon$ (log scale, $\\delta=10^{-5}$) -- smaller is more private")
    plt.ylabel("Test Accuracy")
    plt.title("Privacy-Utility Tradeoff: DP-SGD Federated Training")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / "dp_epsilon_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", PLOTS / "dp_epsilon_tradeoff.png")


if __name__ == "__main__":
    main()
