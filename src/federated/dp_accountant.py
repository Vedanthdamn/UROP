"""Renyi Differential Privacy (RDP) accountant for the sampled Gaussian mechanism.

Implements the standard moments-accountant technique used to report a formal privacy
budget (epsilon, delta) for DP-SGD, following:

- Mironov, "Renyi Differential Privacy" (CSF 2017) -- RDP of the Gaussian mechanism and
  the RDP -> (epsilon, delta)-DP conversion.
- Mironov, Talwar, Zhang, "Renyi Differential Privacy of the Sampled Gaussian Mechanism"
  (2019) -- closed-form RDP of the *subsampled* Gaussian mechanism at integer orders,
  which is what privacy amplification by minibatch sampling relies on.
- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016) -- the moments
  accountant composition strategy (RDP composes additively over training steps).

Only integer Renyi orders are used, which admits an exact closed-form expression (no
numerical integration needed) and is standard practice for a lightweight accountant.
This yields a mathematically valid but somewhat conservative (i.e. an over-estimate of
the true tightest) epsilon compared to fully optimized accountants that also search
fractional orders -- the safe direction of approximation error for a privacy claim.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

DEFAULT_ORDERS: tuple[int, ...] = tuple(range(2, 129))


def _log_comb(n: int, k: int) -> float:
    """log(C(n, k)) via log-gamma, stable for the moderate n used here (n <= ~64)."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP at integer order alpha for one application of the sampled Gaussian mechanism.

    Closed form (Mironov, Talwar & Zhang 2019, Eq. 3.4-3.5):

        exp((alpha - 1) * RDP(alpha)) = sum_{k=0}^{alpha} C(alpha, k)
            * (1 - q)^(alpha - k) * q^k * exp(k * (k - 1) / (2 * sigma^2))

    evaluated in log-space (via logsumexp) to avoid overflow for larger k / smaller sigma.
    q=0 (no subsampling) or sigma=0 (no noise) are handled as the natural limiting cases.
    """
    if sigma <= 0:
        return float("inf")
    if q <= 0:
        return 0.0
    if q >= 1.0:
        # No subsampling amplification: plain (non-subsampled) Gaussian mechanism RDP.
        return alpha / (2.0 * sigma**2)

    log_terms = np.empty(alpha + 1)
    for k in range(alpha + 1):
        log_terms[k] = (
            _log_comb(alpha, k)
            + (alpha - k) * math.log1p(-q)
            + k * math.log(q)
            + (k * (k - 1)) / (2.0 * sigma**2)
        )
    log_a = logsumexp(log_terms)
    return float(log_a / (alpha - 1))


def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: Sequence[int] = DEFAULT_ORDERS) -> np.ndarray:
    """Total RDP across `steps` compositions of the sampled Gaussian mechanism.

    RDP composes additively and exactly under sequential composition, so the total RDP at
    each order is simply `steps` times the per-step RDP at that order.
    """
    per_step = np.array([_rdp_sampled_gaussian(q, noise_multiplier, a) for a in orders])
    return per_step * steps


def get_privacy_spent(orders: Sequence[int], rdp: np.ndarray, delta: float) -> tuple[float, int]:
    """Convert an RDP guarantee to the tightest (epsilon, delta)-DP guarantee it implies.

    Standard conversion (Mironov 2017, Prop. 3): for any order alpha > 1,
        epsilon(delta) = RDP(alpha) + log(1/delta) / (alpha - 1).
    We minimize over the candidate orders to report the tightest epsilon.
    """
    orders = np.asarray(orders, dtype=np.float64)
    eps_per_order = rdp + np.log(1.0 / delta) / (orders - 1.0)
    best_idx = int(np.argmin(eps_per_order))
    return float(eps_per_order[best_idx]), int(orders[best_idx])


def compute_epsilon(
    q: float,
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> tuple[float, int]:
    """Convenience wrapper: (sampling rate, noise multiplier, #steps, delta) -> (epsilon, best_order)."""
    rdp = compute_rdp(q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
    return get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
