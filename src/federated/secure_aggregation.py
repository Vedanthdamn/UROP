"""Simulated secure aggregation via pairwise additive masking.

Implements the core cancellation idea behind Bonawitz et al., "Practical Secure
Aggregation for Privacy-Preserving Machine Learning" (CCS 2017): each pair of clients
shares a random mask; every client adds its outgoing masks and subtracts its incoming
masks before sending its update to the server. When the server sums the masked updates,
every pairwise mask cancels out exactly, so the server recovers the correct aggregate
sum without ever seeing any individual client's plaintext update.

This module simulates that protocol at the algorithmic level (pairwise mask generation
and cancellation) rather than the full cryptographic key-agreement and Shamir
secret-sharing dropout-recovery machinery of the original protocol -- sufficient to
demonstrate, and formally reason about, the "server never sees an individual update"
property in a self-contained way, without a network/crypto stack.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _pairwise_mask(seed: int, client_i: int, client_j: int, shape: Tuple[int, ...]) -> np.ndarray:
    """Deterministic pseudo-random mask shared by exactly one pair (i, j), i != j.

    In the real protocol this shared randomness comes from a Diffie-Hellman key
    agreement between the two clients; here a symmetric deterministic seed keyed on the
    unordered pair (min, max) simulates the same "only these two clients know this mask"
    property for the purpose of the cancellation argument below.
    """
    lo, hi = min(client_i, client_j), max(client_i, client_j)
    pair_seed = seed ^ (lo * 1_000_003 + hi * 7_919)
    rng = np.random.default_rng(pair_seed)
    return rng.normal(loc=0.0, scale=1.0, size=shape)


def mask_vector(client_id: int, vector: np.ndarray, num_clients: int, seed: int) -> np.ndarray:
    """Mask one client's (already weight-scaled) update before it leaves the client.

    masked_i = vector_i + sum_{j > i} mask(i, j) - sum_{j < i} mask(i, j)

    Every mask(i, j) appears with opposite sign in client i's and client j's masked
    vectors, so it cancels out of the sum over all clients.
    """
    masked = vector.astype(np.float64).copy()
    for j in range(num_clients):
        if j == client_id:
            continue
        m = _pairwise_mask(seed, client_id, j, vector.shape)
        masked += m if j > client_id else -m
    return masked


def secure_weighted_aggregate(
    client_updates: List[Tuple[List[np.ndarray], int]],
    seed: int = 12345,
) -> List[np.ndarray]:
    """Securely compute the sample-weighted average of client model weights.

    Mirrors flwr's FedAvg weighting (each client contributes num_examples / total_examples
    of its update) but does so via masked per-client vectors: the server only ever sums
    masked vectors and never observes any individual client's plaintext weights.
    """
    num_clients = len(client_updates)
    total_examples = sum(n for _, n in client_updates)

    num_layers = len(client_updates[0][0])
    masked_per_client: List[List[np.ndarray]] = []
    for cid, (weights, num_examples) in enumerate(client_updates):
        coeff = num_examples / total_examples
        masked_layers = [
            mask_vector(cid, coeff * w, num_clients, seed=seed + layer_idx)
            for layer_idx, w in enumerate(weights)
        ]
        masked_per_client.append(masked_layers)

    aggregated = []
    for layer_idx in range(num_layers):
        layer_sum = sum(masked_per_client[cid][layer_idx] for cid in range(num_clients))
        aggregated.append(layer_sum.astype(np.float32))
    return aggregated


def verify_secure_aggregation(client_updates: List[Tuple[List[np.ndarray], int]], seed: int = 12345) -> dict:
    """Numerically demonstrate correctness and the non-leakage property.

    Returns:
        max_abs_error: the aggregate recovered via masking vs. the plaintext weighted
            average, which should be ~0 (correctness of the cancellation).
        individual_leakage_corr: correlation between one client's plaintext update and
            its masked version, which should be ~0 (a masked update looks like noise
            and does not resemble the plaintext it was derived from).
    """
    num_clients = len(client_updates)
    total_examples = sum(n for _, n in client_updates)

    num_layers = len(client_updates[0][0])
    plaintext_avg = []
    for layer_idx in range(num_layers):
        layer_sum = sum((n / total_examples) * weights[layer_idx] for weights, n in client_updates)
        plaintext_avg.append(layer_sum)

    secure_avg = secure_weighted_aggregate(client_updates, seed=seed)
    max_abs_error = max(float(np.max(np.abs(p - s))) for p, s in zip(plaintext_avg, secure_avg))

    # Leakage check on one client's first layer: correlate its plaintext (weighted)
    # contribution against its masked (pre-sum) vector.
    cid = 0
    weights0, n0 = client_updates[cid]
    coeff0 = n0 / total_examples
    plaintext0 = (coeff0 * weights0[0]).ravel()
    masked0 = mask_vector(cid, coeff0 * weights0[0], num_clients, seed=seed + 0).ravel()
    corr = float(np.corrcoef(plaintext0, masked0)[0, 1]) if plaintext0.size > 1 else float("nan")

    return {"max_abs_error": max_abs_error, "individual_leakage_corr": corr}
