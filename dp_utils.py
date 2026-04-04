"""Differential privacy helpers for gradient perturbation."""

from __future__ import annotations

from typing import List

import tensorflow as tf


def apply_dp(
    gradients: List[tf.Tensor | None],
    noise_multiplier: float,
    clip_norm: float = 1.0,
) -> List[tf.Tensor | None]:
    """Clip gradients and add Gaussian noise for differential privacy.

    Args:
        gradients: Raw gradients from backpropagation.
        noise_multiplier: Noise scale factor relative to clip_norm.
        clip_norm: L2 norm bound used for clipping.

    Returns:
        A list of privatized gradients matching input structure.
    """
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be >= 0.")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0.")

    processed: List[tf.Tensor | None] = []
    for grad in gradients:
        if grad is None:
            processed.append(None)
            continue

        clipped = tf.clip_by_norm(grad, clip_norm)
        noise_std = noise_multiplier * clip_norm
        noise = tf.random.normal(shape=tf.shape(clipped), mean=0.0, stddev=noise_std, dtype=clipped.dtype)
        processed.append(clipped + noise)

    return processed
