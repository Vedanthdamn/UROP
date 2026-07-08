"""Federated pipeline with real DP-SGD: per-example gradient clipping + calibrated
Gaussian noise, and a formal (epsilon, delta) privacy budget via the RDP accountant.

This mirrors the non-DP FedProx pipeline (same leakage-free data, same model
architecture) so the two are directly comparable; the only difference is the local
optimizer, which follows the DP-SGD algorithm of Abadi et al. (2016):

    1. Compute the PER-EXAMPLE gradient of the loss (not the batch-averaged gradient).
    2. Clip each example's gradient to L2 norm <= `clip_norm`.
    3. Average the clipped per-example gradients over the batch.
    4. Add Gaussian noise with std = clip_norm * noise_multiplier / batch_size.

Clipping bounds each example's contribution to the update (the mechanism's
sensitivity); the added noise is what makes the mechanism differentially private. The
resulting sequence of noisy updates is accounted for with `src.federated.dp_accountant`
to report a single formal epsilon for the whole federated training run.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from flwr.server.strategy.aggregate import aggregate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_privacy_preserving_splits
from src.federated.audit_ledger import AuditLedger
from src.federated.dp_accountant import compute_epsilon
from src.federated.secure_aggregation import secure_weighted_aggregate
from src.models import build_full_model

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DP_METRICS_PATH = PROCESSED_DIR / "dp_metrics.json"
AUDIT_LEDGER_PATH = PROCESSED_DIR / "audit_ledger.json"


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _flatten_per_example_grads(jacobians: List[tf.Tensor], batch_size: int) -> tf.Tensor:
    """Stack per-variable per-example jacobians into one (batch, total_params) matrix."""
    flat = [tf.reshape(j, (batch_size, -1)) for j in jacobians]
    return tf.concat(flat, axis=1)


def _make_dp_sgd_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, batch_size: int):
    """Build a @tf.function-compiled DP-SGD step bound to a persistent model/optimizer.

    Compiling once and reusing the same model/optimizer object across every client and
    round (only resetting weights between clients) avoids repeated Python-eager op
    dispatch and retracing, which otherwise dominates runtime for per-example jacobians.
    """
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")

    @tf.function(reduce_retracing=True)
    def step(x_batch: tf.Tensor, y_batch: tf.Tensor, clip_norm: tf.Tensor, noise_multiplier: tf.Tensor) -> None:
        _dp_sgd_step_body(model, optimizer, bce, x_batch, y_batch, clip_norm, noise_multiplier, batch_size)

    return step


def _dp_sgd_step_body(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    bce: tf.keras.losses.Loss,
    x_batch: tf.Tensor,
    y_batch: tf.Tensor,
    clip_norm: tf.Tensor,
    noise_multiplier: tf.Tensor,
    batch_size: int,
) -> None:
    """One DP-SGD update: per-example clip -> average -> add calibrated Gaussian noise."""
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        per_example_loss = bce(tf.reshape(y_batch, (-1, 1)), preds)

    # Per-example gradients for every trainable variable: shape (batch, *var.shape).
    jacobians = tape.jacobian(per_example_loss, model.trainable_variables)
    flat_grads = _flatten_per_example_grads(jacobians, batch_size)  # (batch, total_params)

    # Clip each example's gradient to L2 norm <= clip_norm (bounds per-example sensitivity).
    per_example_norms = tf.norm(flat_grads, axis=1, keepdims=True)
    clip_factor = clip_norm / tf.maximum(per_example_norms, clip_norm)
    clipped = flat_grads * clip_factor

    # Average clipped per-example gradients, then add noise calibrated to the clip bound.
    summed = tf.reduce_sum(clipped, axis=0)
    noise_std = clip_norm * noise_multiplier
    noise = tf.random.normal(shape=summed.shape, mean=0.0, stddev=noise_std, dtype=summed.dtype)
    noised_mean = (summed + noise) / float(batch_size)

    # Unflatten back into per-variable gradients and apply the update.
    grads = []
    offset = 0
    for var in model.trainable_variables:
        size = var.shape.num_elements()  # static (Python int), safe inside a traced tf.function
        grads.append(tf.reshape(noised_mean[offset:offset + size], var.shape))
        offset += size
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def _fit_dp_sgd(
    model: tf.keras.Model,
    compiled_step,
    X: np.ndarray,
    y: np.ndarray,
    local_epochs: int,
    batch_size: int,
    clip_norm: float,
    noise_multiplier: float,
    seed: int,
) -> int:
    """Run local DP-SGD training with a pre-compiled step; returns steps taken."""
    dataset = (
        tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)  # fixed batch size keeps sensitivity/noise consistent
    )
    clip_norm_t = tf.constant(clip_norm, dtype=tf.float32)
    noise_multiplier_t = tf.constant(noise_multiplier, dtype=tf.float32)
    steps = 0
    for _ in range(local_epochs):
        for x_batch, y_batch in dataset:
            y_batch = tf.cast(y_batch, tf.float32)
            compiled_step(x_batch, y_batch, clip_norm_t, noise_multiplier_t)
            steps += 1
    return steps


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "loss": float(loss),
    }


def run_dp_federated_pipeline(
    rounds: int = 15,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    random_state: int = 42,
    save_metrics: bool = True,
    use_secure_aggregation: bool = False,
    record_ledger: bool = True,
) -> Dict[str, Any]:
    """Run federated training with per-client DP-SGD and report a formal epsilon.

    Uses the same leakage-free splits and model architecture as the non-DP pipelines so
    results are directly comparable; only the local optimizer changes. Optionally routes
    aggregation through the simulated secure-aggregation protocol (pairwise masking, see
    `src.federated.secure_aggregation`) so the server never sees an individual client's
    plaintext update, and records a hash-chained audit-ledger entry every round.
    """
    _set_global_determinism(random_state)
    ledger = AuditLedger() if record_ledger else None

    split_payload = build_privacy_preserving_splits(random_state=random_state, force_rebuild=False)
    clients = split_payload["clients"]
    feature_dim = int(split_payload["feature_dim"])
    X_test, y_test = split_payload["global_test"]

    global_model = build_full_model(input_dim=feature_dim, learning_rate=learning_rate)
    global_weights = global_model.get_weights()

    # One persistent model + compiled DP-SGD step PER CLIENT, reused across rounds (only
    # weights are reset each round). This lets tf.function trace once instead of retracing
    # for a freshly-built model every round, which otherwise dominates runtime.
    local_models: Dict[int, tf.keras.Model] = {}
    compiled_steps: Dict[int, Any] = {}
    for cid in sorted(clients):
        model = build_full_model(input_dim=feature_dim, learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        local_models[cid] = model
        compiled_steps[cid] = _make_dp_sgd_step(model, optimizer, batch_size)

    round_acc: List[float] = []
    round_loss: List[float] = []
    total_local_steps = 0
    min_client_train_size = min(len(clients[cid]["X_train"]) for cid in clients)

    for round_idx in range(1, rounds + 1):
        local_results: List[Tuple[List[np.ndarray], int]] = []
        for cid in sorted(clients):
            data = clients[cid]
            local_model = local_models[cid]
            local_model.set_weights(global_weights)

            steps = _fit_dp_sgd(
                model=local_model,
                compiled_step=compiled_steps[cid],
                X=data["X_train"],
                y=data["y_train"],
                local_epochs=local_epochs,
                batch_size=batch_size,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                seed=random_state + round_idx + cid,
            )
            if cid == 0:
                total_local_steps += steps  # per-client step count is identical across clients

            local_results.append((local_model.get_weights(), len(data["X_train"])))

        if use_secure_aggregation:
            global_weights = secure_weighted_aggregate(local_results, seed=random_state + round_idx)
        else:
            global_weights = aggregate(local_results)
        global_model.set_weights(global_weights)

        X_val, y_val = split_payload["global_val"]
        val_loss, val_acc = global_model.evaluate(X_val, y_val, verbose=0)
        round_acc.append(float(val_acc))
        round_loss.append(float(val_loss))
        print(f"[DP-SGD Round {round_idx}] noise={noise_multiplier} | Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        if ledger is not None:
            ledger.append_round(round_idx, global_weights, metrics={"val_accuracy": float(val_acc), "val_loss": float(val_loss)})

    test_loss, _ = global_model.evaluate(X_test, y_test, verbose=0)
    y_prob = global_model.predict(X_test, verbose=0).ravel()
    metrics = _compute_metrics(y_test, y_prob, float(test_loss))

    q = batch_size / min_client_train_size
    epsilon, best_order = compute_epsilon(q=q, noise_multiplier=noise_multiplier, steps=total_local_steps, delta=delta)

    ledger_valid = None
    if ledger is not None:
        ledger_valid, _ = ledger.verify_chain()
        if save_metrics:
            ledger.save(AUDIT_LEDGER_PATH)

    result = {
        "noise_multiplier": noise_multiplier,
        "clip_norm": clip_norm,
        "epsilon": epsilon,
        "delta": delta,
        "rdp_order": best_order,
        "sampling_rate_q": q,
        "total_local_steps": total_local_steps,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "use_secure_aggregation": use_secure_aggregation,
        "audit_ledger_valid": ledger_valid,
        "audit_ledger_blocks": len(ledger.chain) if ledger is not None else 0,
        "round_global_accuracy": round_acc,
        "round_global_loss": round_loss,
        **metrics,
    }

    if save_metrics:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        DP_METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nDP-SGD Federated Test Metrics (noise_multiplier={noise_multiplier}, epsilon={epsilon:.3f})")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Loss     : {metrics['loss']:.4f}")
    print(f"(epsilon={epsilon:.3f}, delta={delta}, best RDP order={best_order})")

    return result


if __name__ == "__main__":
    run_dp_federated_pipeline()
