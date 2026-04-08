"""SplitFed pipeline with privacy-safe training and publication-grade plotting."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flwr.server.strategy.aggregate import aggregate
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_privacy_preserving_splits
from src.models import build_full_model, get_client_model, get_server_model

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "plots"
SPLITFED_METRICS_PATH = PROCESSED_DIR / "splitfed_metrics.json"
FINAL_METRICS_PATH = PROCESSED_DIR / "final_metrics.json"
SPLITFED_MODEL_PATH = PROJECT_ROOT / "models" / "splitfed_client_model.h5"
SPLITFED_SERVER_MODEL_PATH = PROJECT_ROOT / "models" / "splitfed_server_model.h5"
CENTRAL_MODEL_PATH = PROJECT_ROOT / "models" / "centralized_model.h5"
FED_MODEL_PATH = PROJECT_ROOT / "models" / "federated_global_model.h5"


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


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


def _evaluate_split_pair(
    client_model: tf.keras.Model,
    server_model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, np.ndarray]:
    y_col = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy()
    activations = client_model(X, training=False)
    probs = server_model(activations, training=False).numpy().ravel()
    loss = float(bce(y_col, probs.reshape(-1, 1)).numpy())
    return loss, probs


def _train_client_split(
    client_model: tf.keras.Model,
    server_model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Tuple[List[np.ndarray], float, float]:
    class_weight_values = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight_map = {int(cls): float(w) for cls, w in zip(np.unique(y_train), class_weight_values)}

    client_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    server_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    bce = tf.keras.losses.BinaryCrossentropy()

    dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
    )

    best_val_loss = float("inf")
    best_weights = client_model.get_weights()
    wait = 0

    for _ in range(local_epochs):
        for x_batch, y_batch in dataset:
            y_batch_col = tf.cast(tf.reshape(y_batch, (-1, 1)), tf.float32)
            sample_weight = tf.constant(
                [class_weight_map[int(v)] for v in y_batch.numpy()],
                dtype=tf.float32,
            )

            with tf.GradientTape() as tape_client_forward:
                activations = client_model(x_batch, training=True)

            with tf.GradientTape(persistent=True) as tape_server:
                tape_server.watch(activations)
                preds = server_model(activations, training=True)
                server_loss = bce(y_batch_col, preds, sample_weight=sample_weight)

            server_grads = tape_server.gradient(server_loss, server_model.trainable_variables)
            grad_activations = tape_server.gradient(server_loss, activations)
            del tape_server
            server_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in server_grads]
            server_optimizer.apply_gradients(zip(server_grads, server_model.trainable_variables))

            with tf.GradientTape() as tape_client_backward:
                activations_2 = client_model(x_batch, training=True)
                surrogate_loss = tf.reduce_sum(activations_2 * tf.stop_gradient(grad_activations))

            client_grads = tape_client_backward.gradient(surrogate_loss, client_model.trainable_variables)
            client_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in client_grads]
            client_optimizer.apply_gradients(zip(client_grads, client_model.trainable_variables))

        val_loss, val_prob = _evaluate_split_pair(client_model, server_model, X_val, y_val)
        val_pred = (val_prob >= 0.5).astype(np.int32)
        val_acc = float(accuracy_score(y_val, val_pred))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = client_model.get_weights()
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait >= 2:
                break

    client_model.set_weights(best_weights)
    return client_model.get_weights(), float(best_val_loss), float(best_val_acc)


def _load_standard_model_predictions(model_path: Path, X: np.ndarray) -> np.ndarray | None:
    if not model_path.exists():
        return None
    model = tf.keras.models.load_model(model_path)
    return model.predict(X, verbose=0).ravel()


def _save_publication_plots(
    centralized_prob: np.ndarray,
    federated_prob: np.ndarray,
    splitfed_prob: np.ndarray,
    y_test: np.ndarray,
    round_acc: List[float],
    all_metrics: Dict[str, Dict[str, float]],
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    labels = ["Centralized", "Federated", "SplitFed"]
    values = [all_metrics["centralized"]["accuracy"], all_metrics["federated"]["accuracy"], all_metrics["splitfed"]["accuracy"]]
    bars = plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.title("Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, min(0.99, v + 0.01), f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()

    rounds = np.arange(1, len(round_acc) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_acc, marker="o", linewidth=2)
    plt.title("SplitFed Rounds vs Validation Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "splitfed_rounds_accuracy.png", dpi=300)
    plt.close()

    y_pred_splitfed = (splitfed_prob >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_test, y_pred_splitfed)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (SplitFed)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    for name, prob in [
        ("Centralized", centralized_prob),
        ("Federated", federated_prob),
        ("SplitFed", splitfed_prob),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, prob in [
        ("Centralized", centralized_prob),
        ("Federated", federated_prob),
        ("SplitFed", splitfed_prob),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()


def run_splitfed_pipeline(
    rounds: int = 20,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run SplitFed with strict test-only final evaluation."""
    _set_global_determinism(random_state)

    split_payload = build_privacy_preserving_splits(random_state=random_state, force_rebuild=False)
    client_data = split_payload["clients"]
    X_val_global, y_val_global = split_payload["global_val"]
    X_test_global, y_test_global = split_payload["global_test"]
    input_dim = int(split_payload["feature_dim"])

    global_client_model = get_client_model(input_dim=input_dim)
    global_server_model = get_server_model(activation_dim=32)
    global_weights = global_client_model.get_weights()

    round_acc: List[float] = []
    round_loss: List[float] = []
    client_accuracy_by_round: Dict[int, Dict[str, float]] = {}

    for round_idx in range(1, rounds + 1):
        local_results: List[Tuple[List[np.ndarray], int]] = []
        per_client_acc: Dict[str, float] = {}

        for cid in sorted(client_data):
            data = client_data[cid]
            local_client = get_client_model(input_dim=input_dim)
            local_client.set_weights(global_weights)

            local_weights, _, val_acc = _train_client_split(
                client_model=local_client,
                server_model=global_server_model,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=random_state + round_idx + cid,
            )

            local_results.append((local_weights, len(data["X_train"])))
            per_client_acc[f"hospital_{cid + 1}"] = float(val_acc)

        global_weights = aggregate(local_results)
        global_client_model.set_weights(global_weights)

        val_loss, val_prob = _evaluate_split_pair(global_client_model, global_server_model, X_val_global, y_val_global)
        val_pred = (val_prob >= 0.5).astype(np.int32)
        val_acc = float(accuracy_score(y_val_global, val_pred))
        round_loss.append(val_loss)
        round_acc.append(val_acc)
        client_accuracy_by_round[round_idx] = per_client_acc

        print(f"[Round {round_idx}] Validation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

    splitfed_test_loss, splitfed_test_prob = _evaluate_split_pair(
        global_client_model,
        global_server_model,
        X_test_global,
        y_test_global,
    )
    splitfed_metrics = _compute_metrics(y_test_global, splitfed_test_prob, splitfed_test_loss)

    centralized_prob = _load_standard_model_predictions(CENTRAL_MODEL_PATH, X_test_global)
    federated_prob = _load_standard_model_predictions(FED_MODEL_PATH, X_test_global)
    if centralized_prob is None or federated_prob is None:
        raise RuntimeError("Centralized or federated model missing. Run those pipelines first.")

    centralized_loss = float(tf.keras.losses.BinaryCrossentropy()(y_test_global.reshape(-1, 1), centralized_prob.reshape(-1, 1)).numpy())
    federated_loss = float(tf.keras.losses.BinaryCrossentropy()(y_test_global.reshape(-1, 1), federated_prob.reshape(-1, 1)).numpy())
    centralized_metrics = _compute_metrics(y_test_global, centralized_prob, centralized_loss)
    federated_metrics = _compute_metrics(y_test_global, federated_prob, federated_loss)

    all_metrics = {
        "centralized": centralized_metrics,
        "federated": federated_metrics,
        "splitfed": splitfed_metrics,
    }

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(round_acc) + 1), round_acc, marker="o", linewidth=2)
    plt.title("SplitFed Rounds vs Validation Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "splitfed_rounds_accuracy.png", dpi=300)
    plt.close()

    _save_publication_plots(
        centralized_prob=centralized_prob,
        federated_prob=federated_prob,
        splitfed_prob=splitfed_test_prob,
        y_test=y_test_global,
        round_acc=round_acc,
        all_metrics=all_metrics,
    )

    SPLITFED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    global_client_model.save(SPLITFED_MODEL_PATH, include_optimizer=False)
    global_server_model.save(SPLITFED_SERVER_MODEL_PATH, include_optimizer=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITFED_METRICS_PATH.write_text(
        json.dumps(
            {
                "round_global_accuracy": round_acc,
                "round_global_loss": round_loss,
                "client_accuracy_by_round": client_accuracy_by_round,
                "final_splitfed_test_accuracy": splitfed_metrics["accuracy"],
                "precision": splitfed_metrics["precision"],
                "recall": splitfed_metrics["recall"],
                "f1_score": splitfed_metrics["f1_score"],
                "roc_auc": splitfed_metrics["roc_auc"],
                "loss": splitfed_metrics["loss"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    FINAL_METRICS_PATH.write_text(
        json.dumps(
            {
                "centralized": {
                    "accuracy": centralized_metrics["accuracy"],
                    "precision": centralized_metrics["precision"],
                    "recall": centralized_metrics["recall"],
                    "f1_score": centralized_metrics["f1_score"],
                    "loss": centralized_metrics["loss"],
                },
                "federated": {
                    "accuracy": federated_metrics["accuracy"],
                    "precision": federated_metrics["precision"],
                    "recall": federated_metrics["recall"],
                    "f1_score": federated_metrics["f1_score"],
                    "loss": federated_metrics["loss"],
                },
                "splitfed": {
                    "accuracy": splitfed_metrics["accuracy"],
                    "precision": splitfed_metrics["precision"],
                    "recall": splitfed_metrics["recall"],
                    "f1_score": splitfed_metrics["f1_score"],
                    "loss": splitfed_metrics["loss"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSplitFed Test Metrics")
    print(f"Accuracy : {splitfed_metrics['accuracy']:.4f}")
    print(f"Precision: {splitfed_metrics['precision']:.4f}")
    print(f"Recall   : {splitfed_metrics['recall']:.4f}")
    print(f"F1-Score : {splitfed_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {splitfed_metrics['roc_auc']:.4f}")
    print(f"Loss     : {splitfed_metrics['loss']:.4f}")

    return {
        "round_global_accuracy": round_acc,
        "round_global_loss": round_loss,
        "splitfed_metrics": splitfed_metrics,
        "all_metrics": all_metrics,
    }


if __name__ == "__main__":
    run_splitfed_pipeline()
