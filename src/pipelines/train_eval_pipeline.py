"""Centralized training and strict test evaluation pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_privacy_preserving_splits
from src.models import build_full_model


def _set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, loss: float) -> Dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "loss": float(loss),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "y_prob": y_prob.tolist(),
        "y_true": y_true.tolist(),
    }


def _save_loss_curve(history: tf.keras.callbacks.History) -> None:
    Path("plots").mkdir(parents=True, exist_ok=True)
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, linewidth=2, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_loss, linewidth=2, label="Validation Loss")
    plt.title("Centralized Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/loss_curve.png", dpi=300)
    plt.close()


def run_training_pipeline(
    csv_path: str = "data/raw/heart_disease_uci.csv",
    target_size: int = 12000,
    epochs: int = 50,
    batch_size: int = 32,
    random_state: int = 42,
    learning_rate: float = 2e-4,
    model_output_path: str = "models/centralized_model.h5",
) -> Tuple[Dict[str, Any], tf.keras.callbacks.History]:
    """Train centralized model and evaluate strictly on unseen test split."""
    _set_global_determinism(random_state)

    split_payload = build_privacy_preserving_splits(
        csv_path=str(PROJECT_ROOT / csv_path),
        target_size=target_size,
        test_size=0.2,
        val_size_from_train=0.1,
        random_state=random_state,
        force_rebuild=False,
    )

    X_train, y_train = split_payload["global_train"]
    X_val, y_val = split_payload["global_val"]
    X_test, y_test = split_payload["global_test"]

    class_weight = _compute_class_weights(y_train)

    model = build_full_model(input_dim=X_train.shape[1], learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks,
    )

    test_loss, _ = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).ravel()
    metrics = _evaluate(y_true=y_test, y_prob=y_prob, loss=float(test_loss))

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path, include_optimizer=False)

    metrics_path = Path("data/processed/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "loss": metrics["loss"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_loss_curve(history)

    print("\nCentralized Test Metrics")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Loss     : {metrics['loss']:.4f}")

    return metrics, history


if __name__ == "__main__":
    run_training_pipeline()
