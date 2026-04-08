"""Centralized TensorFlow/Keras binary classifier for healthcare tabular data."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def _to_binary_labels(y: np.ndarray) -> np.ndarray:
    """Convert labels to binary targets for disease/no-disease classification."""
    y = np.asarray(y)
    unique = np.unique(y)

    if set(unique.tolist()).issubset({0, 1}):
        return y.astype(np.int32)

    # Common heart-disease convention: 0 means no disease, >0 means disease present.
    return (y > 0).astype(np.int32)


def _build_model(input_dim: int) -> tf.keras.Model:
    """Build the required Keras binary classification architecture."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _plot_history(history: tf.keras.callbacks.History) -> None:
    """Plot training vs validation accuracy and loss curves."""
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(hist.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(hist.get("val_accuracy", []), label="Val Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(hist.get("loss", []), label="Train Loss")
    axes[1].plot(hist.get("val_loss", []), label="Val Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def train_centralized_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, float]]:
    """Train and evaluate a centralized binary classification model.

    Args:
        X: Feature matrix.
        y: Target labels.
        epochs: Number of training epochs.
        batch_size: Batch size for model training.
        test_size: Validation/test split ratio.
        random_state: Random seed for split reproducibility.

    Returns:
        A tuple of (trained model, training history, metrics dict).
    """
    X = np.asarray(X, dtype=np.float32)
    y_binary = _to_binary_labels(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if y_binary.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if X.shape[0] != y_binary.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary,
    )

    model = _build_model(input_dim=X.shape[1])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")

    _plot_history(history)

    return model, history, metrics
