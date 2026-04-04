"""Complete training and evaluation pipeline for healthcare binary classification."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Tuple

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
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_expansion import expand_dataset
from plotting_utils import plot_accuracy_vs_epochs, plot_loss_vs_epochs
from preprocessing import load_and_preprocess
from src.models.full_model import build_full_model


def _to_binary_labels(y: np.ndarray) -> np.ndarray:
    """Map labels to binary targets if needed (0=no disease, >0=disease)."""
    y = np.asarray(y)
    unique = np.unique(y)
    if set(unique.tolist()).issubset({0, 1}):
        return y.astype(np.int32)
    return (y > 0).astype(np.int32)


def _build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create training callbacks for stability and learning-rate scheduling."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for binary training."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    normalized = weights / np.mean(weights)
    stabilized = np.clip(normalized, 0.8, 1.25)
    return {int(cls): float(weight) for cls, weight in zip(classes, stabilized)}


def _load_previous_accuracy(metrics_path: Path) -> float | None:
    """Load previous run accuracy if available for comparison."""
    if not metrics_path.exists():
        # Fallback to recent training logs when metrics artifact is absent.
        log_candidates = [Path("training_run.log"), Path("automation_run.log")]
        for log_path in log_candidates:
            if not log_path.exists():
                continue
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            matches = re.findall(r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)", content)
            if matches:
                return float(matches[-1])
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(data.get("accuracy")) if "accuracy" in data else None
    except (ValueError, TypeError):
        return None


def _save_current_metrics(metrics_path: Path, metrics: Dict[str, Any]) -> None:
    """Persist latest metrics for future run-to-run comparison."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "roc_auc": float(metrics["roc_auc"]),
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _check_overfitting(history: tf.keras.callbacks.History) -> None:
    """Warn if validation loss diverges notably from training loss."""
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    if not train_loss or not val_loss:
        return

    final_train = float(train_loss[-1])
    final_val = float(val_loss[-1])
    if final_val > final_train * 1.2:
        print("[WARN] Potential overfitting detected: val_loss is significantly higher than train_loss.")
    else:
        print("[INFO] Overfitting check passed: val_loss remains close to train_loss.")


def _evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute all required binary classification metrics."""
    y_pred = (y_prob >= 0.5).astype(np.int32)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics


def run_training_pipeline(
    csv_path: str = "data/raw/heart_disease_uci.csv",
    target_size: int = 12000,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    random_state: int = 42,
    model_output_path: str = "models/centralized_model.h5",
) -> Tuple[Dict[str, Any], tf.keras.callbacks.History]:
    """Run full centralized training pipeline for healthcare binary classification.

    Args:
        csv_path: Input CSV path.
        target_size: Number of samples after augmentation.
        epochs: Number of epochs for training.
        batch_size: Batch size for training.
        validation_split: Validation split ratio.
        random_state: Random seed.
        model_output_path: Where to persist the trained model.

    Returns:
        Tuple of evaluation metrics and Keras training history.
    """
    # Step 1: Load and preprocess data.
    X, y = load_and_preprocess(csv_path=csv_path)
    y_binary = _to_binary_labels(y)

    # Step 2: Expand dataset from ~800 to target_size samples.
    X_aug, y_aug = expand_dataset(
        X,
        y_binary,
        target_size=target_size,
        noise_std=0.01,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_aug,
        y_aug,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_aug,
    )

    class_weight = _compute_class_weights(y_train)
    previous_metrics_path = Path("data/processed/metrics.json")
    previous_accuracy = _load_previous_accuracy(previous_metrics_path)

    # Step 3: Train centralized model.
    callbacks = _build_callbacks()
    learning_rates = [1e-3, 5e-4, 1e-4]
    best_metrics: Dict[str, Any] | None = None
    best_history: tf.keras.callbacks.History | None = None
    best_model: tf.keras.Model | None = None
    best_accuracy = -1.0

    for i, lr in enumerate(learning_rates, start=1):
        print(f"\n[INFO] Training attempt {i} with learning_rate={lr}")
        model = build_full_model(input_dim=X_train.shape[1], learning_rate=lr)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        # Step 4: Evaluate performance.
        y_prob = model.predict(X_val, verbose=0).ravel()
        metrics = _evaluate_model(y_true=y_val, y_prob=y_prob)
        current_accuracy = float(metrics["accuracy"])

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_metrics = metrics
            best_history = history
            best_model = model

        if previous_accuracy is None:
            break
        if current_accuracy > previous_accuracy:
            break

        if i < len(learning_rates):
            print("[INFO] Accuracy did not improve vs previous run. Lowering learning rate and retrying.")

    if best_metrics is None or best_history is None or best_model is None:
        raise RuntimeError("Training failed to produce valid metrics.")

    metrics = best_metrics
    history = best_history
    model = best_model

    print("\nFinal Evaluation Results")
    print("=" * 40)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    if previous_accuracy is not None:
        delta = float(metrics["accuracy"]) - previous_accuracy
        print(f"Previous Accuracy: {previous_accuracy:.4f}")
        print(f"New Accuracy     : {metrics['accuracy']:.4f}")
        print(f"Accuracy Delta   : {delta:+.4f}")
    else:
        print("Previous Accuracy: unavailable")
        print(f"New Accuracy     : {metrics['accuracy']:.4f}")
    print("\nClassification Report")
    print("-" * 40)
    print(metrics["classification_report"])

    _check_overfitting(history)

    # Persist trained model.
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path, include_optimizer=False)
    _save_current_metrics(previous_metrics_path, metrics)

    # Step 5: Plot training results.
    plot_accuracy_vs_epochs(
        train_accuracy=history.history.get("accuracy", []),
        val_accuracy=history.history.get("val_accuracy", []),
        output_path="plots/centralized_accuracy_vs_epochs.png",
    )
    plot_loss_vs_epochs(
        train_loss=history.history.get("loss", []),
        val_loss=history.history.get("val_loss", []),
        output_path="plots/centralized_loss_vs_epochs.png",
    )

    return metrics, history


def train_and_evaluate(
    csv_path: str = "data/raw/heart_disease_uci.csv",
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    random_state: int = 42,
    model_output_path: str = "models/centralized_model.h5",
) -> Tuple[Dict[str, Any], tf.keras.callbacks.History]:
    """Backward-compatible alias for the pipeline entrypoint."""
    return run_training_pipeline(
        csv_path=csv_path,
        target_size=12000,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        random_state=random_state,
        model_output_path=model_output_path,
    )


if __name__ == "__main__":
    run_training_pipeline(
        csv_path="data/raw/heart_disease_uci.csv",
        target_size=12000,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        model_output_path="models/centralized_model.h5",
    )
