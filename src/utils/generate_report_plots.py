"""Regenerate every report figure from the current models and saved metrics.

All plots in ``plots/`` are rebuilt from a single source of truth (the trained models
in ``models/`` plus the metric JSON files in ``data/processed/``) so they always match
the latest evaluation scores. Both the canonical figure names and the earlier legacy
names are written, so no stale graph is left behind.

Run:  python src/utils/generate_report_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_privacy_preserving_splits
from src.models import build_full_model

PROCESSED = PROJECT_ROOT / "data" / "processed"
PLOTS = PROJECT_ROOT / "plots"
MODELS = PROJECT_ROOT / "models"

COLORS = {"Centralized": "#2563eb", "Federated": "#059669", "SplitFed": "#d97706"}
RANDOM_STATE = 42


def _load_json(name: str) -> dict:
    return json.loads((PROCESSED / name).read_text(encoding="utf-8"))


def _save(fig_paths: list[Path]) -> None:
    """Save the current figure to one or more paths (canonical + legacy aliases)."""
    for p in fig_paths:
        plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()


def _predict_probs(splits: dict) -> dict:
    """Return held-out test probabilities for each method from the saved models."""
    X_test, y_test = splits["global_test"]

    centralized = tf.keras.models.load_model(MODELS / "centralized_model.h5")
    federated = tf.keras.models.load_model(MODELS / "federated_global_model.h5")
    sf_client = tf.keras.models.load_model(MODELS / "splitfed_client_model.h5")
    sf_server = tf.keras.models.load_model(MODELS / "splitfed_server_model.h5")

    sf_prob = sf_server(sf_client(X_test, training=False), training=False).numpy().ravel()
    return {
        "y_test": y_test,
        "Centralized": centralized.predict(X_test, verbose=0).ravel(),
        "Federated": federated.predict(X_test, verbose=0).ravel(),
        "SplitFed": sf_prob,
    }


def _centralized_history(splits: dict) -> tf.keras.callbacks.History:
    """Deterministically re-fit the centralized model to recover per-epoch curves."""
    from sklearn.utils.class_weight import compute_class_weight

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    X_train, y_train = splits["global_train"]
    X_val, y_val = splits["global_val"]
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_full_model(input_dim=X_train.shape[1], learning_rate=3e-4)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=0),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=120, batch_size=32, class_weight=class_weight, verbose=0, callbacks=callbacks,
    )


def plot_accuracy_comparison(final: dict) -> None:
    names = ["Centralized", "Federated", "SplitFed"]
    keys = ["centralized", "federated", "splitfed"]
    values = [final[k]["accuracy"] for k in keys]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=[COLORS[n] for n in names])
    plt.ylim(0.0, 1.0)
    plt.title("Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.4f}", ha="center", fontweight="bold")
    _save([PLOTS / "accuracy_comparison.png", PLOTS / "accuracy_comparison_splitfed.png"])


def plot_metrics_grouped(final: dict, aucs: dict) -> None:
    names = ["Centralized", "Federated", "SplitFed"]
    keys = ["centralized", "federated", "splitfed"]
    metrics = ["accuracy", "f1_score", "roc_auc"]
    labels = ["Accuracy", "F1 Score", "ROC-AUC"]
    x = np.arange(len(labels))
    width = 0.26

    plt.figure(figsize=(9, 5))
    for i, (name, key) in enumerate(zip(names, keys)):
        vals = [final[key]["accuracy"], final[key]["f1_score"], aucs[key]]
        plt.bar(x + (i - 1) * width, vals, width, label=name, color=COLORS[name])
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Metric Comparison Across Learning Paradigms")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _save([PLOTS / "metrics_comparison.png"])


def plot_centralized_vs_federated(final: dict) -> None:
    names = ["Centralized", "Federated"]
    values = [final["centralized"]["accuracy"], final["federated"]["accuracy"]]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(names, values, color=[COLORS[n] for n in names])
    plt.ylim(0.0, 1.0)
    plt.title("Centralized vs Federated Accuracy")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.4f}", ha="center", fontweight="bold")
    _save([PLOTS / "centralized_vs_federated_accuracy.png"])


def plot_roc(probs: dict) -> None:
    y = probs["y_test"]
    plt.figure(figsize=(8, 5))
    for name in ["Centralized", "Federated", "SplitFed"]:
        fpr, tpr, _ = roc_curve(y, probs[name])
        plt.plot(fpr, tpr, linewidth=2, color=COLORS[name], label=f"{name} (AUC={roc_auc_score(y, probs[name]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    _save([PLOTS / "roc_curve.png"])


def plot_pr(probs: dict) -> None:
    y = probs["y_test"]
    plt.figure(figsize=(8, 5))
    for name in ["Centralized", "Federated", "SplitFed"]:
        prec, rec, _ = precision_recall_curve(y, probs[name])
        ap = average_precision_score(y, probs[name])
        plt.plot(rec, prec, linewidth=2, color=COLORS[name], label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    _save([PLOTS / "precision_recall_curve.png"])


def plot_confusion(probs: dict) -> None:
    y = probs["y_test"]
    y_pred = (probs["SplitFed"] >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (SplitFed, held-out test)")
    fig.tight_layout()
    for p in [PLOTS / "confusion_matrix.png"]:
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rounds(history_key: str, title: str, color: str, out_names: list[str]) -> None:
    data = _load_json(history_key)
    acc = data.get("round_global_accuracy", [])
    loss = data.get("round_global_loss", [])
    rounds = np.arange(1, len(acc) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rounds, acc, marker="o", linewidth=2, color=color, label="Validation Accuracy")
    ax1.set_xlabel("Aggregation Round")
    ax1.set_ylabel("Accuracy", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(alpha=0.3)
    if loss:
        ax2 = ax1.twinx()
        ax2.plot(rounds, loss, marker="s", linewidth=1.5, color="#9ca3af", alpha=0.8, label="Validation Loss")
        ax2.set_ylabel("Loss", color="#6b7280")
        ax2.tick_params(axis="y", labelcolor="#6b7280")
    plt.title(title)
    fig.tight_layout()
    for p in out_names:
        fig.savefig(PLOTS / p, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_centralized_curves(history: tf.keras.callbacks.History) -> None:
    h = history.history
    epochs = np.arange(1, len(h.get("loss", [])) + 1)

    # Loss vs epochs
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, h.get("loss", []), linewidth=2, label="Train Loss", color="#2563eb")
    if h.get("val_loss"):
        plt.plot(epochs, h["val_loss"], linewidth=2, label="Validation Loss", color="#d97706")
    plt.title("Centralized Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    _save([PLOTS / "loss_curve.png", PLOTS / "centralized_loss_vs_epochs.png"])

    # Accuracy vs epochs
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, h.get("accuracy", []), linewidth=2, label="Train Accuracy", color="#2563eb")
    if h.get("val_accuracy"):
        plt.plot(epochs, h["val_accuracy"], linewidth=2, label="Validation Accuracy", color="#d97706")
    plt.title("Centralized Training Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    _save([PLOTS / "centralized_accuracy_vs_epochs.png"])


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    final = _load_json("final_metrics.json")
    aucs = {
        "centralized": _load_json("metrics.json")["roc_auc"],
        "federated": _load_json("fl_metrics.json")["roc_auc"],
        "splitfed": _load_json("splitfed_metrics.json")["roc_auc"],
    }

    splits = build_privacy_preserving_splits(random_state=RANDOM_STATE, force_rebuild=False)
    probs = _predict_probs(splits)

    plot_accuracy_comparison(final)
    plot_metrics_grouped(final, aucs)
    plot_centralized_vs_federated(final)
    plot_roc(probs)
    plot_pr(probs)
    plot_confusion(probs)
    plot_rounds("fl_metrics.json", "Federated Rounds vs Validation Accuracy", COLORS["Federated"],
                ["fl_rounds_accuracy.png", "fl_rounds_vs_accuracy.png"])
    plot_rounds("splitfed_metrics.json", "SplitFed Rounds vs Validation Accuracy", COLORS["SplitFed"],
                ["splitfed_rounds_accuracy.png", "splitfed_rounds_vs_accuracy.png"])
    plot_centralized_curves(_centralized_history(splits))

    print("Regenerated all report figures in", PLOTS)


if __name__ == "__main__":
    main()
