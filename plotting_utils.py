"""Plotting utilities for centralized and federated healthcare experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt


DEFAULT_PLOT_DIR = Path("plots")


def _prepare_output_path(output_path: str) -> Path:
    """Ensure parent directories exist for plot output."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_accuracy_vs_epochs(
    train_accuracy: Sequence[float],
    val_accuracy: Sequence[float] | None = None,
    output_path: str = str(DEFAULT_PLOT_DIR / "accuracy_vs_epochs.png"),
) -> str:
    """Plot and save training/validation accuracy across epochs."""
    save_path = _prepare_output_path(output_path)
    epochs = list(range(1, len(train_accuracy) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracy, label="Train Accuracy", linewidth=2)
    if val_accuracy is not None and len(val_accuracy) > 0:
        plt.plot(epochs, val_accuracy, label="Validation Accuracy", linewidth=2)

    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return str(save_path)


def plot_loss_vs_epochs(
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    output_path: str = str(DEFAULT_PLOT_DIR / "loss_vs_epochs.png"),
) -> str:
    """Plot and save training/validation loss across epochs."""
    save_path = _prepare_output_path(output_path)
    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    if val_loss is not None and len(val_loss) > 0:
        plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)

    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return str(save_path)


def plot_federated_rounds_vs_global_accuracy(
    global_accuracy_by_round: Dict[int, float] | Sequence[float],
    output_path: str = str(DEFAULT_PLOT_DIR / "federated_rounds_vs_global_accuracy.png"),
) -> str:
    """Plot and save federated rounds against global model accuracy."""
    save_path = _prepare_output_path(output_path)

    if isinstance(global_accuracy_by_round, dict):
        rounds = sorted(global_accuracy_by_round.keys())
        accuracy = [global_accuracy_by_round[r] for r in rounds]
    else:
        accuracy = list(global_accuracy_by_round)
        rounds = list(range(1, len(accuracy) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracy, marker="o", linewidth=2)
    plt.title("Federated Rounds vs Global Accuracy")
    plt.xlabel("Federated Round")
    plt.ylabel("Global Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return str(save_path)


def plot_hospital_wise_accuracy(
    hospital_accuracy: Dict[str, float] | Iterable[tuple[str, float]],
    output_path: str = str(DEFAULT_PLOT_DIR / "hospital_wise_accuracy.png"),
) -> str:
    """Plot and save hospital-wise accuracy comparison as a bar chart."""
    save_path = _prepare_output_path(output_path)

    if isinstance(hospital_accuracy, dict):
        items = list(hospital_accuracy.items())
    else:
        items = list(hospital_accuracy)

    if not items:
        raise ValueError("hospital_accuracy cannot be empty.")

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values)
    plt.title("Hospital-wise Accuracy Comparison")
    plt.xlabel("Hospital")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.99, value + 0.01),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return str(save_path)
