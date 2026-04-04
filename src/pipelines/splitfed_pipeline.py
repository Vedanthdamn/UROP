"""SplitFed pipeline for healthcare binary classification."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.server.strategy.aggregate import aggregate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client_model import get_client_model
from data_expansion import expand_dataset
from hospital_split import create_hospital_splits
from preprocessing import load_and_preprocess
from server_model import get_server_model

CLIENT_DIR = PROJECT_ROOT / "clients"
SPLITFED_CLIENT_DIR = CLIENT_DIR / "splitfed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "plots"
SPLITFED_METRICS_PATH = PROCESSED_DIR / "splitfed_metrics.json"
SPLITFED_MODEL_PATH = PROJECT_ROOT / "models" / "splitfed_client_model.h5"
SPLITFED_TRAIN_EXPANDED_PATH = PROCESSED_DIR / "splitfed_train_expanded.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"


def _resolve_target_column(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    if "num" in df.columns:
        return "num"
    raise ValueError("Target column not found. Expected one of: target, num.")


def _to_binary_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    unique = np.unique(y)
    if set(unique.tolist()).issubset({0, 1}):
        return y.astype(np.int32)
    return (y > 0).astype(np.int32)


def _build_splitfed_datasets(random_state: int) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    # Global split before any training/augmentation to keep a fully untouched test set.
    X_raw, y_raw = load_and_preprocess(csv_path=str(RAW_DATA_PATH))
    y_raw = _to_binary_labels(y_raw)
    X_train_base, X_test_global, y_train_base, y_test_global = _safe_train_test_split(
        X=X_raw,
        y=y_raw,
        test_size=0.2,
        random_state=random_state,
    )

    X_train_expanded, y_train_expanded = expand_dataset(
        X_train_base,
        y_train_base,
        target_size=12000,
        random_state=random_state,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"feature_{i}" for i in range(X_train_expanded.shape[1])]
    splitfed_train_df = pd.DataFrame(X_train_expanded, columns=feature_cols)
    splitfed_train_df["target"] = y_train_expanded
    splitfed_train_df.to_csv(SPLITFED_TRAIN_EXPANDED_PATH, index=False)

    SPLITFED_CLIENT_DIR.mkdir(parents=True, exist_ok=True)
    create_hospital_splits(
        input_path=str(SPLITFED_TRAIN_EXPANDED_PATH),
        output_dir=str(SPLITFED_CLIENT_DIR),
        total_samples=len(splitfed_train_df),
        random_state=random_state,
    )

    hospital_paths = [SPLITFED_CLIENT_DIR / f"hospital_{i}.csv" for i in range(1, 6)]
    if not all(path.exists() for path in hospital_paths):
        raise RuntimeError("Failed to create all SplitFed hospital datasets.")

    return hospital_paths, X_test_global.astype(np.float32), y_test_global.astype(np.int32)


def _load_hospital_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty hospital dataset: {path}")

    target_col = _resolve_target_column(df)
    y = _to_binary_labels(df[target_col].to_numpy())
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    return X, y


def _safe_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) > 1 and counts.min() >= 2
    stratify = y if can_stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _group_preserving_client_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep duplicate samples in a single partition to avoid bootstrap leakage across train/test.
    frame = pd.DataFrame(X)
    frame["_target"] = y
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()

    if len(np.unique(groups)) < 2:
        return _safe_train_test_split(X=X, y=y, test_size=test_size, random_state=random_state)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    y_train = y[train_idx]
    y_test = y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return _safe_train_test_split(X=X, y=y, test_size=test_size, random_state=random_state)

    return X[train_idx], X[test_idx], y_train, y_test


def _prepare_clients(hospital_paths: List[Path], random_state: int) -> Dict[int, Dict[str, np.ndarray]]:
    client_data: Dict[int, Dict[str, np.ndarray]] = {}
    for i, path in enumerate(hospital_paths):
        X, y = _load_hospital_data(path)
        # Mandatory 80/20 client-side split: training partition and held-out test partition.
        X_train, X_test, y_train, y_test = _group_preserving_client_split(
            X=X,
            y=y,
            test_size=0.2,
            random_state=random_state + i,
        )
        client_data[i] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    return client_data


def _evaluate_pair(
    client_model: tf.keras.Model,
    server_model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    bce = tf.keras.losses.BinaryCrossentropy()
    y_col = y.reshape(-1, 1).astype(np.float32)
    activations = client_model(X, training=False)
    preds = server_model(activations, training=False)
    loss = float(bce(y_col, preds).numpy())
    y_pred = (preds.numpy().ravel() >= 0.5).astype(np.int32)
    acc = float(accuracy_score(y, y_pred))
    return loss, acc


def _train_client_split(
    client_model: tf.keras.Model,
    server_model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    local_val_fraction: float = 0.1,
    early_stop_patience: int = 2,
) -> Tuple[List[np.ndarray], float, float]:
    bce = tf.keras.losses.BinaryCrossentropy()
    client_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    server_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Use only the training partition for local model selection; never touch client test data here.
    X_local_train, X_local_val, y_local_train, y_local_val = _safe_train_test_split(
        X=X_train,
        y=y_train,
        test_size=local_val_fraction,
        random_state=seed,
    )

    dataset = (
        tf.data.Dataset.from_tensor_slices((X_local_train, y_local_train))
        .shuffle(buffer_size=len(X_local_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
    )

    best_val_loss = float("inf")
    wait = 0

    for _ in range(local_epochs):
        for x_batch, y_batch in dataset:
            y_batch = tf.cast(tf.reshape(y_batch, (-1, 1)), tf.float32)

            with tf.GradientTape() as tape_client_forward:
                activations = client_model(x_batch, training=True)

            with tf.GradientTape(persistent=True) as tape_server:
                tape_server.watch(activations)
                preds = server_model(activations, training=True)
                server_loss = bce(y_batch, preds)

            server_grads = tape_server.gradient(server_loss, server_model.trainable_variables)
            grad_activations = tape_server.gradient(server_loss, activations)
            del tape_server

            server_optimizer.apply_gradients(zip(server_grads, server_model.trainable_variables))

            with tf.GradientTape() as tape_client_backward:
                activations_for_backward = client_model(x_batch, training=True)
                surrogate_loss = tf.reduce_sum(activations_for_backward * tf.stop_gradient(grad_activations))

            client_grads = tape_client_backward.gradient(surrogate_loss, client_model.trainable_variables)
            client_optimizer.apply_gradients(zip(client_grads, client_model.trainable_variables))

        val_loss, _ = _evaluate_pair(client_model, server_model, X_local_val, y_local_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                break

    final_val_loss, final_val_acc = _evaluate_pair(client_model, server_model, X_local_val, y_local_val)
    return client_model.get_weights(), final_val_loss, final_val_acc


def _load_baseline_accuracies() -> Tuple[float | None, float | None]:
    centralized_acc = None
    federated_acc = None

    metrics_path = PROCESSED_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            centralized_acc = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("accuracy"))
        except (TypeError, ValueError):
            centralized_acc = None

    fl_metrics_path = PROCESSED_DIR / "fl_metrics.json"
    if fl_metrics_path.exists():
        try:
            federated_acc = float(json.loads(fl_metrics_path.read_text(encoding="utf-8")).get("final_federated_accuracy"))
        except (TypeError, ValueError):
            federated_acc = None

    return centralized_acc, federated_acc


def _save_outputs(
    round_accuracies: List[float],
    round_losses: List[float],
    client_acc_by_round: Dict[int, Dict[str, float]],
    splitfed_train_acc: float,
    splitfed_test_acc: float,
    splitfed_test_loss: float,
    centralized_acc: float | None,
    federated_acc: float | None,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "round_global_accuracy": round_accuracies,
        "round_global_loss": round_losses,
        "client_accuracy_by_round": client_acc_by_round,
        "final_splitfed_accuracy": splitfed_test_acc,
        "final_splitfed_train_accuracy": splitfed_train_acc,
        "final_splitfed_test_accuracy": splitfed_test_acc,
        "final_splitfed_test_loss": splitfed_test_loss,
        "centralized_accuracy": centralized_acc,
        "federated_accuracy": federated_acc,
    }
    SPLITFED_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    rounds = np.arange(1, len(round_accuracies) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_accuracies, marker="o", linewidth=2)
    plt.title("SplitFed Rounds vs Global Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "splitfed_rounds_vs_accuracy.png", dpi=200)
    plt.close()

    labels = ["SplitFed"]
    values = [splitfed_test_acc]
    if centralized_acc is not None and federated_acc is not None:
        labels = ["Centralized", "Federated", "SplitFed"]
        values = [centralized_acc, federated_acc, splitfed_test_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.title("Accuracy Comparison: Centralized vs FL vs SplitFed")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, min(0.99, val + 0.01), f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_comparison_splitfed.png", dpi=200)
    plt.close()


def run_splitfed_pipeline(
    rounds: int = 20,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run SplitFed training with FedAvg aggregation of client-side weights."""
    _ = fl.__version__  # Explicit Flower dependency usage.

    hospital_paths, global_test_X, global_test_y = _build_splitfed_datasets(random_state=random_state)
    client_data = _prepare_clients(hospital_paths, random_state=random_state)

    input_dim = next(iter(client_data.values()))["X_train"].shape[1]
    global_client_model = get_client_model(input_dim=input_dim)
    global_client_weights = global_client_model.get_weights()

    global_server_model = get_server_model(activation_dim=32)
    global_server_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    all_train_X = np.concatenate([client_data[cid]["X_train"] for cid in sorted(client_data)], axis=0)
    all_train_y = np.concatenate([client_data[cid]["y_train"] for cid in sorted(client_data)], axis=0)

    round_global_accuracy: List[float] = []
    round_global_loss: List[float] = []
    client_accuracy_by_round: Dict[int, Dict[str, float]] = {}

    loss_increase_streak = 0

    for round_idx in range(1, rounds + 1):
        client_results: List[Tuple[List[np.ndarray], int]] = []
        round_client_acc: Dict[str, float] = {}

        for cid in sorted(client_data):
            data = client_data[cid]

            local_client_model = get_client_model(input_dim=input_dim)
            local_client_model.set_weights(global_client_weights)

            local_weights, _, local_train_proxy_acc = _train_client_split(
                client_model=local_client_model,
                server_model=global_server_model,
                X_train=data["X_train"],
                y_train=data["y_train"],
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=random_state + round_idx + cid,
                local_val_fraction=0.1,
                early_stop_patience=2,
            )

            client_results.append((local_weights, len(data["X_train"])))
            round_client_acc[f"hospital_{cid + 1}"] = float(local_train_proxy_acc)

        global_client_weights = aggregate(client_results)
        global_client_model.set_weights(global_client_weights)

        # Mandatory: evaluate only on held-out test data after each communication round.
        global_loss, global_acc = _evaluate_pair(global_client_model, global_server_model, global_test_X, global_test_y)
        round_global_loss.append(global_loss)
        round_global_accuracy.append(global_acc)
        client_accuracy_by_round[round_idx] = round_client_acc

        print(f"[Round {round_idx}] SplitFed Global Accuracy: {global_acc:.4f} | Loss: {global_loss:.4f}")

        if len(round_global_loss) >= 2 and round_global_loss[-1] > round_global_loss[-2]:
            loss_increase_streak += 1
        else:
            loss_increase_streak = 0

        if loss_increase_streak >= 5:
            print("[WARN] Loss increased for 5 consecutive rounds. Stopping early for stability.")
            break

    final_train_loss, final_train_acc = _evaluate_pair(
        global_client_model,
        global_server_model,
        all_train_X,
        all_train_y,
    )

    final_splitfed_loss, final_splitfed_acc = _evaluate_pair(
        global_client_model,
        global_server_model,
        global_test_X,
        global_test_y,
    )

    centralized_acc, federated_acc = _load_baseline_accuracies()

    SPLITFED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    global_client_model.save(SPLITFED_MODEL_PATH, include_optimizer=False)

    _save_outputs(
        round_accuracies=round_global_accuracy,
        round_losses=round_global_loss,
        client_acc_by_round=client_accuracy_by_round,
        splitfed_train_acc=final_train_acc,
        splitfed_test_acc=final_splitfed_acc,
        splitfed_test_loss=final_splitfed_loss,
        centralized_acc=centralized_acc,
        federated_acc=federated_acc,
    )

    print("\nFinal SplitFed Metrics")
    print(f"SplitFed Train Accuracy      : {final_train_acc:.4f}")
    print(f"Final SplitFed Test Accuracy : {final_splitfed_acc:.4f}")
    print(f"SplitFed Final Test Loss     : {final_splitfed_loss:.4f}")
    if centralized_acc is not None:
        print(f"Centralized Accuracy: {centralized_acc:.4f}")
    else:
        print("Centralized Accuracy: unavailable")
    if federated_acc is not None:
        print(f"Federated Accuracy  : {federated_acc:.4f}")
    else:
        print("Federated Accuracy  : unavailable")

    return {
        "splitfed_train_accuracy": final_train_acc,
        "splitfed_test_accuracy": final_splitfed_acc,
        "splitfed_test_loss": final_splitfed_loss,
        "splitfed_train_loss": final_train_loss,
        "centralized_accuracy": centralized_acc,
        "federated_accuracy": federated_acc,
        "round_global_accuracy": round_global_accuracy,
        "round_global_loss": round_global_loss,
        "client_accuracy_by_round": client_accuracy_by_round,
    }


if __name__ == "__main__":
    run_splitfed_pipeline(rounds=20, local_epochs=3, batch_size=32, learning_rate=3e-4)
