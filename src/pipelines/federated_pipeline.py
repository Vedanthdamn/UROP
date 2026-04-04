"""End-to-end Federated Learning pipeline for healthcare binary classification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hospital_split import create_hospital_splits
from data_expansion import expand_dataset
from preprocessing import load_and_preprocess
from src.models.full_model import build_full_model

CLIENT_DIR = PROJECT_ROOT / "clients"
FL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "fl_metrics.json"
GLOBAL_MODEL_PATH = PROJECT_ROOT / "models" / "federated_global_model.h5"
ROUND_PLOT_PATH = PROJECT_ROOT / "plots" / "fl_rounds_vs_accuracy.png"
COMPARE_PLOT_PATH = PROJECT_ROOT / "plots" / "centralized_vs_federated_accuracy.png"


EARLY_STOP_MARKER = "EARLY_STOP_LOSS_TREND"


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


def _ensure_hospital_datasets() -> List[Path]:
    """Ensure 5 hospital CSVs exist; generate if needed."""
    expected = [CLIENT_DIR / f"hospital_{i}.csv" for i in range(1, 6)]
    if all(path.exists() for path in expected):
        return expected

    expanded_path = PROJECT_ROOT / "data" / "processed" / "expanded_12000.csv"
    if not expanded_path.exists():
        print("[INFO] Expanded dataset not found. Building expanded_12000.csv...")
        X, y = load_and_preprocess(csv_path=str(PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"))
        expand_dataset(X, y, target_size=12000)

    print("[INFO] Hospital datasets not found. Generating non-IID client splits...")
    create_hospital_splits(input_path=str(expanded_path), output_dir=str(CLIENT_DIR))
    if not all(path.exists() for path in expected):
        raise RuntimeError("Failed to create all hospital datasets under clients/.")
    return expected


def _load_hospital_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty client dataset: {path}")

    target_col = _resolve_target_column(df)
    y = _to_binary_labels(df[target_col].to_numpy())
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    return X, y


def _weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate global accuracy across clients by number of examples."""
    if not metrics:
        return {}

    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}

    weighted_acc = 0.0
    for num_examples, m in metrics:
        weighted_acc += num_examples * float(m.get("accuracy", 0.0))
    return {"accuracy": weighted_acc / total}


def _shuffle_and_balance_client_data(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle and rebalance binary client data to reduce heterogeneity impact."""
    rng = np.random.default_rng(random_state)
    y = _to_binary_labels(y)

    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]

    if len(class0_idx) == 0 or len(class1_idx) == 0:
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    target_per_class = min(len(class0_idx), len(class1_idx))
    keep0 = rng.choice(class0_idx, size=target_per_class, replace=False)
    keep1 = rng.choice(class1_idx, size=target_per_class, replace=False)
    keep = np.concatenate([keep0, keep1])
    rng.shuffle(keep)
    return X[keep], y[keep]


class HealthcareFLClient(fl.client.NumPyClient):
    """Flower NumPyClient for one hospital dataset."""

    def __init__(
        self,
        client_id: int,
        csv_path: Path,
        local_epochs: int,
        batch_size: int,
        random_state: int,
        learning_rate: float,
        mu: float,
        early_stop_patience: int = 3,
    ) -> None:
        try:
            # Multi-process FL client actors on macOS/Metal can be unstable on GPU.
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
        self.early_stop_patience = early_stop_patience
        self.random_state = random_state

        X, y = _load_hospital_csv(csv_path)
        X, y = _shuffle_and_balance_client_data(X, y, random_state=random_state + client_id)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )
        self.model = build_full_model(input_dim=self.X_train.shape[1], learning_rate=self.learning_rate)

    def _local_train_fedprox(self, local_epochs: int, batch_size: int) -> None:
        """Train client model with FedProx objective and local early stopping."""
        dataset = (
            tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
            .shuffle(buffer_size=len(self.X_train), seed=self.random_state + self.client_id, reshuffle_each_iteration=True)
            .batch(batch_size)
        )

        global_trainable = [tf.identity(var) for var in self.model.trainable_variables]
        optimizer = self.model.optimizer
        bce = tf.keras.losses.BinaryCrossentropy()

        best_val_loss = float("inf")
        patience_counter = 0

        for _ in range(local_epochs):
            for batch_x, batch_y in dataset:
                batch_y = tf.cast(tf.reshape(batch_y, (-1, 1)), tf.float32)
                with tf.GradientTape() as tape:
                    preds = self.model(batch_x, training=True)
                    base_loss = bce(batch_y, preds)
                    prox_term = tf.constant(0.0, dtype=tf.float32)
                    for var, global_w in zip(self.model.trainable_variables, global_trainable):
                        prox_term += tf.nn.l2_loss(var - global_w)
                    total_loss = base_loss + (self.mu / 2.0) * prox_term

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            val_loss, _ = self.model.evaluate(self.X_val, self.y_val, batch_size=batch_size, verbose=0)
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.model.set_weights(parameters)
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        batch_size = int(config.get("batch_size", self.batch_size))

        self._local_train_fedprox(local_epochs=local_epochs, batch_size=batch_size)

        train_loss, train_acc = self.model.evaluate(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            verbose=0,
        )
        return self.model.get_weights(), len(self.X_train), {"accuracy": float(train_acc), "loss": float(train_loss)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, batch_size=self.batch_size, verbose=0)
        return float(loss), len(self.X_val), {"accuracy": float(accuracy), "client_id": int(self.client_id)}


class TrackingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy with per-round and per-client metric tracking."""

    def __init__(self, input_dim: int, initial_lr: float = 5e-4, **kwargs: Any) -> None:
        self.global_round_accuracy: List[float] = []
        self.global_round_loss: List[float] = []
        self.client_accuracy_by_round: Dict[int, Dict[str, float]] = {}
        self.latest_parameters: NDArrays | None = None
        self.loss_increase_streak = 0
        self.stopped_due_to_loss = False

        init_model = build_full_model(input_dim=input_dim, learning_rate=initial_lr)
        initial_parameters = ndarrays_to_parameters(init_model.get_weights())

        super().__init__(
            initial_parameters=initial_parameters,
            **kwargs,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters | None, Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        round_client_acc: Dict[str, float] = {}
        for client_proxy, eval_res in results:
            cid = str(client_proxy.cid)
            acc = float(eval_res.metrics.get("accuracy", 0.0))
            round_client_acc[cid] = acc

        self.client_accuracy_by_round[server_round] = round_client_acc

        global_acc = float(aggregated_metrics.get("accuracy", 0.0))
        self.global_round_accuracy.append(global_acc)
        if aggregated_loss is not None:
            round_loss = float(aggregated_loss)
            self.global_round_loss.append(round_loss)

            if len(self.global_round_loss) >= 2 and self.global_round_loss[-1] > self.global_round_loss[-2]:
                self.loss_increase_streak += 1
            else:
                self.loss_increase_streak = 0

            print(f"[Round {server_round}] Global Accuracy: {global_acc:.4f} | Global Loss: {round_loss:.4f}")
            if self.loss_increase_streak >= 5:
                self.stopped_due_to_loss = True
                raise RuntimeError(EARLY_STOP_MARKER)
        else:
            print(f"[Round {server_round}] Global Accuracy: {global_acc:.4f} | Global Loss: unavailable")

        return aggregated_loss, aggregated_metrics


def _save_fl_metrics(strategy: TrackingFedAvg, final_accuracy: float) -> None:
    FL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round_global_accuracy": strategy.global_round_accuracy,
        "round_global_loss": strategy.global_round_loss,
        "client_accuracy_by_round": strategy.client_accuracy_by_round,
        "final_federated_accuracy": final_accuracy,
        "stopped_due_to_loss_increase": strategy.stopped_due_to_loss,
    }
    FL_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_round_accuracy(round_acc: List[float]) -> None:
    ROUND_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rounds = list(range(1, len(round_acc) + 1))
    values = np.asarray(round_acc, dtype=float)
    window = min(5, len(values))
    kernel = np.ones(window) / window
    smooth = np.convolve(values, kernel, mode="same")

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_acc, marker="o", linewidth=1.5, alpha=0.45, label="Raw")
    plt.plot(rounds, smooth, linewidth=2.5, label="Smoothed")
    plt.title("Federated Rounds vs Global Accuracy")
    plt.xlabel("FL Round")
    plt.ylabel("Global Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROUND_PLOT_PATH, dpi=200)
    plt.close()


def _load_centralized_accuracy() -> float | None:
    metrics_path = PROJECT_ROOT / "data" / "processed" / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "accuracy" in data:
                return float(data["accuracy"])
        except (ValueError, TypeError):
            pass

    log_candidates = [PROJECT_ROOT / "improved_training_run.log", PROJECT_ROOT / "training_run.log"]
    for log_path in log_candidates:
        if not log_path.exists():
            continue
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)", content)
        if matches:
            return float(matches[-1])

    return None


def _plot_centralized_vs_federated(centralized_acc: float | None, federated_acc: float) -> None:
    COMPARE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Federated"]
    values = [federated_acc]
    if centralized_acc is not None:
        labels = ["Centralized", "Federated"]
        values = [centralized_acc, federated_acc]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.title("Centralized vs Federated Accuracy")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, min(0.99, value + 0.01), f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(COMPARE_PLOT_PATH, dpi=200)
    plt.close()


def _evaluate_global_model(weights: NDArrays, hospital_paths: List[Path], learning_rate: float = 5e-4) -> float:
    frames = [pd.read_csv(path) for path in hospital_paths]
    full_df = pd.concat(frames, axis=0, ignore_index=True)

    target_col = _resolve_target_column(full_df)
    y = _to_binary_labels(full_df[target_col].to_numpy())
    X = full_df.drop(columns=[target_col]).to_numpy(dtype=np.float32)

    model = build_full_model(input_dim=X.shape[1], learning_rate=learning_rate)
    model.set_weights(weights)
    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return float(accuracy_score(y, y_pred))


def _describe_loss_trend(losses: List[float], stopped_due_to_loss: bool) -> Tuple[str, str]:
    """Return textual loss trend and convergence status."""
    if not losses:
        return "No loss values captured.", "unknown"

    first_loss = losses[0]
    last_loss = losses[-1]
    delta = last_loss - first_loss
    trend = f"start={first_loss:.4f}, end={last_loss:.4f}, delta={delta:+.4f}"

    if stopped_due_to_loss:
        return trend, "diverged"
    if delta < 0:
        return trend, "converged"
    return trend, "unstable"


def _run_single_simulation(
    hospital_paths: List[Path],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    random_state: int,
    learning_rate: float,
    mu: float,
) -> Tuple[TrackingFedAvg, float]:
    """Run one FL simulation attempt and return strategy tracker and final global accuracy."""
    num_clients = 5
    sample_X, _ = _load_hospital_csv(hospital_paths[0])
    input_dim = sample_X.shape[1]

    def client_fn(cid: str) -> fl.client.Client:
        client_idx = int(cid)
        client = HealthcareFLClient(
            client_id=client_idx,
            csv_path=hospital_paths[client_idx],
            local_epochs=local_epochs,
            batch_size=batch_size,
            random_state=random_state,
            learning_rate=learning_rate,
            mu=mu,
        )
        return client.to_client()

    strategy = TrackingFedAvg(
        input_dim=input_dim,
        initial_lr=learning_rate,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=_weighted_accuracy,
        on_fit_config_fn=lambda _: {"local_epochs": local_epochs, "batch_size": batch_size},
    )

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1},
        )
    except Exception as exc:
        cause_text = str(getattr(exc, "__cause__", ""))
        if EARLY_STOP_MARKER not in str(exc) and EARLY_STOP_MARKER not in cause_text:
            raise
        print("[WARN] Early stopping triggered due to 5 consecutive global-loss increases.")

    if strategy.latest_parameters is None:
        raise RuntimeError("Federated training completed without aggregated model weights.")

    final_federated_accuracy = _evaluate_global_model(
        strategy.latest_parameters,
        hospital_paths,
        learning_rate=learning_rate,
    )
    return strategy, final_federated_accuracy


def run_federated_pipeline(
    rounds: int = 20,
    local_epochs: int = 3,
    batch_size: int = 32,
    random_state: int = 42,
    learning_rate: float = 3e-4,
    mu: float = 0.005,
) -> Dict[str, Any]:
    """Run a complete 5-client Flower federated training pipeline."""
    hospital_paths = _ensure_hospital_datasets()

    print("[INFO] Starting federated training...")

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{existing_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

    print(
        f"[INFO] Single FL run config: rounds={rounds}, local_epochs={local_epochs}, "
        f"batch_size={batch_size}, lr={learning_rate}, mu={mu}"
    )

    strategy, final_federated_accuracy = _run_single_simulation(
        hospital_paths=hospital_paths,
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        random_state=random_state,
        learning_rate=learning_rate,
        mu=mu,
    )
    centralized_acc = _load_centralized_accuracy()

    GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_X, _ = _load_hospital_csv(hospital_paths[0])
    input_dim = sample_X.shape[1]
    global_model = build_full_model(input_dim=input_dim, learning_rate=learning_rate)
    global_model.set_weights(strategy.latest_parameters)
    global_model.save(GLOBAL_MODEL_PATH, include_optimizer=False)

    _save_fl_metrics(strategy=strategy, final_accuracy=final_federated_accuracy)
    _plot_round_accuracy(strategy.global_round_accuracy)
    _plot_centralized_vs_federated(centralized_acc=centralized_acc, federated_acc=final_federated_accuracy)

    loss_trend, status = _describe_loss_trend(strategy.global_round_loss, strategy.stopped_due_to_loss)

    print(f"Final Federated Model Accuracy: {final_federated_accuracy:.4f}")
    if centralized_acc is not None:
        print(f"Centralized Accuracy         : {centralized_acc:.4f}")
        print(f"Federated - Centralized Delta: {final_federated_accuracy - centralized_acc:+.4f}")
    else:
        print("Centralized Accuracy         : unavailable")
    print(f"Loss trend                  : {loss_trend}")
    print(f"Training status             : {status}")

    return {
        "final_federated_accuracy": final_federated_accuracy,
        "centralized_accuracy": centralized_acc,
        "round_global_accuracy": strategy.global_round_accuracy,
        "round_global_loss": strategy.global_round_loss,
        "loss_trend": loss_trend,
        "training_status": status,
        "client_accuracy_by_round": strategy.client_accuracy_by_round,
    }


if __name__ == "__main__":
    run_federated_pipeline(rounds=20, local_epochs=3, batch_size=32)
