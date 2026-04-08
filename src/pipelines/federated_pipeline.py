"""Federated learning pipeline using FedProx with strict test-only final evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_privacy_preserving_splits
from src.models import build_full_model

FL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "fl_metrics.json"
GLOBAL_MODEL_PATH = PROJECT_ROOT / "models" / "federated_global_model.h5"
ROUND_PLOT_PATH = PROJECT_ROOT / "plots" / "fl_rounds_accuracy.png"


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}
    weighted_acc = sum(num_examples * float(m.get("accuracy", 0.0)) for num_examples, m in metrics)
    return {"accuracy": weighted_acc / total}


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "loss": float(loss),
    }


class HealthcareFLClient(fl.client.NumPyClient):
    """Flower NumPyClient with FedProx local optimization and class balancing."""

    def __init__(
        self,
        client_id: int,
        data: Dict[str, np.ndarray],
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        default_mu: float,
        random_state: int,
    ) -> None:
        self.client_id = client_id
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.default_mu = default_mu
        self.random_state = random_state

        self.model = build_full_model(input_dim=self.X_train.shape[1], learning_rate=self.learning_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def _class_weight_lookup(self) -> Dict[int, float]:
        classes = np.unique(self.y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
        return {int(cls): float(w) for cls, w in zip(classes, weights)}

    def _fit_fedprox(self, local_epochs: int, mu: float) -> None:
        class_weight_map = self._class_weight_lookup()
        global_weights = [tf.identity(v) for v in self.model.trainable_variables]

        dataset = (
            tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
            .shuffle(len(self.X_train), seed=self.random_state + self.client_id, reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )

        for _ in range(local_epochs):
            for batch_x, batch_y in dataset:
                batch_y = tf.cast(tf.reshape(batch_y, (-1, 1)), tf.float32)
                sample_weight = tf.constant(
                    [class_weight_map[int(v)] for v in tf.reshape(batch_y, (-1,)).numpy()],
                    dtype=tf.float32,
                )

                with tf.GradientTape() as tape:
                    preds = self.model(batch_x, training=True)
                    supervised = self.loss_fn(batch_y, preds, sample_weight=sample_weight)
                    prox_term = tf.constant(0.0, dtype=tf.float32)
                    for local_var, global_var in zip(self.model.trainable_variables, global_weights):
                        prox_term += tf.nn.l2_loss(local_var - global_var)
                    total_loss = supervised + (mu / 2.0) * prox_term

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.model.set_weights(parameters)
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        mu = float(config.get("proximal_mu", self.default_mu))
        self._fit_fedprox(local_epochs=local_epochs, mu=mu)

        train_loss, train_acc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return self.model.get_weights(), len(self.X_train), {"accuracy": float(train_acc), "loss": float(train_loss)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.model.set_weights(parameters)
        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return float(val_loss), len(self.X_val), {"accuracy": float(val_acc)}


class TrackingFedProx(fl.server.strategy.FedProx):
    """FedProx strategy with round-wise metric tracking."""

    def __init__(self, input_dim: int, learning_rate: float, **kwargs: Any) -> None:
        self.global_round_accuracy: List[float] = []
        self.global_round_loss: List[float] = []
        self.latest_parameters: NDArrays | None = None

        init_model = build_full_model(input_dim=input_dim, learning_rate=learning_rate)
        initial_parameters = ndarrays_to_parameters(init_model.get_weights())
        super().__init__(initial_parameters=initial_parameters, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters | None, Dict[str, Scalar]]:
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.latest_parameters = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.global_round_accuracy.append(float(metrics.get("accuracy", 0.0)))
        self.global_round_loss.append(float(loss) if loss is not None else float("nan"))
        print(
            f"[Round {server_round}] Validation Accuracy: {self.global_round_accuracy[-1]:.4f} | "
            f"Validation Loss: {self.global_round_loss[-1]:.4f}"
        )
        return loss, metrics


def _plot_round_accuracy(round_acc: List[float]) -> None:
    ROUND_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(1, len(round_acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, round_acc, marker="o", linewidth=2)
    plt.title("Federated Rounds vs Validation Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROUND_PLOT_PATH, dpi=300)
    plt.close()


def run_federated_pipeline(
    rounds: int = 40,
    local_epochs: int = 5,
    batch_size: int = 32,
    random_state: int = 42,
    learning_rate: float = 2e-4,
    mu: float = 0.01,
) -> Dict[str, Any]:
    """Run FedProx pipeline and report final metrics on held-out test set only."""
    _set_global_determinism(random_state)

    split_payload = build_privacy_preserving_splits(random_state=random_state, force_rebuild=False)
    clients = split_payload["clients"]
    feature_dim = int(split_payload["feature_dim"])
    X_test, y_test = split_payload["global_test"]

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        client = HealthcareFLClient(
            client_id=idx,
            data=clients[idx],
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            default_mu=mu,
            random_state=random_state,
        )
        return client.to_client()

    strategy = TrackingFedProx(
        input_dim=feature_dim,
        learning_rate=learning_rate,
        proximal_mu=mu,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=_weighted_accuracy,
        on_fit_config_fn=lambda _: {"local_epochs": local_epochs, "batch_size": batch_size, "proximal_mu": mu},
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    if strategy.latest_parameters is None:
        raise RuntimeError("FedProx did not produce final aggregated parameters.")

    model = build_full_model(input_dim=feature_dim, learning_rate=learning_rate)
    model.set_weights(strategy.latest_parameters)

    GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(GLOBAL_MODEL_PATH, include_optimizer=False)

    test_loss, _ = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).ravel()
    final_metrics = _compute_binary_metrics(y_true=y_test, y_prob=y_prob, loss=float(test_loss))

    FL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FL_METRICS_PATH.write_text(
        json.dumps(
            {
                "round_global_accuracy": strategy.global_round_accuracy,
                "round_global_loss": strategy.global_round_loss,
                "final_federated_accuracy": final_metrics["accuracy"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "f1_score": final_metrics["f1_score"],
                "roc_auc": final_metrics["roc_auc"],
                "loss": final_metrics["loss"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_round_accuracy(strategy.global_round_accuracy)

    print("\nFederated Test Metrics")
    print(f"Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall   : {final_metrics['recall']:.4f}")
    print(f"F1-Score : {final_metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {final_metrics['roc_auc']:.4f}")
    print(f"Loss     : {final_metrics['loss']:.4f}")

    return {
        "final_federated_accuracy": final_metrics["accuracy"],
        "round_global_accuracy": strategy.global_round_accuracy,
        "round_global_loss": strategy.global_round_loss,
        "metrics": final_metrics,
    }


if __name__ == "__main__":
    run_federated_pipeline()
