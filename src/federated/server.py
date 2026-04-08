"""Flower federated server for healthcare training with FedAvg."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays

from src.models import get_client_model, get_server_model

if TYPE_CHECKING:
    import tensorflow as tf


def _build_global_model(input_dim: int) -> "tf.keras.Model":
    """Build the full model used for global checkpointing on the server."""
    import tensorflow as tf

    client = get_client_model(input_dim=input_dim)
    server = get_server_model(activation_dim=32)

    inputs = tf.keras.Input(shape=(input_dim,), name="federated_input")
    activations = client(inputs)
    outputs = server(activations)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="global_federated_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class SavingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy with round logging and global model checkpointing."""

    def __init__(self, model_save_path: str, input_dim: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._model_save_path = Path(model_save_path)
        self._model_save_path.parent.mkdir(parents=True, exist_ok=True)

        self._model = _build_global_model(input_dim=input_dim)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[object],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            self._model.set_weights(ndarrays)
            self._model.save(self._model_save_path, include_optimizer=False)

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[object],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            print(f"[Round {server_round}] Global Accuracy: {float(accuracy):.4f}")
        else:
            print(f"[Round {server_round}] Global Accuracy: not reported")

        return aggregated_loss, metrics


def _weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate accuracy from clients using sample-count weighting."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    weighted_accuracy = 0.0
    for num_examples, client_metrics in metrics:
        if "accuracy" in client_metrics:
            weighted_accuracy += num_examples * float(client_metrics["accuracy"])

    return {"accuracy": weighted_accuracy / total_examples}


def start_federated_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 20,
    fraction_fit: float = 1.0,
    input_dim: int = 13,
    model_save_path: str = "models/global_model.keras",
) -> fl.server.history.History:
    """Start a Flower FedAvg server for healthcare federated learning.

    Args:
        server_address: Flower server bind address.
        num_rounds: Number of federated rounds (must be in [20, 50]).
        fraction_fit: Fraction of available clients used during fit.
        input_dim: Number of feature columns for model construction.
        model_save_path: Path to save the latest global model each round.

    Returns:
        Flower history object from training.
    """
    if not (20 <= num_rounds <= 50):
        raise ValueError("num_rounds must be between 20 and 50.")

    strategy = SavingFedAvg(
        model_save_path=model_save_path,
        input_dim=input_dim,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=_weighted_average,
    )

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    return history


if __name__ == "__main__":
    start_federated_server()
