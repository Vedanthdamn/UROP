"""Flower NumPyClient implementation for hospital-specific healthcare FL training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from client_model import get_client_model
from dp_utils import apply_dp
from server_model import get_server_model


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


def _load_hospital_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Hospital dataset is empty: {csv_path}")

    target_col = _resolve_target_column(df)
    y = _to_binary_labels(df[target_col].to_numpy())
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    return X, y


def _build_federated_model(input_dim: int) -> tf.keras.Model:
    """Compose client-side and server-side Keras parts into one trainable model."""
    client = get_client_model(input_dim=input_dim)
    server = get_server_model(activation_dim=32)

    inputs = tf.keras.Input(shape=(input_dim,), name="federated_input")
    activations = client(inputs)
    outputs = server(activations)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="federated_healthcare_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class HospitalClient(fl.client.NumPyClient):
    """Flower client that trains/evaluates on one hospital's local dataset."""

    def __init__(
        self,
        hospital_csv_path: str,
        local_epochs: int = 1,
        batch_size: int = 32,
        validation_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        X, y = _load_hospital_data(hospital_csv_path)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=random_state,
            stratify=y,
        )

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.model = _build_federated_model(input_dim=self.X_train.shape[1])

    def _train_with_dp(self, local_epochs: int, batch_size: int, noise_multiplier: float) -> None:
        """Run local training using DP-clipped and noised gradients."""
        dataset = (
            tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
            .shuffle(buffer_size=len(self.X_train), reshuffle_each_iteration=True)
            .batch(batch_size)
        )

        for _ in range(local_epochs):
            for batch_x, batch_y in dataset:
                batch_y = tf.cast(tf.reshape(batch_y, (-1, 1)), tf.float32)
                with tf.GradientTape() as tape:
                    preds = self.model(batch_x, training=True)
                    loss = self.model.compiled_loss(batch_y, preds)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                dp_gradients = apply_dp(gradients, noise_multiplier=noise_multiplier)
                self.model.optimizer.apply_gradients(zip(dp_gradients, self.model.trainable_variables))

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.model.set_weights(parameters)

        local_epochs = int(config.get("local_epochs", self.local_epochs))
        batch_size = int(config.get("batch_size", self.batch_size))
        noise_multiplier = float(config.get("noise_multiplier", 0.02))

        self._train_with_dp(
            local_epochs=local_epochs,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
        )

        train_loss, train_accuracy, train_precision, train_recall = self.model.evaluate(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            verbose=0,
        )

        metrics = {
            "loss": float(train_loss),
            "accuracy": float(train_accuracy),
            "precision": float(train_precision),
            "recall": float(train_recall),
        }

        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_weights(parameters)

        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_val,
            self.y_val,
            batch_size=self.batch_size,
            verbose=0,
        )

        return float(loss), len(self.X_val), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }


def start_hospital_client(server_address: str, hospital_csv_path: str) -> None:
    """Utility runner to launch one Flower client process."""
    client = HospitalClient(hospital_csv_path=hospital_csv_path)
    fl.client.start_numpy_client(server_address=server_address, client=client)
