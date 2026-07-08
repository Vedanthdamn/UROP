"""Centralized full model for healthcare binary classification."""

from __future__ import annotations

import tensorflow as tf


def build_full_model(input_dim: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Build and compile the centralized binary classifier.

    Args:
        input_dim: Number of feature columns.

    Returns:
        Compiled Keras model.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")

    # LayerNormalization (instead of BatchNormalization) normalizes per-sample and
    # carries no running statistics, so it stays consistent when weights are averaged
    # across non-IID hospital clients in the federated / SplitFed pipelines. This keeps
    # the same normalize -> dropout -> dense structure while fixing the miscalibration
    # (high loss) that batch-statistic averaging caused under FedProx aggregation.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model
