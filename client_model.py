"""Client-side model definition for split learning."""

from __future__ import annotations

import tensorflow as tf


def get_client_model(input_dim: int) -> tf.keras.Model:
    """Build the client-side model that outputs intermediate activations.

    Args:
        input_dim: Number of input features.

    Returns:
        Keras model with two hidden layers and no final prediction layer.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")

    inputs = tf.keras.Input(shape=(input_dim,), name="client_input")
    x = tf.keras.layers.Dense(64, activation="relu", name="client_dense_64")(inputs)
    outputs = tf.keras.layers.Dense(32, activation="relu", name="client_dense_32")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="client_model")
