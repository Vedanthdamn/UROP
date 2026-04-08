"""Server-side model definition for split learning."""

from __future__ import annotations

import tensorflow as tf


def get_server_model(activation_dim: int = 32) -> tf.keras.Model:
    """Build and compile the server-side model for split learning.

    Args:
        activation_dim: Size of client activation vectors received by the server.

    Returns:
        Compiled Keras model that produces binary predictions.
    """
    if activation_dim <= 0:
        raise ValueError("activation_dim must be a positive integer.")

    inputs = tf.keras.Input(shape=(activation_dim,), name="server_input")
    x = tf.keras.layers.Dense(16, activation="relu", kernel_initializer="he_normal", name="server_dense_16")(inputs)
    x = tf.keras.layers.BatchNormalization(name="server_bn_16")(x)
    x = tf.keras.layers.Dropout(0.2, name="server_dropout_16")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="server_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="server_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
