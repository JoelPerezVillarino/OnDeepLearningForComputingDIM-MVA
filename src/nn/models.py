import numpy as np
import tensorflow as tf


def loadSequentialModel(
    units,
    num_layers,
    num_outputs,
    kernel_init="glorot_uniform",
    preprocessing_layer=None, 
):
    model = tf.keras.Sequential() 
    if preprocessing_layer is not None:
        model.add(preprocessing_layer)
    for _ in range(num_layers):
        model.add(
            tf.keras.layers.Dense(
                units=units,
                activation="relu",
                kernel_initializer=kernel_init,
            )
        )
    # Output layer
    model.add(
        tf.keras.layers.Dense(
            units=num_outputs,
            activation="linear",
            kernel_initializer=kernel_init,
        )
    )
    return model

