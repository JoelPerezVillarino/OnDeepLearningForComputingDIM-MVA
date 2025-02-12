import numpy as np
import tensorflow as tf


def loadSequentialModel(
    units,
    num_layers,
    num_inputs,
    num_outputs,
    activation,
    kernel_init="glorot_uniform",
    opt=tf.keras.optimizers.Adam(learning_rate=1.e-3),
    preprocessing_layer=None, 
    regularization=None,
    metrics=["mse", "mae"],
    loss ="mse"
):
    if isinstance(activation, list):
        output_activation = activation[1]
        hidden_activation = activation[0]
    else:
        hidden_activation = activation
        output_activation = activation
    model = tf.keras.Sequential() 
    if preprocessing_layer is not None:
        model.add(preprocessing_layer)
    for _ in range(num_layers):
        model.add(
            tf.keras.layers.Dense(
                units=units,
                activation=hidden_activation,
                kernel_initializer=kernel_init,
                kernel_regularizer=regularization
            )
        )
    # Output layer
    model.add(
        tf.keras.layers.Dense(
            units=num_outputs,
            activation=output_activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=regularization
        )
    )

    model.compile(
        optimizer=opt, 
        loss=loss,
        metrics=metrics if metrics is not None else None
    )
    return model


def loadWideDeepModel(
    units,
    num_layers,
    num_inputs,
    num_outputs,
    activation,
    kernel_init="glorot_uniform",
    opt=tf.keras.optimizers.Adam(learning_rate=1.e-3),
    preprocessing_layer=None, 
    regularization=None,
    metrics=["mse", "mae"],
    loss="mse"    
):
    if isinstance(activation, list):
        output_activation = activation[1]
        hidden_activation = activation[0]
    else:
        output_activation = activation
        hidden_activation = activation

    input_ = tf.keras.layers.Input(shape=(num_inputs,))
    if preprocessing_layer is not None:
        input_ = preprocessing_layer(input_)
    y = tf.keras.layers.Dense(units, hidden_activation, kernel_initializer=kernel_init)(input_)
    for _ in range(num_layers-1):
        y = tf.keras.layers.Dense(units, hidden_activation, kernel_initializer=kernel_init)(y)
    concat = tf.keras.layers.Concatenate()([input_, y])
    output = tf.keras.layers.Dense(num_outputs, output_activation, 
                kernel_initializer=kernel_init)(concat)
    model = tf.keras.Model(inputs=[input_], outputs=[output])
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    model.summary()
    return model
