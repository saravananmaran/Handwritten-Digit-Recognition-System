"""
Convolutional Neural Network (CNN) architecture and model utilities
for handwritten digit recognition.
"""

import os
from typing import Optional


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.3):
    """
    Constructs the CNN architecture specified for the digit recognition project:
      1. Input layer: 28 x 28 x 1
      2. Conv2D: 32 filters, 3x3 kernel, ReLU
      3. MaxPooling2D: 2x2 pool size
      4. Conv2D: 64 filters, 3x3 kernel, ReLU
      5. MaxPooling2D: 2x2 pool size
      6. Flatten layer
      7. Dense: 128 units, ReLU
      8. Dropout: dropout_rate (to prevent overfitting)
      9. Dense (Output): 10 units, Softmax activation
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_features"),
        layers.Dropout(dropout_rate, name="dropout_regularization"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities")
    ], name="mnist_cnn_classifier")

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compiles the model with Adam optimizer, sparse categorical crossentropy loss,
    and accuracy metric.
    """
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_trained_model(model_path: str = "models/mnist_cnn.keras"):
    """
    Loads a saved Keras model file safely.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Please train the model first by running 'python train.py'."
        )

    import tensorflow as tf
    return tf.keras.models.load_model(model_path)
