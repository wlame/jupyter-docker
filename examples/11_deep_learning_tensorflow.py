#!/usr/bin/env python3
"""
TensorFlow and Keras Deep Learning Basics
=========================================
Demonstrates fundamental operations with TensorFlow and Keras.

TensorFlow: https://tensorflow.org/
Keras: https://keras.io/
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# =============================================================================
# TensorFlow Basics
# =============================================================================
print("=" * 60)
print("TensorFlow Basics")
print("=" * 60)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs available: {len(gpus)}")

# Creating tensors
x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
y = tf.random.normal((3, 4))
z = tf.zeros((2, 3))
ones = tf.ones((2, 2))

print(f"\n1D Tensor: {x.numpy()}")
print(f"Random tensor shape: {y.shape}")
print(f"Zeros tensor:\n{z.numpy()}")

# Tensor operations
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

print(f"\nMatrix multiplication:\n{tf.matmul(a, b).numpy()}")
print(f"Element-wise multiplication:\n{(a * b).numpy()}")
print(f"Sum: {tf.reduce_sum(a).numpy()}")
print(f"Mean: {tf.reduce_mean(a).numpy()}")

# =============================================================================
# Automatic Differentiation with GradientTape
# =============================================================================
print("\n" + "=" * 60)
print("Automatic Differentiation with GradientTape")
print("=" * 60)

x = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    y = x ** 2 + 3 * x + 1

gradients = tape.gradient(y, x)
print(f"x = {x.numpy()}")
print(f"y = x^2 + 3x + 1 = {y.numpy()}")
print(f"dy/dx = 2x + 3 = {gradients.numpy()}")

# =============================================================================
# Keras Sequential Model
# =============================================================================
print("\n" + "=" * 60)
print("Keras Sequential Model")
print("=" * 60)

# Create a simple sequential model
model = models.Sequential([
    layers.Dense(64, activation="relu", input_shape=(10,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# =============================================================================
# Keras Functional API
# =============================================================================
print("\n" + "=" * 60)
print("Keras Functional API")
print("=" * 60)

# Define model using Functional API
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

functional_model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
print(f"Functional model: {functional_model.name}")
print(f"Input shape: {functional_model.input_shape}")
print(f"Output shape: {functional_model.output_shape}")

# =============================================================================
# Training Example (XOR Problem)
# =============================================================================
print("\n" + "=" * 60)
print("Training Example (XOR Problem)")
print("=" * 60)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build XOR model
xor_model = models.Sequential([
    layers.Dense(4, activation="relu", input_shape=(2,)),
    layers.Dense(1, activation="sigmoid"),
])

xor_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
print("Training XOR classifier...")
history = xor_model.fit(X, y, epochs=500, verbose=0)

# Print training progress
print(f"Initial loss: {history.history['loss'][0]:.4f}")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")

# Test predictions
predictions = xor_model.predict(X, verbose=0)
print(f"\nPredictions after training:")
for i in range(len(X)):
    print(f"  {X[i].tolist()} -> {predictions[i][0]:.4f} (expected: {y[i][0]})")

# =============================================================================
# Custom Layer Example
# =============================================================================
print("\n" + "=" * 60)
print("Custom Layer Example")
print("=" * 60)


class CustomDense(layers.Layer):
    """Custom dense layer with trainable weights."""

    def __init__(self, units):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Use custom layer
custom_layer = CustomDense(32)
test_input = tf.random.normal((4, 16))
output = custom_layer(test_input)
print(f"Custom layer input shape: {test_input.shape}")
print(f"Custom layer output shape: {output.shape}")
print(f"Custom layer weights: {len(custom_layer.trainable_weights)}")

# =============================================================================
# Callbacks Example
# =============================================================================
print("\n" + "=" * 60)
print("Callbacks Example")
print("=" * 60)

# Define callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=10,
    restore_best_weights=True,
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=5,
)

print("Defined callbacks:")
print("  - EarlyStopping: stops training when loss stops improving")
print("  - ReduceLROnPlateau: reduces learning rate when loss plateaus")

# =============================================================================
# Model Save/Load
# =============================================================================
print("\n" + "=" * 60)
print("Model Save/Load")
print("=" * 60)

# Save model architecture to JSON
model_json = xor_model.to_json()
print("Model can be saved as:")
print("  - Full model: model.save('model.keras')")
print("  - Weights only: model.save_weights('weights.weights.h5')")
print("  - Architecture: model.to_json()")
print(f"\nXOR model architecture (JSON preview): {model_json[:100]}...")

print("\n" + "=" * 60)
print("TensorFlow/Keras example complete!")
print("=" * 60)
