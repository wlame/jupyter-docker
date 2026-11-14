#!/usr/bin/env python3
# # The Same Network in PyTorch and in Keras
#
# One small multilayer perceptron, trained on the handwritten digits that ship
# with scikit-learn, written twice: in PyTorch with an explicit training loop and
# in Keras with `model.fit`. Same architecture, optimizer, learning rate, batch
# size, and number of epochs, so the side-by-side code shows how the two
# frameworks express the same ideas. Both should reach a similar test accuracy.
#
# - [PyTorch](https://pytorch.org/docs/stable/)
# - [Keras](https://keras.io/)

# PyTorch first: in images that also ship TensorFlow, torch must load before it.
import torch
from torch import nn

import json
import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 0
HIDDEN = 64
EPOCHS = 30
BATCH = 64
LEARNING_RATE = 1e-3
torch.manual_seed(SEED)
keras.utils.set_random_seed(SEED)

# ## Shared data
#
# 1,797 8×8 grey images of digits, flattened to 64 features and standardized with
# statistics from the training split only.

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, stratify=digits.target, random_state=SEED
)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train).astype(np.float32), scaler.transform(X_test).astype(np.float32)
print(f"train {X_train.shape}, test {X_test.shape}, classes {len(np.unique(y_train))}")

# ## PyTorch: define the model and write the loop
#
# PyTorch gives you the pieces (module, loss, optimizer) and you write the loop:
# forward pass, loss, backward pass, optimizer step.

torch_model = nn.Sequential(nn.Linear(64, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 10))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(torch_model.parameters(), lr=LEARNING_RATE)
features, labels = torch.from_numpy(X_train), torch.from_numpy(y_train)

torch_history = []
started = time.perf_counter()
for epoch in range(EPOCHS):
    torch_model.train()
    order = torch.randperm(len(features))
    epoch_loss = 0.0
    for start in range(0, len(order), BATCH):
        batch = order[start:start + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(torch_model(features[batch]), labels[batch])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    torch_history.append(epoch_loss / len(order))
torch_seconds = time.perf_counter() - started

torch_model.eval()
with torch.no_grad():
    torch_accuracy = (torch_model(torch.from_numpy(X_test)).argmax(dim=1).numpy() == y_test).mean()
print(f"PyTorch: test accuracy {torch_accuracy:.3f}, {torch_seconds:.1f} s")

# ## Keras: describe the model and call fit
#
# Keras bundles the loop: `compile` picks the loss and optimizer, `fit` runs the
# epochs and batches and records the history.

keras_model = keras.Sequential([
    keras.Input(shape=(64,)),
    keras.layers.Dense(HIDDEN, activation='relu'),
    keras.layers.Dense(HIDDEN, activation='relu'),
    keras.layers.Dense(10),
])
keras_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
started = time.perf_counter()
history = keras_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0)
keras_seconds = time.perf_counter() - started
_, keras_accuracy = keras_model.evaluate(X_test, y_test, verbose=0)
print(f"Keras:   test accuracy {keras_accuracy:.3f}, {keras_seconds:.1f} s")

# ## Compare
#
# Both have the same number of parameters; the loss curves differ a little because
# the frameworks initialize weights differently.

torch_params = sum(p.numel() for p in torch_model.parameters())
print(f"Parameters: PyTorch {torch_params:,}, Keras {keras_model.count_params():,}")
if min(torch_accuracy, keras_accuracy) < 0.9:
    raise SystemExit("one of the models failed to learn the digits")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, EPOCHS + 1), torch_history, label=f'PyTorch (test acc {torch_accuracy:.3f})')
ax.plot(range(1, EPOCHS + 1), history.history['loss'], label=f'Keras (test acc {keras_accuracy:.3f})')
ax.set(xlabel='epoch', ylabel='training loss', yscale='log', title='Same MLP, two frameworks')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'torch_keras_loss.png'), dpi=120)
plt.show()

with open(os.path.join(OUTPUT_DIR, 'torch_keras_summary.json'), 'w') as f:
    json.dump({'torch_accuracy': float(torch_accuracy), 'keras_accuracy': float(keras_accuracy),
               'parameters': torch_params}, f, indent=2)
print("Saved: torch_keras_loss.png, torch_keras_summary.json")
