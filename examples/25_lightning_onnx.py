#!/usr/bin/env python3
"""
Training Loops and Deployment: Lightning, torchmetrics, Accelerate, einops, ONNX
=================================================================================
Trains a small classifier with PyTorch Lightning (metrics from torchmetrics,
logs for TensorBoard), runs the same model through a hand-written Accelerate
loop, reshapes tensors with einops, then writes the trained weights into an
ONNX graph and checks onnxruntime reproduces the PyTorch predictions.

Lightning:    https://lightning.ai/docs/pytorch/stable/
torchmetrics: https://lightning.ai/docs/torchmetrics/stable/
Accelerate:   https://huggingface.co/docs/accelerate/
einops:       https://einops.rocks/
ONNX:         https://onnx.ai/onnx/   onnxruntime: https://onnxruntime.ai/docs/
"""

import os

import lightning as L
import matplotlib
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchmetrics
from accelerate import Accelerator
from einops import rearrange, reduce
from onnx import helper, numpy_helper
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

L.seed_everything(0, verbose=False)
rng = np.random.default_rng(seed=0)

# =============================================================================
# Synthetic three-class data
# =============================================================================
n_per_class, n_features = 400, 8
centers = rng.normal(scale=2.5, size=(3, n_features))
X = np.concatenate([c + rng.normal(size=(n_per_class, n_features)) for c in centers]).astype('float32')
y = np.repeat(np.arange(3), n_per_class)
order = rng.permutation(len(y))
X, y = X[order], y[order]
split = int(0.8 * len(y))
train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))


def make_mlp() -> nn.Sequential:
    """Two-layer MLP shared by the Lightning and Accelerate sections."""
    return nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 3))


# =============================================================================
# Lightning + torchmetrics + TensorBoard
# =============================================================================
print("=" * 60)
print("Lightning: Train with torchmetrics")
print("=" * 60)


class Classifier(L.LightningModule):
    """Wraps the MLP with loss, accuracy, and an optimizer."""

    def __init__(self):
        super().__init__()
        self.net = make_mlp()
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=3)
        self.history: list[tuple[int, float, float]] = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, target = batch
        loss = nn.functional.cross_entropy(self(x), target)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, target = batch
        logits = self(x)
        self.val_acc.update(logits, target)
        self.log('val_loss', nn.functional.cross_entropy(logits, target), on_epoch=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute().item()
        self.log('val_acc', acc)
        loss = self.trainer.callback_metrics.get('train_loss')
        if loss is not None:
            self.history.append((self.current_epoch, float(loss), acc))
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


model = Classifier()
trainer = L.Trainer(
    max_epochs=8,
    accelerator='cpu',
    logger=L.pytorch.loggers.TensorBoardLogger(OUTPUT_DIR, name='lightning_logs'),
    enable_progress_bar=False,
    enable_model_summary=False,
    enable_checkpointing=False,
)
trainer.fit(model, DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(val_ds, batch_size=256))
final_acc = model.history[-1][2]
print(f"Validation accuracy after {trainer.current_epoch} epochs: {final_acc:.3f}")

epochs, losses, accs = zip(*model.history)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(epochs, losses, marker='o', label='train loss')
ax.plot(epochs, accs, marker='s', label='val accuracy')
ax.set_xlabel('epoch')
ax.legend()
ax.set_title('Lightning training curve')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lightning_training.png'), dpi=120)
plt.close()
print("Saved: lightning_training.png")

# =============================================================================
# Accelerate — the same model in a plain training loop
# =============================================================================
print("\n" + "=" * 60)
print("Accelerate: Device-Agnostic Loop")
print("=" * 60)

accelerator = Accelerator(cpu=True)
net = make_mlp()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loader = DataLoader(train_ds, batch_size=64, shuffle=True)
net, optimizer, loader = accelerator.prepare(net, optimizer, loader)
writer = SummaryWriter(os.path.join(OUTPUT_DIR, 'accelerate_logs'))
for epoch in range(5):
    total = 0.0
    for x, target in loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(net(x), target)
        accelerator.backward(loss)
        optimizer.step()
        total += loss.item()
    writer.add_scalar('loss', total / len(loader), epoch)
writer.close()
print(f"Device: {accelerator.device}; final epoch loss {total / len(loader):.4f} (logged for TensorBoard)")

# =============================================================================
# einops — readable tensor reshapes
# =============================================================================
print("\n" + "=" * 60)
print("einops: Patches and Pooling")
print("=" * 60)

images = torch.randn(4, 3, 32, 32)
patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8)
pooled = reduce(images, 'b c (h 2) (w 2) -> b c h w', 'max')
print(f"Images {tuple(images.shape)} -> patches {tuple(patches.shape)}, 2x2 max-pool {tuple(pooled.shape)}")

# =============================================================================
# ONNX + onnxruntime — the trained MLP as a portable graph
# =============================================================================
print("\n" + "=" * 60)
print("ONNX: Build, Check, and Run with onnxruntime")
print("=" * 60)

first, second = model.net[0], model.net[2]
weights = [
    numpy_helper.from_array(first.weight.detach().numpy().T.copy(), 'W1'),
    numpy_helper.from_array(first.bias.detach().numpy(), 'B1'),
    numpy_helper.from_array(second.weight.detach().numpy().T.copy(), 'W2'),
    numpy_helper.from_array(second.bias.detach().numpy(), 'B2'),
]
graph = helper.make_graph(
    [
        helper.make_node('MatMul', ['x', 'W1'], ['h0']),
        helper.make_node('Add', ['h0', 'B1'], ['h1']),
        helper.make_node('Relu', ['h1'], ['h2']),
        helper.make_node('MatMul', ['h2', 'W2'], ['h3']),
        helper.make_node('Add', ['h3', 'B2'], ['logits']),
    ],
    'tiny_mlp',
    [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [None, n_features])],
    [helper.make_tensor_value_info('logits', onnx.TensorProto.FLOAT, [None, 3])],
    initializer=weights,
)
# onnxruntime can lag the newest IR version onnx writes by default (onnx 1.23
# writes IR 14; onnxruntime 1.30 reads up to 13), so pin a widely supported one.
onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10)
onnx.checker.check_model(onnx_model)
onnx_path = os.path.join(OUTPUT_DIR, 'tiny_mlp.onnx')
onnx.save(onnx_model, onnx_path)

session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
x_val = X[split:]
ort_logits = session.run(['logits'], {'x': x_val})[0]
with torch.no_grad():
    torch_logits = model(torch.from_numpy(x_val)).numpy()
print(f"onnxruntime matches PyTorch: {np.allclose(ort_logits, torch_logits, atol=1e-5)}")
print(f"Saved: tiny_mlp.onnx ({os.path.getsize(onnx_path):,} bytes)")

print("\nDone.")
