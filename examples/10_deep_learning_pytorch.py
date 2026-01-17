#!/usr/bin/env python3
"""
PyTorch Deep Learning Basics
============================
Demonstrates fundamental operations with PyTorch.

PyTorch: https://pytorch.org/
TorchVision: https://pytorch.org/vision/
TorchAudio: https://pytorch.org/audio/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchaudio

# =============================================================================
# Tensor Basics
# =============================================================================
print("=" * 60)
print("PyTorch Tensor Basics")
print("=" * 60)

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Creating tensors
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.randn(3, 4)  # Random normal distribution
z = torch.zeros(2, 3)
ones = torch.ones(2, 2)

print(f"\n1D Tensor: {x}")
print(f"Random tensor shape: {y.shape}")
print(f"Zeros tensor:\n{z}")

# Tensor operations
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"\nMatrix multiplication:\n{torch.matmul(a, b)}")
print(f"Element-wise multiplication:\n{a * b}")
print(f"Sum: {a.sum()}")
print(f"Mean: {a.mean()}")

# =============================================================================
# Autograd - Automatic Differentiation
# =============================================================================
print("\n" + "=" * 60)
print("Autograd - Automatic Differentiation")
print("=" * 60)

# Create tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# Compute gradients
y.sum().backward()
print(f"x = {x.data}")
print(f"y = x^2 + 3x + 1 = {y.data}")
print(f"dy/dx = 2x + 3 = {x.grad}")

# =============================================================================
# Simple Neural Network
# =============================================================================
print("\n" + "=" * 60)
print("Simple Neural Network")
print("=" * 60)


class SimpleNet(nn.Module):
    """A simple feedforward neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Create model
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
print(f"Model architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# =============================================================================
# Training Loop Example
# =============================================================================
print("\n" + "=" * 60)
print("Training Loop Example (XOR Problem)")
print("=" * 60)

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Simple network for XOR
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


xor_model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(xor_model.parameters(), lr=0.1)

# Training
print("Training XOR classifier...")
for epoch in range(1000):
    # Forward pass
    outputs = xor_model(X)
    loss = criterion(outputs, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")

# Test predictions
with torch.no_grad():
    predictions = xor_model(X)
    print(f"\nPredictions after training:")
    for i in range(len(X)):
        print(f"  {X[i].tolist()} -> {predictions[i].item():.4f} (expected: {y[i].item()})")

# =============================================================================
# TorchVision Transforms
# =============================================================================
print("\n" + "=" * 60)
print("TorchVision Transforms")
print("=" * 60)

# Define a transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Transform pipeline defined for image preprocessing")
print("  - Resize to 224x224")
print("  - Convert to tensor")
print("  - Normalize with ImageNet stats")

# =============================================================================
# TorchAudio Basics
# =============================================================================
print("\n" + "=" * 60)
print("TorchAudio Basics")
print("=" * 60)

print(f"TorchAudio version: {torchaudio.__version__}")

# Create a simple synthetic waveform
sample_rate = 16000
duration = 1.0  # seconds
t = torch.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # Hz (A4 note)
waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

print(f"\nSynthetic waveform created:")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Duration: {duration} seconds")
print(f"  Frequency: {frequency} Hz (A4 note)")
print(f"  Shape: {waveform.shape}")

# Apply a simple transform
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=64,
)
mel_spec = mel_spectrogram(waveform)
print(f"  Mel spectrogram shape: {mel_spec.shape}")

print("\n" + "=" * 60)
print("PyTorch example complete!")
print("=" * 60)
