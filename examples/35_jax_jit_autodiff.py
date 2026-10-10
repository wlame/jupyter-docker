#!/usr/bin/env python3
"""
JAX: JIT, Vectorization, Autodiff, Optax, Flax, and Equinox
===========================================================
Computes the Mandelbrot set with NumPy and with a JIT-compiled JAX function,
differentiates a function with `jax.grad` (checked against the exact
derivative), fits a curve by gradient descent with Optax, and trains the same
small network twice: with Flax NNX and with Equinox.

The image ships the CPU build of jaxlib. For an NVIDIA GPU, run
`pip install "jax[cuda13]"` inside the container; the code stays the same.

JAX:      https://docs.jax.dev/
Optax:    https://optax.readthedocs.io/
Flax NNX: https://flax.readthedocs.io/
Equinox:  https://docs.kidger.site/equinox/
"""

import json
import os
import time
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import nnx

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ITER = 100
TRAIN_STEPS = 1_500
summary: dict[str, object] = {'devices': [str(device) for device in jax.devices()]}
print(f"JAX {jax.__version__} on {summary['devices']}")

# =============================================================================
# JIT: the Mandelbrot set, NumPy versus compiled JAX
# =============================================================================
print("\n" + "=" * 60)
print("JIT: Mandelbrot Set")
print("=" * 60)

real, imag = np.meshgrid(np.linspace(-2.2, 0.8, 900), np.linspace(-1.2, 1.2, 720))
grid = (real + 1j * imag).astype(np.complex64)


def mandelbrot_numpy(c: np.ndarray, max_iter: int) -> np.ndarray:
    """Escape-time counts with NumPy: one masked update per iteration."""
    z = np.zeros_like(c)
    counts = np.zeros(c.shape, dtype=np.int32)
    for _ in range(max_iter):
        inside = np.abs(z) <= 2
        z[inside] = z[inside] ** 2 + c[inside]
        counts += inside
    return counts


@partial(jax.jit, static_argnums=1)
def mandelbrot_jax(c: jax.Array, max_iter: int) -> jax.Array:
    """The same algorithm as one XLA program: the loop runs inside the compiled code."""

    def step(_, state):
        z, counts = state
        inside = jnp.abs(z) <= 2
        return jnp.where(inside, z * z + c, z), counts + inside

    return jax.lax.fori_loop(0, max_iter, step, (jnp.zeros_like(c), jnp.zeros(c.shape, jnp.int32)))[1]


started = time.perf_counter()
numpy_counts = mandelbrot_numpy(grid, MAX_ITER)
numpy_seconds = time.perf_counter() - started

device_grid = jnp.asarray(grid)
mandelbrot_jax(device_grid, MAX_ITER).block_until_ready()  # compile once, outside the timing
started = time.perf_counter()
jax_counts = np.asarray(mandelbrot_jax(device_grid, MAX_ITER).block_until_ready())
jax_seconds = time.perf_counter() - started

agreement = float((numpy_counts == jax_counts).mean())
print(f"{grid.size:,} points x {MAX_ITER} iterations")
print(f"NumPy {numpy_seconds * 1000:.0f} ms, JAX (compiled) {jax_seconds * 1000:.0f} ms, "
      f"{numpy_seconds / jax_seconds:.1f}x faster")
print(f"Identical escape counts on {agreement:.2%} of points (float rounding differs at the boundary)")
summary['mandelbrot_speedup'] = round(numpy_seconds / jax_seconds, 1)

fig, ax = plt.subplots(figsize=(8, 6.4))
ax.imshow(np.log1p(jax_counts), extent=(-2.2, 0.8, -1.2, 1.2), cmap='magma', origin='lower')
ax.set(title='Mandelbrot set (JAX, jit + fori_loop)', xlabel='Re', ylabel='Im')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'jax_mandelbrot.png'), dpi=110)
plt.close()
print("Saved: jax_mandelbrot.png")

# =============================================================================
# Autodiff: jax.grad against the exact derivative
# =============================================================================
print("\n" + "=" * 60)
print("Autodiff: jax.grad and jax.vmap")
print("=" * 60)


def f(x):
    """A scalar function with a known derivative: x^2 sin(x)."""
    return x**2 * jnp.sin(x)


points = jnp.linspace(-3.0, 3.0, 7)
automatic = jax.vmap(jax.grad(f))(points)  # grad works on one scalar; vmap maps it over the array
exact = 2 * points * jnp.sin(points) + points**2 * jnp.cos(points)
print(f"max |grad - exact| over {points.size} points: {float(jnp.abs(automatic - exact).max()):.2e}")
print(f"second derivative at x=1: {float(jax.grad(jax.grad(f))(1.0)):.4f}")

# =============================================================================
# Optax: fit y = a sin(b x) + c by gradient descent
# =============================================================================
print("\n" + "=" * 60)
print("Optax: Curve Fitting with Adam")
print("=" * 60)

rng = np.random.default_rng(seed=0)
x_data = jnp.asarray(np.linspace(-4, 4, 200), dtype=jnp.float32)
true_params = {'a': 1.5, 'b': 0.8, 'c': -0.3}
y_data = true_params['a'] * jnp.sin(true_params['b'] * x_data) + true_params['c']
y_data = y_data + jnp.asarray(rng.normal(scale=0.1, size=x_data.shape), dtype=jnp.float32)


def curve_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean squared error of the sine model."""
    return jnp.mean((params['a'] * jnp.sin(params['b'] * x) + params['c'] - y) ** 2)


optimizer = optax.adam(learning_rate=0.05)
params = {'a': 1.0, 'b': 1.0, 'c': 0.0}
opt_state = optimizer.init(params)


@jax.jit
def curve_step(params, opt_state):
    """One Adam step on the curve parameters."""
    loss, grads = jax.value_and_grad(curve_loss)(params, x_data, y_data)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss


for _ in range(500):
    params, opt_state, loss = curve_step(params, opt_state)
print("fitted: " + ", ".join(f"{k}={float(v):.3f} (true {true_params[k]})" for k, v in params.items()))
print(f"final MSE {float(loss):.4f} (noise variance 0.0100)")

# =============================================================================
# The same MLP in Flax NNX and in Equinox
# =============================================================================
print("\n" + "=" * 60)
print("Flax NNX and Equinox: Regressing sin(3x)")
print("=" * 60)

x_train = jnp.linspace(-1, 1, 256).reshape(-1, 1)
y_train = jnp.sin(3 * x_train)


class FlaxMLP(nnx.Module):
    """1 -> 32 -> 32 -> 1 multilayer perceptron with tanh activations."""

    def __init__(self, rngs: nnx.Rngs):
        self.hidden1 = nnx.Linear(1, 32, rngs=rngs)
        self.hidden2 = nnx.Linear(32, 32, rngs=rngs)
        self.out = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x):
        return self.out(jnp.tanh(self.hidden2(jnp.tanh(self.hidden1(x)))))


flax_model = FlaxMLP(nnx.Rngs(0))
flax_optimizer = nnx.Optimizer(flax_model, optax.adam(1e-2), wrt=nnx.Param)


@nnx.jit
def flax_step(model: FlaxMLP, optimizer: nnx.Optimizer) -> jax.Array:
    """One training step; NNX updates the model's parameters in place."""
    loss, grads = nnx.value_and_grad(lambda m: jnp.mean((m(x_train) - y_train) ** 2))(model)
    optimizer.update(model, grads)
    return loss


equinox_model = eqx.nn.MLP(in_size=1, out_size=1, width_size=32, depth=2, activation=jnp.tanh, key=jax.random.key(0))
equinox_optimizer = optax.adam(1e-2)
equinox_state = equinox_optimizer.init(eqx.filter(equinox_model, eqx.is_array))


@eqx.filter_jit
def equinox_step(model: eqx.nn.MLP, state):
    """One training step; Equinox models are immutable PyTrees, so it returns a new model."""
    loss, grads = eqx.filter_value_and_grad(lambda m: jnp.mean((jax.vmap(m)(x_train) - y_train) ** 2))(model)
    updates, state = equinox_optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), state, loss


flax_losses, equinox_losses = [], []
for _ in range(TRAIN_STEPS):
    flax_losses.append(float(flax_step(flax_model, flax_optimizer)))
    equinox_model, equinox_state, loss = equinox_step(equinox_model, equinox_state)
    equinox_losses.append(float(loss))
print(f"after {TRAIN_STEPS} Adam steps: Flax NNX MSE {flax_losses[-1]:.1e}, Equinox MSE {equinox_losses[-1]:.1e}")
summary['final_mse'] = {'flax': flax_losses[-1], 'equinox': equinox_losses[-1]}

fig, (ax_loss, ax_fit) = plt.subplots(1, 2, figsize=(11, 4))
ax_loss.semilogy(flax_losses, label='Flax NNX')
ax_loss.semilogy(equinox_losses, label='Equinox')
ax_loss.set(xlabel='step', ylabel='MSE', title='Training loss')
ax_loss.legend()
ax_fit.plot(x_train, y_train, 'k', lw=3, alpha=0.3, label='sin(3x)')
ax_fit.plot(x_train, flax_model(x_train), label='Flax NNX')
ax_fit.plot(x_train, jax.vmap(equinox_model)(x_train), '--', label='Equinox')
ax_fit.set(title='Fitted functions')
ax_fit.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'jax_training.png'), dpi=110)
plt.close()
print("Saved: jax_training.png")

with open(os.path.join(OUTPUT_DIR, 'jax_summary.json'), 'w') as f_out:
    json.dump(summary, f_out, indent=2)
print("Saved: jax_summary.json")
print("\nDone.")
