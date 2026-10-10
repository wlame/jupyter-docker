#!/usr/bin/env python3
"""
Lorenz Attractor: SymPy, Numba, FFT, and Dask
=============================================
Explores the Lorenz system, the classic example of deterministic chaos. SymPy
finds its fixed points exactly, Numba compiles a Runge-Kutta integrator to
machine code, an FFT shows the dominant oscillation, and Dask runs an ensemble
of nearly identical trajectories in parallel to measure how fast they diverge
(the largest Lyapunov exponent, about 0.9 for these parameters).

In a notebook, `Client()` shows a link to the Dask dashboard. The images serve
it through Jupyter at /proxy/8787/status (jupyter-server-proxy), so it works
without publishing another port.

SymPy:   https://docs.sympy.org/
Numba:   https://numba.readthedocs.io/
SciPy:   https://docs.scipy.org/doc/scipy/
Dask:    https://docs.dask.org/
xarray:  https://docs.xarray.dev/
"""

import os
import time

import matplotlib
import numpy as np
import sympy as sp
import xarray as xr
from distributed import Client
from numba import njit
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.005
START_STATE = np.array([1.0, 1.0, 1.0])
ENSEMBLE_SIZE = 32
PERTURBATION = 1e-9
KNOWN_LYAPUNOV = 0.906

# =============================================================================
# SymPy: fixed points and their stability
# =============================================================================
print("=" * 60)
print("SymPy: Fixed Points of the Lorenz System")
print("=" * 60)

x, y, z = sp.symbols('x y z', real=True)
sigma, rho, beta = sp.symbols('sigma rho beta', positive=True)
field = sp.Matrix([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
fixed_points = sp.solve(list(field), [x, y, z], dict=True)
jacobian = field.jacobian([x, y, z])
values = {sigma: SIGMA, rho: RHO, beta: sp.Rational(8, 3)}
for point in fixed_points:
    exact = tuple(sp.simplify(point[v]) for v in (x, y, z))
    eigenvalues = np.linalg.eigvals(np.array(jacobian.subs(point).subs(values), dtype=float))
    print(f"  {exact}")
    print(f"     largest real eigenvalue at rho=28: {eigenvalues.real.max():+.3f} (unstable if > 0)")


# =============================================================================
# Numba: a compiled RK4 integrator
# =============================================================================
@njit(nogil=True)
def lorenz_rk4(state0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """Integrate the Lorenz system with fourth-order Runge-Kutta; returns (steps + 1, 3)."""

    def field(s):
        return np.array([SIGMA * (s[1] - s[0]), s[0] * (RHO - s[2]) - s[1], s[0] * s[1] - BETA * s[2]])

    path = np.empty((steps + 1, 3))
    path[0] = state0
    for i in range(steps):
        s = path[i]
        k1 = field(s)
        k2 = field(s + 0.5 * dt * k1)
        k3 = field(s + 0.5 * dt * k2)
        k4 = field(s + dt * k3)
        path[i + 1] = s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return path


print("\n" + "=" * 60)
print("Numba: Compiled Runge-Kutta Integrator")
print("=" * 60)

steps = 40_000
lorenz_rk4(START_STATE, 10, DT)  # first call compiles; keep it out of the timing
started = time.perf_counter()
trajectory = lorenz_rk4(START_STATE, steps, DT)
compiled_seconds = time.perf_counter() - started

started = time.perf_counter()
lorenz_rk4.py_func(START_STATE, steps // 20, DT)
python_seconds = (time.perf_counter() - started) * 20
print(f"{steps:,} steps: compiled {compiled_seconds * 1000:.1f} ms, "
      f"pure Python about {python_seconds * 1000:.0f} ms ({python_seconds / compiled_seconds:.0f}x slower)")

# Chaos amplifies any difference, so compare with SciPy only over a short horizon.
horizon = 2.0
reference = solve_ivp(
    lambda _t, s: [SIGMA * (s[1] - s[0]), s[0] * (RHO - s[2]) - s[1], s[0] * s[1] - BETA * s[2]],
    t_span=(0.0, horizon),
    y0=START_STATE,
    rtol=1e-10,
    atol=1e-12,
)
gap = np.abs(trajectory[int(horizon / DT)] - reference.y[:, -1]).max()
print(f"Agreement with SciPy solve_ivp at t={horizon}: max difference {gap:.2e}")

settled = trajectory[2_000:]  # drop the transient
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(settled[:, 0], settled[:, 1], settled[:, 2], lw=0.3, color='#1f4e79')
ax.set(xlabel='x', ylabel='y', zlabel='z', title='Lorenz attractor (rho=28)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lorenz_attractor.png'), dpi=120)
plt.close()
print("Saved: lorenz_attractor.png")

# =============================================================================
# FFT: the dominant oscillation of z(t)
# =============================================================================
print("\n" + "=" * 60)
print("FFT: Spectrum of z(t)")
print("=" * 60)

z_signal = settled[:, 2] - settled[:, 2].mean()
power = np.abs(np.fft.rfft(z_signal * np.hanning(len(z_signal)))) ** 2
frequencies = np.fft.rfftfreq(len(z_signal), d=DT)
peaks, _ = find_peaks(power, height=power.max() * 0.2)
dominant = frequencies[peaks[np.argmax(power[peaks])]]
print(f"Dominant frequency: {dominant:.3f} cycles per time unit (period {1 / dominant:.3f})")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.semilogy(frequencies, power, lw=0.8)
ax.axvline(dominant, color='crimson', ls='--', label=f'peak {dominant:.2f}')
ax.set(xlim=(0, 5), xlabel='frequency (cycles per time unit)', ylabel='power', title='Power spectrum of z(t)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lorenz_spectrum.png'), dpi=120)
plt.close()
print("Saved: lorenz_spectrum.png")

# =============================================================================
# Dask: an ensemble of nearby trajectories, in parallel
# =============================================================================
print("\n" + "=" * 60)
print("Dask: Ensemble Divergence and the Lyapunov Exponent")
print("=" * 60)

ensemble_steps = 6_000  # 30 time units
origin = settled[0]
offsets = np.random.default_rng(seed=0).normal(size=(ENSEMBLE_SIZE, 3))
starts = list(origin + PERTURBATION * offsets)

# Threads in this process: the integrator releases the GIL (nogil=True), so
# members run in parallel without copying data between processes.
with Client(processes=False, threads_per_worker=os.cpu_count() or 2, n_workers=1, dashboard_address=':0') as client:
    print(f"Dask client: {sum(client.nthreads().values())} threads, dashboard at {client.dashboard_link}")
    baseline = client.submit(lorenz_rk4, origin, ensemble_steps, DT)
    members = client.map(lorenz_rk4, starts, steps=ensemble_steps, dt=DT)
    baseline_path = baseline.result()
    paths = np.stack(client.gather(members))

# xarray (backed by Dask chunks) labels the ensemble and reduces it lazily.
times = np.arange(ensemble_steps + 1) * DT
ensemble = xr.DataArray(paths, dims=('member', 'time', 'axis'), coords={'time': times, 'axis': ['x', 'y', 'z']})
ensemble = ensemble.chunk({'member': 8})
separation = np.sqrt(((ensemble - baseline_path) ** 2).sum('axis'))
log_separation = np.log(separation).mean('member').compute()

growth = (times > 1.0) & (times < 15.0)
slope, intercept = np.polyfit(times[growth], log_separation.values[growth], deg=1)
print(f"Estimated largest Lyapunov exponent: {slope:.2f} (literature value about {KNOWN_LYAPUNOV})")
print(f"A {PERTURBATION:g} difference reaches order 1 after about {np.log(1 / PERTURBATION) / slope:.0f} time units.")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(times, log_separation, lw=1, label='mean log separation')
ax.plot(times[growth], slope * times[growth] + intercept, 'r--', label=f'fit, slope {slope:.2f}')
ax.set(xlabel='time', ylabel='ln |delta|', title=f'Divergence of {ENSEMBLE_SIZE} trajectories')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lorenz_divergence.png'), dpi=120)
plt.close()
print("Saved: lorenz_divergence.png")

print("\nDone.")
