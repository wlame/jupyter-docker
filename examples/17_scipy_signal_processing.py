#!/usr/bin/env python3
"""
SciPy: Signal Processing and Optimization
==========================================
Demonstrates signal processing, FFT, filtering, and optimization with SciPy.

SciPy Signal: https://docs.scipy.org/doc/scipy/tutorial/signal.html
SciPy FFT:    https://docs.scipy.org/doc/scipy/tutorial/fft.html
SciPy Optim.: https://docs.scipy.org/doc/scipy/tutorial/optimize.html
SciPy Stats:  https://docs.scipy.org/doc/scipy/tutorial/stats.html
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import signal, fft, optimize, stats, interpolate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

# =============================================================================
# Signal Generation — Composite Noisy Signal
# =============================================================================
print("=" * 60)
print("Signal Generation")
print("=" * 60)

fs = 1000        # Sampling rate (Hz)
T = 2.0          # Duration (s)
t = np.linspace(0, T, int(fs * T), endpoint=False)

# True signal: 50 Hz + 120 Hz sine waves
freq1, freq2 = 50.0, 120.0
amp1, amp2 = 1.0, 0.5
clean = amp1 * np.sin(2 * np.pi * freq1 * t) + amp2 * np.sin(2 * np.pi * freq2 * t)

# Add Gaussian noise
noise_std = 0.4
noisy = clean + rng.normal(scale=noise_std, size=len(t))

print(f"Sampling rate : {fs} Hz")
print(f"Duration      : {T} s  ({len(t)} samples)")
print(f"Signal freqs  : {freq1} Hz + {freq2} Hz")
print(f"Noise std     : {noise_std}")
print(f"SNR           : {10 * np.log10((clean**2).mean() / noise_std**2):.1f} dB")

# =============================================================================
# FFT Analysis
# =============================================================================
print("\n" + "=" * 60)
print("FFT Analysis")
print("=" * 60)

N = len(t)
freqs = fft.rfftfreq(N, d=1 / fs)
spectrum_noisy = np.abs(fft.rfft(noisy)) * 2 / N
spectrum_clean = np.abs(fft.rfft(clean)) * 2 / N

# Find dominant frequencies
peaks_idx, peak_props = signal.find_peaks(spectrum_noisy, height=0.1, distance=10)
peak_freqs = freqs[peaks_idx]
peak_amps = spectrum_noisy[peaks_idx]

print("Dominant frequencies found:")
for f, a in sorted(zip(peak_freqs[:5], peak_amps[:5]), key=lambda x: -x[1]):
    print(f"  {f:.1f} Hz — amplitude {a:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(t[:500], clean[:500], label='Clean signal', linewidth=1.2, color='steelblue')
axes[0].plot(t[:500], noisy[:500], label='Noisy signal', linewidth=0.6, alpha=0.7, color='tomato')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Time Domain")
axes[0].legend()

axes[1].semilogy(freqs, spectrum_noisy, label='Noisy spectrum', linewidth=0.8, alpha=0.7, color='tomato')
axes[1].semilogy(freqs, spectrum_clean, label='Clean spectrum', linewidth=1.5, color='steelblue')
axes[1].plot(peak_freqs, peak_amps, 'x', color='black', markersize=10, markeredgewidth=2, label='Peaks')
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude (log)")
axes[1].set_title("Frequency Domain (FFT)")
axes[1].set_xlim([0, 300])
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scipy_fft.png'), dpi=150)
print("Saved: scipy_fft.png")
plt.close()

# =============================================================================
# Digital Filtering — Lowpass, Highpass, Bandpass, Notch
# =============================================================================
print("\n" + "=" * 60)
print("Digital Filtering")
print("=" * 60)

def apply_butter(data: np.ndarray, cutoff, fs: int, btype: str, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth filter via forward-backward filtering."""
    nyq = fs / 2
    norm = np.array(cutoff) / nyq
    b, z = signal.butter(order, norm, btype=btype)
    return signal.filtfilt(b, z, data)

# Lowpass: keep components below 80 Hz
lp_filtered = apply_butter(noisy, cutoff=80, fs=fs, btype='low')

# Highpass: keep components above 80 Hz
hp_filtered = apply_butter(noisy, cutoff=80, fs=fs, btype='high')

# Bandpass: isolate 40–70 Hz region (keeps the 50 Hz component)
bp_filtered = apply_butter(noisy, cutoff=[40, 70], fs=fs, btype='band')

# Notch filter: remove 50 Hz hum (b0, a0)
b_notch, a_notch = signal.iirnotch(w0=50, Q=30, fs=fs)
notch_filtered = signal.filtfilt(b_notch, a_notch, noisy)

filters = {
    'Lowpass < 80 Hz': lp_filtered,
    'Highpass > 80 Hz': hp_filtered,
    'Bandpass 40–70 Hz': bp_filtered,
    'Notch at 50 Hz': notch_filtered,
}

print("Filter results (RMS of filtered output):")
for name, out in filters.items():
    print(f"  {name:<22}: RMS = {np.sqrt(np.mean(out**2)):.4f}")

# Visualise filter frequency responses
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

for ax, (fname, filt_signal), color in zip(axes.ravel(), filters.items(), colors):
    ax.plot(t[:500], noisy[:500], alpha=0.35, linewidth=0.6, color='gray', label='Noisy')
    ax.plot(t[:500], filt_signal[:500], linewidth=1.5, color=color, label=fname)
    ax.set_title(fname)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)

plt.suptitle("Digital Filtering Results", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scipy_filters.png'), dpi=150)
print("Saved: scipy_filters.png")
plt.close()

# =============================================================================
# Spectrogram — Short-Time Fourier Transform
# =============================================================================
print("\n" + "=" * 60)
print("Spectrogram (STFT)")
print("=" * 60)

# Chirp signal: frequency sweeps from 50 to 300 Hz
chirp_signal = signal.chirp(t, f0=50, f1=300, t1=T, method='linear')
chirp_noisy = chirp_signal + rng.normal(scale=0.3, size=len(t))

stft = fft.ShortTimeFFT(
    win=signal.windows.hann(256),
    hop=64,
    fs=fs,
    mfft=512,
)
Sx = stft.spectrogram(chirp_noisy)

print(f"Chirp: 50 → 300 Hz sweep over {T} s")
print(f"STFT spectrogram shape: {Sx.shape}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(t, chirp_signal, linewidth=0.8, color='steelblue')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Chirp Signal (50 → 300 Hz)")

extent = stft.extent(len(chirp_noisy))
im = axes[1].imshow(
    10 * np.log10(Sx + 1e-10),
    aspect='auto',
    origin='lower',
    extent=extent,
    cmap='inferno',
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title("Spectrogram (ShortTimeFFT)")
axes[1].set_ylim([0, 400])
plt.colorbar(im, ax=axes[1], label="Power (dB)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scipy_spectrogram.png'), dpi=150)
print("Saved: scipy_spectrogram.png")
plt.close()

# =============================================================================
# Optimization: minimize scalar and multivariate functions
# =============================================================================
print("\n" + "=" * 60)
print("Optimization")
print("=" * 60)

# --- 1D: Rosenbrock-like curved valley
def rosenbrock_1d(x: np.ndarray) -> float:
    """Classic Rosenbrock banana function (2D)."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

result_nm = optimize.minimize(rosenbrock_1d, x0=[-1.0, 2.0], method='Nelder-Mead')
result_bfgs = optimize.minimize(rosenbrock_1d, x0=[-1.0, 2.0], method='BFGS',
                                 jac='2-point')

print("Rosenbrock minimization:")
print(f"  Nelder-Mead  → x={result_nm.x}, f={result_nm.fun:.6f}, iters={result_nm.nit}")
print(f"  BFGS         → x={result_bfgs.x}, f={result_bfgs.fun:.6e}, iters={result_bfgs.nit}")

# --- Curve fitting: fit a damped oscillation
def damped_oscillation(t: np.ndarray, amp: float, decay: float, freq: float, phi: float) -> np.ndarray:
    """Damped sinusoidal model."""
    return amp * np.exp(-decay * t) * np.cos(2 * np.pi * freq * t + phi)

# Generate data with known parameters + noise
true_params = (2.5, 0.8, 5.0, np.pi / 4)
t_fit = np.linspace(0, 3, 300)
y_fit = damped_oscillation(t_fit, *true_params) + rng.normal(scale=0.15, size=300)

# Fit the model
p0 = [2.0, 1.0, 4.5, 0.0]  # initial guess
popt, pcov = optimize.curve_fit(
    damped_oscillation, t_fit, y_fit, p0=p0, maxfev=5000
)
perr = np.sqrt(np.diag(pcov))

print("\nCurve fitting (damped oscillation):")
param_names = ['amplitude', 'decay', 'frequency', 'phase']
for name, true, fitted, err in zip(param_names, true_params, popt, perr):
    print(f"  {name:<12}: true={true:.3f}  fitted={fitted:.3f} ± {err:.3f}")

# --- Root finding
def transcendental(x: float) -> float:
    """x * cos(x) - sin(x) = 0"""
    return x * np.cos(x) - np.sin(x)

roots = [optimize.brentq(transcendental, a, b) for a, b in [(3, 5), (6, 8), (9, 12)]]
print(f"\nRoots of x·cos(x) − sin(x) = 0: {[f'{r:.5f}' for r in roots]}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Rosenbrock surface
x_grid = np.linspace(-2, 2, 200)
y_grid = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

axes[0].contourf(X, Y, np.log1p(Z), levels=40, cmap='viridis')
axes[0].plot(*result_nm.x, 'rs', markersize=10, label=f'Nelder-Mead: {result_nm.x}')
axes[0].plot(*result_bfgs.x, 'w^', markersize=10, label=f'BFGS: {result_bfgs.x}')
axes[0].set_xlabel("x₀")
axes[0].set_ylabel("x₁")
axes[0].set_title("Rosenbrock Function (log scale)")
axes[0].legend(fontsize=8)

# Curve fit result
axes[1].scatter(t_fit, y_fit, s=4, alpha=0.5, label='Data', color='steelblue')
axes[1].plot(t_fit, damped_oscillation(t_fit, *true_params), '--', linewidth=2,
             label='True model', color='green')
axes[1].plot(t_fit, damped_oscillation(t_fit, *popt), linewidth=2,
             label='Fitted model', color='red')
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Damped Oscillation Curve Fitting")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scipy_optimization.png'), dpi=150)
print("Saved: scipy_optimization.png")
plt.close()

# =============================================================================
# Statistics: Hypothesis Tests and Distributions
# =============================================================================
print("\n" + "=" * 60)
print("Statistical Analysis")
print("=" * 60)

# Generate two groups with slightly different distributions
group_a = rng.normal(loc=100, scale=15, size=80)
group_b = rng.normal(loc=108, scale=18, size=75)

# Normality tests
stat_a, p_a = stats.shapiro(group_a)
stat_b, p_b = stats.shapiro(group_b)
print(f"Shapiro-Wilk normality test:")
print(f"  Group A: W={stat_a:.4f}, p={p_a:.4f} → {'normal' if p_a > 0.05 else 'NOT normal'}")
print(f"  Group B: W={stat_b:.4f}, p={p_b:.4f} → {'normal' if p_b > 0.05 else 'NOT normal'}")

# Variance equality
levene_stat, levene_p = stats.levene(group_a, group_b)
equal_var = levene_p > 0.05
print(f"\nLevene variance test: F={levene_stat:.4f}, p={levene_p:.4f} → {'equal' if equal_var else 'unequal'} variances")

# t-test
t_stat, t_p = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
print(f"\nIndependent t-test: t={t_stat:.4f}, p={t_p:.4f}")
print(f"  Conclusion: {'Significant difference' if t_p < 0.05 else 'No significant difference'} (α=0.05)")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(group_a) - 1) * group_a.std()**2 + (len(group_b) - 1) * group_b.std()**2) /
                     (len(group_a) + len(group_b) - 2))
cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")

# Distribution fitting
print("\nFitting Normal and Student-t distributions to Group A:")
mu_fit, std_fit = stats.norm.fit(group_a)
df_fit, loc_fit, scale_fit = stats.t.fit(group_a)
print(f"  Normal: μ={mu_fit:.2f}, σ={std_fit:.2f}")
print(f"  Student-t: df={df_fit:.2f}, loc={loc_fit:.2f}, scale={scale_fit:.2f}")

# Pearson and Spearman correlation
x_corr = rng.normal(size=100)
y_corr = 0.7 * x_corr + rng.normal(scale=0.5, size=100)
pearson_r, pearson_p = stats.pearsonr(x_corr, y_corr)
spearman_r, spearman_p = stats.spearmanr(x_corr, y_corr)
print(f"\nCorrelation (n=100, true ρ≈0.7):")
print(f"  Pearson  r={pearson_r:.3f}, p={pearson_p:.2e}")
print(f"  Spearman r={spearman_r:.3f}, p={spearman_p:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Distribution comparison
x_range = np.linspace(
    min(group_a.min(), group_b.min()) - 10,
    max(group_a.max(), group_b.max()) + 10,
    300,
)
axes[0].hist(group_a, bins=20, alpha=0.6, density=True, label='Group A', color='steelblue')
axes[0].hist(group_b, bins=20, alpha=0.6, density=True, label='Group B', color='tomato')
axes[0].plot(x_range, stats.norm.pdf(x_range, group_a.mean(), group_a.std()),
             '--', linewidth=2, color='steelblue')
axes[0].plot(x_range, stats.norm.pdf(x_range, group_b.mean(), group_b.std()),
             '--', linewidth=2, color='tomato')
axes[0].set_title(f"Group Comparison (p={t_p:.3f})")
axes[0].set_xlabel("Value")
axes[0].legend()

# Q-Q plot for group A
(osm, osr), (slope, intercept, r) = stats.probplot(group_a, dist='norm')
axes[1].scatter(osm, osr, s=15, color='steelblue', alpha=0.7)
axes[1].plot(osm, slope * np.array(osm) + intercept, 'r--', linewidth=2)
axes[1].set_title(f"Q-Q Plot (Group A), r={r:.3f}")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles")

# Correlation scatter
axes[2].scatter(x_corr, y_corr, s=20, alpha=0.6, color='steelblue')
m, b = np.polyfit(x_corr, y_corr, 1)
axes[2].plot(x_corr, m * x_corr + b, 'r--', linewidth=2)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title(f"Correlation: Pearson r={pearson_r:.3f}")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scipy_statistics.png'), dpi=150)
print("Saved: scipy_statistics.png")
plt.close()

# =============================================================================
# Interpolation
# =============================================================================
print("\n" + "=" * 60)
print("Interpolation")
print("=" * 60)

x_sparse = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10], dtype=float)
y_sparse = np.sin(x_sparse) + rng.normal(scale=0.05, size=len(x_sparse))
x_dense = np.linspace(0, 10, 500)

linear_interp = interpolate.interp1d(x_sparse, y_sparse, kind='linear')
cubic_interp = interpolate.CubicSpline(x_sparse, y_sparse)
rbf_interp = interpolate.RBFInterpolator(x_sparse[:, None], y_sparse, kernel='thin_plate_spline')

print(f"Interpolating {len(x_sparse)} sparse points → {len(x_dense)} dense points")

for name, yi in [
    ('Linear', linear_interp(x_dense)),
    ('Cubic Spline', cubic_interp(x_dense)),
    ('RBF (thin plate)', rbf_interp(x_dense[:, None])),
]:
    rmse = np.sqrt(np.mean((yi - np.sin(x_dense))**2))
    print(f"  {name:<22}: RMSE vs sin(x) = {rmse:.5f}")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(x_dense, np.sin(x_dense), 'k-', linewidth=1.5, label='True sin(x)', alpha=0.5)
ax.scatter(x_sparse, y_sparse, s=60, zorder=5, color='black', label='Sparse data')
ax.plot(x_dense, linear_interp(x_dense), '--', linewidth=1.5, label='Linear', color='tomato')
ax.plot(x_dense, cubic_interp(x_dense), '-', linewidth=1.5, label='Cubic Spline', color='steelblue')
ax.plot(x_dense, rbf_interp(x_dense[:, None]), '-', linewidth=1.5, label='RBF', color='seagreen')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Interpolation Methods Comparison")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scipy_interpolation.png'), dpi=150)
print("Saved: scipy_interpolation.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SciPy signal processing and optimization complete!")
print("Outputs: scipy_fft.png, scipy_filters.png, scipy_spectrogram.png,")
print("         scipy_optimization.png, scipy_statistics.png,")
print("         scipy_interpolation.png")
print("=" * 60)
