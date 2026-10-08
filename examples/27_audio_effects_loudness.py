#!/usr/bin/env python3
"""
Audio Effects, Loudness, Augmentation, and Phonetics
====================================================
Processes a synthetic voice-like signal with Spotify's pedalboard (studio
effects, augmentation for audio ML, and audio file I/O), measures and
normalizes loudness to a broadcast target with pyloudnorm, removes noise with
noisereduce, and measures pitch and formants with Praat through parselmouth.

pedalboard:      https://spotify.github.io/pedalboard/
pyloudnorm:      https://github.com/csteinmetz1/pyloudnorm
noisereduce:     https://github.com/timsainb/noisereduce
parselmouth:     https://parselmouth.readthedocs.io/
"""

import os

import matplotlib
import noisereduce as nr
import numpy as np
import parselmouth
import pyloudnorm as pyln
from pedalboard import Chorus, Compressor, Gain, HighpassFilter, Pedalboard, PitchShift, Reverb, time_stretch
from pedalboard.io import AudioFile

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)
sr = 16_000

# =============================================================================
# A synthetic vowel: glottal pulse train shaped by two formant resonances
# =============================================================================
print("=" * 60)
print("Synthetic Vowel")
print("=" * 60)

duration = 2.0
t = np.arange(int(sr * duration)) / sr
f0 = 140 + 15 * np.sin(2 * np.pi * 1.5 * t)          # gently varying pitch
phase = 2 * np.pi * np.cumsum(f0) / sr
source = np.sign(np.sin(phase)) * 0.3                # buzzy glottal-like source
vowel = np.zeros_like(source)
for formant, bandwidth in [(700, 110), (1220, 120)]:  # roughly an /a/
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2 * np.pi * formant / sr
    a1, a2 = -2 * r * np.cos(theta), r * r
    y = np.zeros_like(source)
    for i in range(2, len(source)):
        y[i] = source[i] - a1 * y[i - 1] - a2 * y[i - 2]
    vowel += y
vowel = (vowel / np.max(np.abs(vowel)) * 0.5).astype(np.float32)
print(f"{duration:.1f} s at {sr} Hz, mean pitch {f0.mean():.1f} Hz")

# =============================================================================
# pedalboard — effects chain and file I/O
# =============================================================================
print("\n" + "=" * 60)
print("pedalboard: Effects Chain")
print("=" * 60)

board = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=80),
    Compressor(threshold_db=-18, ratio=3),
    Chorus(rate_hz=0.8, depth=0.15),
    Reverb(room_size=0.3, wet_level=0.2),
    Gain(gain_db=-3),
])
processed = board(vowel, sr)
fx_path = os.path.join(OUTPUT_DIR, 'pedalboard_fx.wav')
with AudioFile(fx_path, 'w', sr, num_channels=1) as f:
    f.write(processed)
with AudioFile(fx_path) as f:
    print(f"Wrote {f.duration:.2f} s, {f.num_channels} channel(s) at {f.samplerate} Hz -> pedalboard_fx.wav")

# =============================================================================
# pyloudnorm — integrated loudness (ITU-R BS.1770)
# =============================================================================
print("\n" + "=" * 60)
print("pyloudnorm: Loudness Normalization")
print("=" * 60)

meter = pyln.Meter(sr)
loudness = meter.integrated_loudness(processed.astype(np.float64))
normalized = pyln.normalize.loudness(processed.astype(np.float64), loudness, -23.0)
print(f"Integrated loudness {loudness:.1f} LUFS -> {meter.integrated_loudness(normalized):.1f} LUFS (EBU R128 target -23)")

# =============================================================================
# pedalboard — randomized augmentations for training data
# =============================================================================
print("\n" + "=" * 60)
print("pedalboard: Augmentations")
print("=" * 60)

for i in range(3):
    semitones = float(rng.uniform(-2, 2))
    stretch = float(rng.uniform(0.9, 1.1))
    shifted = Pedalboard([PitchShift(semitones=semitones)])(vowel, sr)
    stretched = time_stretch(shifted, sr, stretch_factor=stretch)[0]
    noisy_aug = stretched + rng.normal(scale=0.005, size=stretched.shape).astype(np.float32)
    print(f"variant {i}: pitch {semitones:+.2f} st, speed x{stretch:.2f} -> {len(noisy_aug)} samples")

# =============================================================================
# noisereduce — spectral gating
# =============================================================================
print("\n" + "=" * 60)
print("noisereduce: Denoise")
print("=" * 60)

noisy = vowel + rng.normal(scale=0.05, size=vowel.shape).astype(np.float32)
denoised = nr.reduce_noise(y=noisy, sr=sr)


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-noise ratio of `test` against the clean reference, in dB."""
    return float(10 * np.log10(np.sum(clean**2) / np.sum((clean - test) ** 2)))


print(f"SNR noisy {snr_db(vowel, noisy):.1f} dB -> denoised {snr_db(vowel, denoised):.1f} dB")

# =============================================================================
# parselmouth — Praat pitch and formants
# =============================================================================
print("\n" + "=" * 60)
print("parselmouth: Pitch and Formants")
print("=" * 60)

sound = parselmouth.Sound(vowel.astype(np.float64), sampling_frequency=sr)
pitch = sound.to_pitch()
pitch_values = pitch.selected_array['frequency']
voiced = pitch_values[pitch_values > 0]
formants = sound.to_formant_burg()
midpoint = duration / 2
f1, f2 = formants.get_value_at_time(1, midpoint), formants.get_value_at_time(2, midpoint)
print(f"Praat pitch: median {np.median(voiced):.1f} Hz over {len(voiced)} voiced frames")
print(f"Formants at {midpoint:.1f} s: F1 {f1:.0f} Hz, F2 {f2:.0f} Hz (synthesized at 700 / 1220 Hz)")

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(t, noisy, lw=0.4, label='noisy', alpha=0.6)
axes[0].plot(t, denoised, lw=0.4, label='denoised')
axes[0].legend(loc='upper right')
axes[0].set_ylabel('amplitude')
axes[1].plot(pitch.xs(), np.where(pitch_values > 0, pitch_values, np.nan), color='tab:red')
axes[1].set_ylabel('pitch (Hz)')
axes[1].set_xlabel('time (s)')
fig.suptitle('Denoising and Praat pitch track')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_effects.png'), dpi=120)
plt.close()
print("Saved: audio_effects.png")

print("\nDone.")
