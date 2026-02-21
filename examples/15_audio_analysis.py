#!/usr/bin/env python3
"""
Audio Analysis
==============
Demonstrates audio processing and feature extraction with librosa and torchaudio.

librosa: https://librosa.org/
torchaudio: https://pytorch.org/audio/
soundfile: https://python-soundfile.readthedocs.io/

Uses librosa's built-in audio samples — no external files required.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load Audio — librosa built-in sample
# =============================================================================
print("=" * 60)
print("Loading Audio")
print("=" * 60)

# librosa ships with several example audio files
y, sr = librosa.load(librosa.ex('nutcracker'), duration=30.0)

print(f"Audio shape : {y.shape}")
print(f"Sample rate : {sr} Hz")
print(f"Duration    : {len(y) / sr:.2f} s")
print(f"Min / Max   : {y.min():.4f} / {y.max():.4f}")

# Save a clip as WAV for use by other tools
clip_path = os.path.join(OUTPUT_DIR, 'audio_clip.wav')
sf.write(clip_path, y, sr)
print(f"Saved clip  : {clip_path}")

# =============================================================================
# Waveform and Basic Analysis
# =============================================================================
print("\n" + "=" * 60)
print("Waveform and Zero-Crossing Rate")
print("=" * 60)

zcr = librosa.feature.zero_crossing_rate(y)
rms = librosa.feature.rms(y=y)

print(f"Zero-Crossing Rate — mean: {zcr.mean():.4f}, std: {zcr.std():.4f}")
print(f"RMS Energy         — mean: {rms.mean():.4f}, std: {rms.std():.4f}")

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Waveform
times = np.linspace(0, len(y) / sr, len(y))
axes[0].plot(times, y, linewidth=0.4, color='steelblue')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Waveform")

# ZCR
zcr_times = librosa.times_like(zcr, sr=sr)
axes[1].semilogy(zcr_times, zcr[0], color='darkorange', linewidth=0.8)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("ZCR")
axes[1].set_title("Zero-Crossing Rate")

# RMS
rms_times = librosa.times_like(rms, sr=sr)
axes[2].plot(rms_times, rms[0], color='seagreen', linewidth=0.8)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Energy")
axes[2].set_title("RMS Energy")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_waveform.png'), dpi=150)
print("Saved: audio_waveform.png")
plt.close()

# =============================================================================
# Harmonic-Percussive Source Separation (HPSS)
# =============================================================================
print("\n" + "=" * 60)
print("Harmonic-Percussive Source Separation (HPSS)")
print("=" * 60)

y_harmonic, y_percussive = librosa.effects.hpss(y)

print(f"Harmonic energy  : {np.sum(y_harmonic**2):.2f}")
print(f"Percussive energy: {np.sum(y_percussive**2):.2f}")

fig, axes = plt.subplots(3, 1, figsize=(12, 7))
for ax, signal, title, color in zip(
    axes,
    [y, y_harmonic, y_percussive],
    ['Original', 'Harmonic Component', 'Percussive Component'],
    ['steelblue', 'royalblue', 'tomato'],
):
    ax.plot(np.linspace(0, len(signal) / sr, len(signal)), signal, linewidth=0.3, color=color)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_hpss.png'), dpi=150)
print("Saved: audio_hpss.png")
plt.close()

# =============================================================================
# Beat Tracking and Tempo Estimation
# =============================================================================
print("\n" + "=" * 60)
print("Beat Tracking and Tempo Estimation")
print("=" * 60)

# Use percussive component for more stable beat detection
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# tempo may be an array in newer librosa versions
tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
print(f"Estimated tempo : {tempo_val:.1f} BPM")
print(f"Number of beats : {len(beat_times)}")
print(f"Beat interval   : {np.mean(np.diff(beat_times)):.3f} s ({60/tempo_val:.3f} s expected)")

# Plot beats on waveform
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, y, linewidth=0.4, color='steelblue', alpha=0.7, label='Waveform')
for bt in beat_times:
    ax.axvline(x=bt, color='red', alpha=0.5, linewidth=0.7)
ax.axvline(x=beat_times[0], color='red', alpha=0.8, linewidth=0.7, label='Beat')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Beat Tracking — {tempo_val:.1f} BPM")
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_beats.png'), dpi=150)
print("Saved: audio_beats.png")
plt.close()

# =============================================================================
# Spectral Features: Mel Spectrogram, MFCC, Chroma
# =============================================================================
print("\n" + "=" * 60)
print("Spectral Features")
print("=" * 60)

hop_length = 512
n_mfcc = 13

# Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)

# MFCCs (from harmonic component for cleaner tonal features)
mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
mfcc_delta = librosa.feature.delta(mfcc)

# Chroma (from harmonic component)
chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)

print(f"Mel spectrogram : {mel_spec.shape} (mels x frames)")
print(f"MFCC            : {mfcc.shape} ({n_mfcc} coefficients x frames)")
print(f"Delta MFCC      : {mfcc_delta.shape}")
print(f"Chroma          : {chroma.shape} (12 pitch classes x frames)")

# Spectral centroid and bandwidth
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)

print(f"\nSpectral centroid  — mean: {centroid.mean():.1f} Hz")
print(f"Spectral bandwidth — mean: {bandwidth.mean():.1f} Hz")
print(f"Spectral rolloff   — mean: {rolloff.mean():.1f} Hz")

# Four-panel spectrogram figure
fig, axes = plt.subplots(4, 1, figsize=(13, 14))

img0 = librosa.display.specshow(
    mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[0]
)
axes[0].set_title("Mel Spectrogram (dB)")
fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

img1 = librosa.display.specshow(
    mfcc, sr=sr, hop_length=hop_length, x_axis='time', ax=axes[1]
)
axes[1].set_title(f"MFCCs (first {n_mfcc} coefficients)")
axes[1].set_ylabel("MFCC coefficient")
fig.colorbar(img1, ax=axes[1])

img2 = librosa.display.specshow(
    chroma, sr=sr, hop_length=hop_length, x_axis='time', y_axis='chroma', ax=axes[2]
)
axes[2].set_title("Chroma Features (CQT) — Pitch Class Energy")
fig.colorbar(img2, ax=axes[2])

feat_times = librosa.times_like(centroid, sr=sr, hop_length=hop_length)
axes[3].plot(feat_times, centroid[0], label='Centroid', linewidth=0.8, color='steelblue')
axes[3].plot(feat_times, rolloff[0], label='Rolloff (85%)', linewidth=0.8, color='darkorange')
axes[3].fill_between(
    feat_times,
    centroid[0] - bandwidth[0],
    centroid[0] + bandwidth[0],
    alpha=0.25,
    color='steelblue',
    label='±Bandwidth',
)
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Frequency (Hz)")
axes[3].set_title("Spectral Centroid, Bandwidth, and Rolloff")
axes[3].legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_spectral_features.png'), dpi=150)
print("Saved: audio_spectral_features.png")
plt.close()

# =============================================================================
# MFCC Statistics — Beat-Synchronised Feature Matrix
# =============================================================================
print("\n" + "=" * 60)
print("Beat-Synchronised Feature Matrix")
print("=" * 60)

# Aggregate features at each beat interval — useful for music classification
beat_mfcc = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
beat_features = np.vstack([beat_chroma, beat_mfcc])

print(f"Beat-sync chroma : {beat_chroma.shape}")
print(f"Beat-sync MFCC   : {beat_mfcc.shape}")
print(f"Combined features: {beat_features.shape}  (25 features x {beat_features.shape[1]} beats)")

fig, ax = plt.subplots(figsize=(12, 5))
img = ax.imshow(beat_features, aspect='auto', origin='lower', cmap='coolwarm')
ax.set_xlabel("Beat Number")
ax.set_yticks(range(25))
ax.set_yticklabels(
    [f"C{i}" for i in range(12)] + [f"MFCC{i}" for i in range(13)],
    fontsize=7,
)
ax.set_title("Beat-Synchronised Feature Matrix (Chroma + MFCC)")
fig.colorbar(img, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_beat_features.png'), dpi=150)
print("Saved: audio_beat_features.png")
plt.close()

# =============================================================================
# torchaudio: Transforms and Resampling
# =============================================================================
print("\n" + "=" * 60)
print("torchaudio: Spectrogram and Resampling")
print("=" * 60)

# Convert to tensor
waveform = torch.tensor(y).unsqueeze(0).float()  # shape: (1, T)
print(f"Waveform tensor : {waveform.shape}, dtype={waveform.dtype}")

# Resample to 22050 → 16000 Hz (common for speech models)
target_sr = 16000
resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
waveform_16k = resampler(waveform)
print(f"Resampled       : {waveform.shape[-1]} → {waveform_16k.shape[-1]} samples")
print(f"Duration check  : {waveform_16k.shape[-1] / target_sr:.2f} s")

# Spectrogram via torchaudio
spectrogram_transform = T.Spectrogram(n_fft=1024, hop_length=256, power=2.0)
spec = spectrogram_transform(waveform)
print(f"Spectrogram     : {spec.shape}  (channels x freq_bins x frames)")

# Mel Spectrogram via torchaudio
mel_transform = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)
mel_spec_torch = mel_transform(waveform)
print(f"Mel Spectrogram : {mel_spec_torch.shape}  (channels x mel_bins x frames)")

# AmplitudeToDB conversion
to_db = T.AmplitudeToDB(stype='power', top_db=80)
mel_db_torch = to_db(mel_spec_torch)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

spec_db_torch = to_db(spec)
axes[0].imshow(
    spec_db_torch[0].numpy(),
    aspect='auto',
    origin='lower',
    cmap='magma',
    extent=[0, len(y) / sr, 0, sr / 2],
)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("torchaudio Spectrogram (dB)")

axes[1].imshow(
    mel_db_torch[0].numpy(),
    aspect='auto',
    origin='lower',
    cmap='magma',
    extent=[0, len(y) / sr, 0, 80],
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Mel Band")
axes[1].set_title("torchaudio Mel Spectrogram (dB, 80 mel bands)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_torchaudio.png'), dpi=150)
print("Saved: audio_torchaudio.png")
plt.close()

# =============================================================================
# Summary Statistics Table
# =============================================================================
print("\n" + "=" * 60)
print("Summary: Extracted Audio Features")
print("=" * 60)

feature_stats = {
    'Tempo (BPM)': tempo_val,
    'Duration (s)': len(y) / sr,
    'Sample Rate (Hz)': sr,
    'ZCR mean': float(zcr.mean()),
    'RMS mean': float(rms.mean()),
    'Spectral centroid mean (Hz)': float(centroid.mean()),
    'Spectral bandwidth mean (Hz)': float(bandwidth.mean()),
    f'MFCC[0] mean': float(mfcc[0].mean()),
    f'MFCC[1] mean': float(mfcc[1].mean()),
    'Chroma max pitch class': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][int(chroma.mean(axis=1).argmax())],
}

for name, val in feature_stats.items():
    if isinstance(val, float):
        print(f"  {name:<40}: {val:.4f}")
    else:
        print(f"  {name:<40}: {val}")

print("\n" + "=" * 60)
print("Audio analysis complete!")
print("Outputs: audio_waveform.png, audio_hpss.png, audio_beats.png,")
print("         audio_spectral_features.png, audio_beat_features.png,")
print("         audio_torchaudio.png, audio_clip.wav")
print("=" * 60)
