#!/usr/bin/env python3
# # Synthesize a Chord, Find Its Notes, and Rebuild It from a Spectrogram
#
# Builds an A-minor chord from three notes with harmonics and a natural volume
# envelope, listens to it, finds the notes again from the spectrum, shows the
# chroma (pitch-class) view musicians use, and then rebuilds audio from the mel
# spectrogram alone with Griffin-Lim, the step every neural vocoder improves on.
#
# The players below work in JupyterLab. To analyse your own recording, put a WAV
# file in `/home/jupyter/data/` and set `AUDIO_FILE`.
#
# - [librosa](https://librosa.org/doc/latest/)
# - [soundfile](https://python-soundfile.readthedocs.io/)

import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_FILE = None  # e.g. '/home/jupyter/data/guitar.wav'
SAMPLE_RATE = 22_050
DURATION = 3.0
CHORD = ['A3', 'C4', 'E4']  # A minor
HARMONIC_WEIGHTS = [1.0, 0.5, 0.25, 0.12]  # fundamental and three overtones

# ## Synthesize
#
# Each note is a sum of its fundamental frequency and quieter overtones; a fast
# attack and an exponential decay make it sound plucked rather than electronic.

t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
envelope = np.minimum(t / 0.01, 1.0) * np.exp(-1.2 * t)
if AUDIO_FILE:
    signal, SAMPLE_RATE = librosa.load(AUDIO_FILE, sr=None, mono=True)
else:
    signal = sum(
        weight * np.sin(2 * np.pi * (k + 1) * librosa.note_to_hz(note) * t)
        for note in CHORD
        for k, weight in enumerate(HARMONIC_WEIGHTS)
    ) * envelope
signal = (0.8 * signal / np.abs(signal).max()).astype(np.float32)
sf.write(os.path.join(OUTPUT_DIR, 'chord.wav'), signal, SAMPLE_RATE)
print(f"{len(signal) / SAMPLE_RATE:.1f} s at {SAMPLE_RATE} Hz: {', '.join(CHORD)} "
      f"({', '.join(f'{librosa.note_to_hz(n):.1f} Hz' for n in CHORD)})")
display(Audio(signal, rate=SAMPLE_RATE))

# ## Find the notes again
#
# The average magnitude spectrum has a peak at every partial. Keeping strong
# peaks and naming them gives back the chord's notes (plus their overtones).

spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
frequencies = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
peak_bins = librosa.util.peak_pick(spectrum, pre_max=50, post_max=50, pre_avg=50, post_avg=50, delta=spectrum.max() * 0.05,
                                   wait=50)
peaks = sorted(frequencies[peak_bins][np.argsort(spectrum[peak_bins])[::-1][:6]])
detected = sorted({librosa.hz_to_note(f) for f in peaks}, key=librosa.note_to_hz)
print("Strongest peaks: " + ", ".join(f"{f:.1f} Hz ({librosa.hz_to_note(f)})" for f in peaks))
missing = [note for note in CHORD if note not in detected]
print(f"Chord notes found: {[n for n in CHORD if n in detected]}" + (f"; missing {missing}" if missing else ""))

# ## Waveform, spectrogram, and chroma

stft_db = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
chroma = librosa.feature.chroma_stft(y=signal, sr=SAMPLE_RATE)
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
librosa.display.waveshow(signal, sr=SAMPLE_RATE, ax=axes[0])
axes[0].set_title('Waveform')
image = librosa.display.specshow(stft_db, sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=axes[1])
axes[1].set(title='Spectrogram (log frequency)', ylim=(60, 4000))
fig.colorbar(image, ax=axes[1], format='%+2.0f dB')
librosa.display.specshow(chroma, sr=SAMPLE_RATE, x_axis='time', y_axis='chroma', ax=axes[2])
axes[2].set_title('Chroma: energy per pitch class (A, C, and E light up)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chord_analysis.png'), dpi=110)
plt.show()

# ## Rebuild audio from the mel spectrogram
#
# A mel spectrogram keeps the energy in perceptual frequency bands but throws
# away phase. Griffin-Lim estimates a phase that fits, so the rebuilt chord sounds
# right but slightly "phasey". Comparing spectra measures how close it is.

mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=128)
rebuilt = librosa.feature.inverse.mel_to_audio(mel, sr=SAMPLE_RATE, n_iter=64)
rebuilt = rebuilt[:len(signal)]
sf.write(os.path.join(OUTPUT_DIR, 'chord_rebuilt.wav'), rebuilt, SAMPLE_RATE)
original_mel = librosa.power_to_db(mel)
rebuilt_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=rebuilt, sr=SAMPLE_RATE, n_mels=128))
correlation = float(np.corrcoef(original_mel.ravel(), rebuilt_mel[:, :original_mel.shape[1]].ravel())[0, 1])
print(f"Mel-spectrogram correlation, original vs rebuilt: {correlation:.3f}")
display(Audio(rebuilt, rate=SAMPLE_RATE))

with open(os.path.join(OUTPUT_DIR, 'chord_summary.json'), 'w') as f:
    json.dump({'chord': CHORD, 'detected_notes': detected, 'peaks_hz': [round(float(p), 1) for p in peaks],
               'rebuild_correlation': correlation}, f, indent=2)
print("Saved: chord.wav, chord_rebuilt.wav, chord_analysis.png, chord_summary.json")
