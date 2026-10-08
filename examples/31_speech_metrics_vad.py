#!/usr/bin/env python3
"""
Speech Evaluation and Segmentation: jiwer, silero-vad, parselmouth
==================================================================
Scores speech-recognition output with jiwer (word and character error rates,
with the alignment behind them), finds voiced segments with the silero-vad
model (bundled in the package, no download), and measures voice quality with
Praat's harmonics-to-noise ratio through parselmouth.

jiwer:       https://jitsi.github.io/jiwer/
silero-vad:  https://github.com/snakers4/silero-vad
parselmouth: https://parselmouth.readthedocs.io/
"""

import json
import os

import jiwer
import numpy as np
import parselmouth
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)
results: dict[str, object] = {}

# =============================================================================
# jiwer — word and character error rates
# =============================================================================
print("=" * 60)
print("jiwer: Recognition Error Rates")
print("=" * 60)

references = [
    'the quick brown fox jumps over the lazy dog',
    'speech recognition is improving every year',
]
hypotheses = [
    'the quick brown fox jumped over a lazy dog',
    'speech recognition is improving each year',
]
output = jiwer.process_words(references, hypotheses)
print(f"WER {output.wer:.3f}  (substitutions {output.substitutions}, deletions {output.deletions}, "
      f"insertions {output.insertions})")
print(f"CER {jiwer.cer(references, hypotheses):.3f}")
print(jiwer.visualize_alignment(output, show_measures=False))
results['wer'] = round(output.wer, 4)

# =============================================================================
# silero-vad — voice activity detection
# =============================================================================
print("=" * 60)
print("silero-vad: Voiced Segments")
print("=" * 60)

sr = 16_000
t = np.arange(int(sr * 0.8)) / sr
f0 = 120 + 20 * np.sin(2 * np.pi * 2 * t)
voiced = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / sr)) * np.hanning(len(t)) * 0.4
pause = rng.normal(scale=0.002, size=int(sr * 0.6))
signal = np.concatenate([pause, voiced, pause, voiced, pause]).astype(np.float32)

model = load_silero_vad()
segments = get_speech_timestamps(torch.from_numpy(signal), model, sampling_rate=sr, return_seconds=True)
print(f"Signal: {len(signal) / sr:.1f} s; detected segments: {segments or 'none (synthetic buzz is not speech-like enough)'}")
results['vad_segments'] = segments

# =============================================================================
# parselmouth — harmonics-to-noise ratio
# =============================================================================
print("\n" + "=" * 60)
print("parselmouth: Voice Quality")
print("=" * 60)

clean = parselmouth.Sound(voiced, sampling_frequency=sr)
breathy = parselmouth.Sound(voiced + rng.normal(scale=0.05, size=voiced.shape), sampling_frequency=sr)
for label, sound in (('clean', clean), ('breathy', breathy)):
    harmonicity = sound.to_harmonicity()
    hnr = float(np.mean(harmonicity.values[harmonicity.values != -200]))
    results[f'hnr_{label}_db'] = round(hnr, 2)
    print(f"{label:8} harmonics-to-noise ratio: {hnr:.1f} dB")

with open(os.path.join(OUTPUT_DIR, 'speech_quality.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: speech_quality.json")
print("Done.")
