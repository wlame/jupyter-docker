#!/usr/bin/env python3
# # Transcribe a Recording and Measure the Error Rate
#
# The full speech-recognition loop, offline: make a recording with a known
# script (espeak-ng, a small text-to-speech engine installed in this image),
# transcribe it with Whisper, line up the transcript with the script, and score
# it with the word and character error rates (WER, CER). Detected speech
# segments and the transcript's timestamps are plotted over the waveform.
#
# To transcribe your own recording, put a WAV file in `/home/jupyter/data/` and
# set `AUDIO_FILE` (and `REFERENCE` to its true text if you have it).
#
# - [Whisper](https://github.com/openai/whisper)
# - [jiwer](https://jitsi.github.io/jiwer/)
# - [espeak-ng](https://github.com/espeak-ng/espeak-ng)

import json
import os
import shutil
import subprocess

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import whisper

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_FILE = None  # e.g. '/home/jupyter/data/meeting.wav'
REFERENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Speech recognition turns spoken words into text, and the error rate tells us how well it worked."
)
MODEL = 'tiny'  # baked into the image; 'base' or 'small' are more accurate but download on first use

# ## Make or load the recording

if AUDIO_FILE:
    audio_path = AUDIO_FILE
elif shutil.which('espeak-ng'):
    audio_path = os.path.join(OUTPUT_DIR, 'speech_reference.wav')
    subprocess.run(['espeak-ng', '-s', '150', '-w', audio_path, REFERENCE], check=True)
else:
    raise SystemExit("espeak-ng is not installed here; set AUDIO_FILE to a WAV file instead")
signal, rate = sf.read(audio_path, always_2d=False)
signal = signal.mean(axis=1) if signal.ndim > 1 else signal
print(f"{audio_path}: {len(signal) / rate:.1f} s at {rate} Hz")

# ## Transcribe with Whisper
#
# Whisper resamples to 16 kHz itself. `word_timestamps=True` adds the start and
# end time of every word.

model = whisper.load_model(MODEL)
result = model.transcribe(audio_path, language='en', fp16=False, word_timestamps=True)
hypothesis = result['text'].strip()
print(f"Reference : {REFERENCE}")
print(f"Transcript: {hypothesis}")

# ## Score it
#
# WER counts the substituted, deleted, and inserted words needed to turn the
# transcript into the reference, divided by the reference length. Normalizing
# case and punctuation first keeps the score about words, not formatting.

normalize = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                           jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
words = jiwer.process_words(REFERENCE, hypothesis, reference_transform=normalize, hypothesis_transform=normalize)
cer = jiwer.cer(REFERENCE.lower(), hypothesis.lower())
print(f"WER {words.wer:.1%} (substitutions {words.substitutions}, deletions {words.deletions}, "
      f"insertions {words.insertions}); CER {cer:.1%}")
print(jiwer.visualize_alignment(words, show_measures=False))

# ## Where the words are
#
# Word timestamps from Whisper, drawn over the waveform.

timeline = [(w['start'], w['end'], w['word'].strip()) for seg in result['segments'] for w in seg.get('words', [])]
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(np.arange(len(signal)) / rate, signal, lw=0.4, color='#4c72b0')
for i, (start, end, word) in enumerate(timeline):
    ax.axvspan(start, end, color='#dd8452', alpha=0.15)
    ax.text((start + end) / 2, 0.9 * np.abs(signal).max() * (1 if i % 2 else -1), word, ha='center', fontsize=7)
ax.set(xlabel='seconds', title=f'Whisper {MODEL}: WER {words.wer:.0%}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'speech_roundtrip.png'), dpi=110)
plt.show()

with open(os.path.join(OUTPUT_DIR, 'speech_roundtrip.json'), 'w') as f:
    json.dump({'reference': REFERENCE, 'transcript': hypothesis, 'wer': words.wer, 'cer': cer,
               'words': [{'start': s, 'end': e, 'word': w} for s, e, w in timeline]}, f, indent=2)
print("Saved: speech_roundtrip.png, speech_roundtrip.json")
