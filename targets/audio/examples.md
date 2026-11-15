<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:audio`

Audio processing and analysis.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [15_audio_analysis](15_audio_analysis.ipynb) | Audio Analysis | `audio` |
| [27_audio_effects_loudness](27_audio_effects_loudness.ipynb) | Audio Effects, Loudness, Augmentation, and Phonetics | `audio` |
| [45_chord_synthesis](45_chord_synthesis.ipynb) | Synthesize a Chord, Find Its Notes, and Rebuild It from a Spectrogram | `audio` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
