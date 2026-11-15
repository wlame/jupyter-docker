<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:speech`

Speech recognition and text-to-speech synthesis.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [19_speech_processing](19_speech_processing.ipynb) | Speech Processing | `speech` |
| [31_speech_metrics_vad](31_speech_metrics_vad.ipynb) | Speech Evaluation and Segmentation: jiwer, silero-vad, parselmouth | `speech` |
| [48_speech_roundtrip](48_speech_roundtrip.ipynb) | Transcribe a Recording and Measure the Error Rate | `speech` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
