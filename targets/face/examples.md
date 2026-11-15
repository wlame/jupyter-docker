<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:face`

Face detection, recognition, analysis, and generation.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [20_face_analysis](20_face_analysis.ipynb) | Face Analysis | `face` |
| [47_face_anonymization](47_face_anonymization.ipynb) | Anonymize Faces in a Photo | `face` |
| [54_face_recognition](54_face_recognition.ipynb) | Face Recognition by Similarity | `face` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
