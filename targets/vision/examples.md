<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:vision`

Computer vision and image processing.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [12_image_processing](12_image_processing.ipynb) | Image Processing Basics | `vision` |
| [13_object_detection_yolo](13_object_detection_yolo.ipynb) | Object Detection with Ultralytics YOLO | `vision` |
| [26_vision_backbones_kornia](26_vision_backbones_kornia.ipynb) | Vision Building Blocks: timm, kornia, supervision, OpenCLIP, onnxruntime | `vision` |
| [43_coins_and_superpixels](43_coins_and_superpixels.ipynb) | Counting Coins and Segmenting a Photo | `vision` |
| [50_reliable_api_client](50_reliable_api_client.ipynb) | Calling an API Reliably: Retries, Validation, and Logging | `base` |
| [51_parse_web_and_config](51_parse_web_and_config.ipynb) | Parsing the Formats Data Arrives In: HTML, XML, YAML, .env, and Dates | `base` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
