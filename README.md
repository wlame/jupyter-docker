# Data Science Jupyter Notebook Environment

A modular, multi-target Docker environment for data science with Python 3.13. Build only what you need - from a lightweight base image to a comprehensive full environment.

**Package Manager**: [uv](https://docs.astral.sh/uv/) - Fast Python package manager from Astral

## Quick Start

### Use Prebuilt Images

Prebuilt images are published to GitHub Container Registry on every push to `main`
and rebuilt weekly to pick up OS security patches. Every push also publishes an
immutable `<target>-<short-sha>` tag for pinning and rollback:

```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  ghcr.io/wlame/jupyter-docker:scientific
```

See [Available Targets](#available-targets) for the full list. Replace `scientific` with any target name.

### Build from Source

```bash
# Build only what you need
docker build --target base -t ds-base .
docker build --target scientific -t ds-scientific .
docker build --target ml -t ds-ml .
docker build --target full -t ds-full .
```

### Run the Container

```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  ds-scientific
```

Access Jupyter Lab at: **http://localhost:8888**

## Available Targets

| Target | Size | Description | Inherits From |
|--------|------|-------------|---------------|
| `base` | ~500MB | Common utilities (JSON, HTTP, dates) | Ubuntu 24.04 |
| `scientific` | ~1.2GB | NumPy, SciPy, Pandas, Statsmodels | base |
| `visualization` | ~900MB | Matplotlib, Plotly, Bokeh, Altair | base |
| `dataio` | ~800MB | Parquet, HDF5, Excel, SQL | base |
| `ml` | ~2GB | Scikit-learn, XGBoost, LightGBM | scientific |
| `deeplearn` | ~6GB | PyTorch, TensorFlow, Keras | ml |
| `vision` | ~2GB | OpenCV, Pillow, YOLO | base |
| `audio` | ~3GB | Librosa, TorchAudio, soundfile | base |
| `geospatial` | ~2GB | Cartopy, GeoPandas, Folium | scientific |
| `timeseries` | ~2GB | tsfresh, sktime, Prophet | scientific |
| `nlp` | ~4GB | spaCy, Transformers, NLTK | base |
| `speech` | ~4GB | Whisper, gTTS, SpeechBrain | base |
| `face` | ~7GB | DeepFace, dlib, face-alignment | base |
| `full` | ~14GB | Everything combined | standalone |

### Target Inheritance Tree

```
base
├── scientific
│   ├── ml
│   │   └── deeplearn
│   ├── geospatial
│   └── timeseries
├── visualization
├── dataio
├── vision
├── audio
├── nlp
├── speech
└── face

full (standalone - includes all)
```

## Build All Targets

Use the build script to build and test all targets:

```bash
# Build and test all targets
./build-all.sh

# Build only (no tests)
./build-all.sh --build-only

# Test existing images
./build-all.sh --test-only

# Build specific targets
./build-all.sh base scientific ml
```

## Target Details

### Base (Common Utilities)

Essential utilities included in all specialized targets.

| Library | Description |
|---------|-------------|
| IPython, Jupyter, JupyterLab | Interactive computing |
| orjson, ujson, simplejson | JSON processing |
| lxml, xmltodict, BeautifulSoup4 | XML/HTML parsing |
| PyYAML | YAML processing |
| requests, httpx, aiohttp | HTTP clients |
| Pydantic | Data validation |
| tqdm, loguru | Progress bars, logging |
| python-dotenv | .env file loading |
| python-dateutil, pytz, pendulum | Date/time utilities |
| joblib, toolz, more-itertools | Utilities |

### Scientific (Numerical Computing)

Core libraries for numerical and statistical computing.

| Library | Description |
|---------|-------------|
| NumPy | N-dimensional arrays |
| SciPy | Scientific algorithms |
| Pandas | DataFrames |
| Statsmodels | Statistical models |
| SymPy | Symbolic mathematics |

### Visualization (Charts & Dashboards)

Interactive and static visualization libraries.

| Library | Description |
|---------|-------------|
| Matplotlib | 2D/3D plotting |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Bokeh | Web-based visualization |
| HoloViews, hvPlot | Declarative visualization |
| Panel | Dashboards |
| Altair | Declarative statistical viz |

### DataIO (Data Formats & Databases)

Read and write various data formats.

| Library | Description |
|---------|-------------|
| PyArrow | Apache Arrow columnar data |
| fastparquet | Parquet format |
| h5py, PyTables | HDF5 format |
| openpyxl, xlrd | Excel files |
| SQLAlchemy | Database ORM |

### ML (Machine Learning)

Classical machine learning algorithms.

| Library | Description |
|---------|-------------|
| Scikit-learn | ML algorithms |
| XGBoost | Gradient boosting |
| LightGBM | Fast gradient boosting |
| imbalanced-learn | Imbalanced datasets |
| Optuna | Hyperparameter optimization |

### DeepLearn (Neural Networks)

Deep learning frameworks.

| Library | Description |
|---------|-------------|
| PyTorch | Dynamic neural networks |
| TorchVision | Computer vision for PyTorch |
| TorchAudio | Audio for PyTorch |
| TensorFlow | ML platform |
| Keras | High-level neural network API |

### Vision (Image Processing)

Computer vision and image manipulation.

| Library | Description |
|---------|-------------|
| Pillow | Image processing |
| OpenCV (headless) | Computer vision |
| scikit-image | Image algorithms |
| imageio | Image I/O |
| Ultralytics | YOLOv8 object detection |

### Audio (Audio Processing)

Audio analysis and manipulation.

| Library | Description |
|---------|-------------|
| TorchAudio | Audio for PyTorch |
| torchcodec | Audio/video decoding for torchaudio I/O |
| librosa | Music/audio analysis |
| soundfile | Audio file I/O |
| pydub | Audio manipulation |
| audioread | Audio decoding |

### Geospatial (Maps & GIS)

Geographic data processing and visualization.

| Library | Description |
|---------|-------------|
| Cartopy | Map projections |
| GeoPandas | Geospatial DataFrames |
| Shapely | Geometric operations |
| PyProj | Coordinate transformations |
| Folium | Interactive maps |
| GeoViews | Geographic visualization |

### TimeSeries (Time Series Analysis)

Time series modeling and forecasting.

| Library | Description |
|---------|-------------|
| tsfresh | Feature extraction |
| sktime | Time series ML |
| pmdarima | Auto-ARIMA |
| Prophet | Forecasting |

### NLP (Natural Language Processing)

Text processing and language models.

| Library | Description |
|---------|-------------|
| spaCy | Industrial NLP |
| NLTK | Classic NLP toolkit |
| Transformers | Hugging Face models |
| sentence-transformers | Sentence embeddings |
| tokenizers | Fast tokenization |

### Speech (Speech Recognition & TTS)

Speech-to-text and text-to-speech.

| Library | Description |
|---------|-------------|
| openai-whisper | Best overall ASR |
| faster-whisper | 4x faster ASR (CTranslate2) |
| SpeechRecognition | Lightweight ASR API wrapper |
| coqui-tts | Mature TTS engine |
| gTTS | Google Text-to-Speech |
| piper-tts | ONNX-based CPU-friendly TTS |
| pyannote-audio | Speaker diarization |
| speechbrain | All-in-one speech toolkit |
| torchcodec | Audio decoding for torchaudio I/O |

### Face (Face Detection & Recognition)

Face detection, recognition, analysis, and generation.

| Library | Description |
|---------|-------------|
| DeepFace | Recognition + attribute analysis |
| dlib | Face detection, 68-point landmarks |
| MTCNN | TensorFlow face detection |
| RetinaFace | Face detection with landmarks |
| face-alignment | 2D/3D face landmarks (PyTorch) |
| diffusers | Face generation (Stable Diffusion) |

### Full (Complete Environment)

All libraries from all targets combined. Use when you need everything.

## Updating Dependencies

All package versions live in one file: `targets/matrix.toml`. The per-target
`pyproject.toml` and `verify_imports.py` files are generated from it, and each
target has a committed `uv.lock` for reproducible image builds.

```bash
# 1. Edit targets/matrix.toml (versions, target membership, overrides)
# 2. Regenerate the per-target files
just gen
# 3. Re-resolve the lockfiles
just lock
# 4. Run every fast gate CI enforces
just ci
```

Version `overrides` in the matrix document real constraints (e.g. sktime capping
scikit-learn); remove an override once the upstream constraint is gone.

## Docker Commands Reference

### Build Commands

```bash
# Build specific target
docker build --target scientific -t ds-scientific .

# Build with no cache
docker build --no-cache --target ml -t ds-ml .

# Build full environment
docker build --target full -t ds-full .
```

### Run Commands

```bash
# Run with volume mounts
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  ds-scientific

# Run in background
docker run -d -p 8888:8888 --name jupyter \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  ds-ml

# Verify imports
docker run --rm ds-scientific uv run --no-project python /home/jupyter/scripts/verify_scientific.py

# Run IPython
docker run --rm -it ds-scientific uv run --no-project ipython
```

### Volume Mounts

| Local Directory | Container Path | Purpose |
|-----------------|----------------|---------|
| `./notebooks` | `/home/jupyter/notebooks` | Your notebooks |
| `./data` | `/home/jupyter/data` | Data files |
| `./examples` | `/home/jupyter/examples` | Example scripts |

## Example Files

The `examples/` directory contains Python scripts and Jupyter notebooks. The
`.py` files are the source of truth; notebooks are generated from them with
jupytext (`just nb`) and CI verifies they stay in sync:

| Example | Description |
|---------|-------------|
| `01_numpy_scipy_basics` | NumPy arrays, SciPy statistics |
| `02_pandas_data_analysis` | DataFrame operations |
| `03_matplotlib_seaborn_viz` | Static visualizations |
| `04_plotly_interactive` | Interactive charts |
| `05_bokeh_holoviews` | Bokeh and HoloViews |
| `06_geospatial` | Maps with Cartopy, GeoPandas, Folium |
| `07_timeseries_analysis` | Time series, ARIMA, forecasting |
| `08_data_io_serialization` | JSON, XML, Parquet, HDF5 |
| `09_machine_learning` | Classification, regression |
| `10_deep_learning_pytorch` | PyTorch neural networks |
| `11_deep_learning_tensorflow` | TensorFlow and Keras |
| `12_image_processing` | PIL, OpenCV, scikit-image |
| `13_object_detection_yolo` | YOLOv8 object detection |
| `14_nlp_text_analysis` | spaCy, NLTK, sentence-transformers |
| `15_audio_analysis` | librosa, torchaudio features |
| `16_altair_panel_viz` | Altair, hvPlot, Panel dashboards |
| `17_scipy_signal_processing` | FFT, filters, spectrograms |
| `18_sqlalchemy_database` | SQLAlchemy ORM, Parquet, HDF5 |
| `19_speech_processing` | Whisper ASR, gTTS, torchaudio |
| `20_face_analysis` | dlib, DeepFace, face-alignment |

## Choosing the Right Target

| Use Case | Recommended Target |
|----------|-------------------|
| Data analysis with Pandas | `scientific` |
| Creating charts/dashboards | `visualization` |
| Machine learning models | `ml` |
| Deep learning/neural networks | `deeplearn` |
| Image processing | `vision` |
| Audio processing | `audio` |
| Geographic data/maps | `geospatial` |
| Time series forecasting | `timeseries` |
| Text/NLP work | `nlp` |
| Speech recognition/synthesis | `speech` |
| Face detection/recognition | `face` |
| Need everything | `full` |

## Container Details

- **Base Image**: Ubuntu 24.04
- **Python**: 3.13 (via deadsnakes PPA)
- **Package Manager**: uv
- **User**: `jupyter` (non-root, UID 1000 — bind mounts keep host ownership)
- **Working Directory**: `/home/jupyter`
- **Exposed Port**: 8888

## Security Note

Jupyter Lab requires **token authentication** by default. A random token is generated
on every container start — find the login URL (with `?token=...`) in the logs:

```bash
docker logs jupyter
```

To set a fixed token instead, pass the `JUPYTER_TOKEN` environment variable:

```bash
docker run -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secret-token \
  ds-scientific
```

Do not expose the port beyond localhost without a token and TLS — put a reverse proxy
in front for anything non-local.

## GPU Support

For NVIDIA GPU support:

```bash
docker run --gpus all -p 8888:8888 ds-deeplearn
```

Note: Requires NVIDIA Container Toolkit.

## Project Structure

```
.
├── Dockerfile              # Multi-stage Dockerfile
├── justfile                # Dev entrypoint (just --list)
├── build-all.sh            # Build and test all targets
├── README.md
├── scripts/
│   └── gen_targets.py      # Regenerates per-target files from the matrix
├── targets/
│   ├── matrix.toml         # SOURCE OF TRUTH: packages, versions, target tree
│   ├── base/
│   │   ├── pyproject.toml  # GENERATED — edit matrix.toml instead
│   │   ├── uv.lock         # Committed lockfile (reproducible builds)
│   │   └── verify_imports.py
│   ├── scientific/
│   ├── visualization/
│   ├── dataio/
│   ├── ml/
│   ├── deeplearn/
│   ├── vision/
│   ├── audio/
│   ├── geospatial/
│   ├── timeseries/
│   ├── nlp/
│   ├── speech/
│   ├── face/
│   └── full/
├── examples/
│   ├── *.py                # Python scripts
│   └── *.ipynb             # Jupyter notebooks
├── notebooks/              # Your notebooks (mounted)
└── data/                   # Your data (mounted)
```
