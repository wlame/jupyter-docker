# Data Science Jupyter Notebook Environment

A modular, multi-target Docker environment for data science on Python 3.14 (and 3.13). Build only what you need - from a lightweight base image to a comprehensive full environment.

**Package Manager**: [uv](https://docs.astral.sh/uv/) - Fast Python package manager from Astral

## Quick Start

### Use Prebuilt Images

Prebuilt images are published to GitHub Container Registry on every push to `main`
and rebuilt weekly to pick up OS security patches. Each target is published for
every Python it supports: `:<target>` is its default (3.14 for most targets),
`:<target>-py3.13` / `:<target>-py3.14` pick one explicitly, and every push also
adds immutable `-<short-sha>` tags for pinning and rollback:

```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  ghcr.io/wlame/jupyter-docker:scientific
```

See [Available Targets](#available-targets) for the full list. Replace `scientific` with any target name, or
append `-py3.13` to choose Python 3.13 (`deeplearn`, `face`, and `full` are 3.13-only for now).

### Build from Source

```bash
# Build only what you need (each on its default Python from the matrix)
just build base
just build scientific
just build ml 3.13      # pick a Python explicitly
just build full         # TensorFlow targets default to 3.13
```

Without `just`, pass the Python as a build argument (default 3.14):

```bash
docker build --target scientific -t ds-scientific .
docker build --build-arg PYTHON_VERSION=3.13 --target full -t ds-full .
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
| `optimization` | not yet measured | OR-Tools, CVXPY, Pyomo, PuLP | scientific |
| `jax` | not yet measured | JAX, Optax, Flax, Equinox, NumPyro | scientific |
| `probabilistic` | not yet measured | PyMC, nutpie, ArviZ, Bambi | jax |
| `nlp` | ~4GB | spaCy, Transformers, NLTK | base |
| `genai` | not yet measured | TRL, FAISS, bm25s, diffusers, LLM clients | nlp |
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
│   ├── timeseries
│   ├── optimization
│   └── jax
│       └── probabilistic
├── visualization
├── dataio
├── vision
├── audio
├── nlp
│   └── genai
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
| tqdm, loguru, rich | Progress bars, logging, terminal formatting |
| tenacity | Retries with backoff |
| python-dotenv | .env file loading |
| python-dateutil, pytz, pendulum | Date/time utilities |
| joblib, toolz, more-itertools | Utilities |
| Jupytext, jupyterlab-git, nbdime | Notebook ↔ `.py` pairing, Git UI, notebook diffs |
| jupyterlab-lsp, python-lsp-server | Completion, hover, diagnostics |
| jupyter-resource-usage | Memory/CPU indicator |

### Scientific (Numerical Computing)

Core libraries for numerical and statistical computing.

| Library | Description |
|---------|-------------|
| NumPy | N-dimensional arrays |
| SciPy | Scientific algorithms |
| Pandas | DataFrames |
| Statsmodels | Statistical models |
| SymPy | Symbolic mathematics |
| polars, DuckDB, PyArrow | Fast DataFrames, SQL over DataFrames, Arrow |
| xarray, pint | Labeled N-D arrays, physical units |
| numexpr, bottleneck | pandas accelerators |
| Numba | JIT compiler for numeric Python |
| Dask, distributed | Parallel and out-of-core computing (dashboard through JupyterLab) |

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
| plotnine | ggplot2 grammar of graphics |
| great-tables | Publication-quality tables |
| datashader | Rendering millions of points |
| vl-convert | Static Altair export |
| itables | Sortable, searchable DataFrame tables |
| ipympl, jupyter-bokeh | Interactive matplotlib and Bokeh widgets |

### DataIO (Data Formats & Databases)

Read and write various data formats.

| Library | Description |
|---------|-------------|
| PyArrow | Apache Arrow columnar data |
| fastparquet | Parquet format |
| h5py, PyTables | HDF5 format |
| openpyxl, xlrd | Excel files |
| SQLAlchemy | Database ORM |
| polars, DuckDB | Fast DataFrames and SQL |
| deltalake | Delta Lake tables |
| python-calamine, xlsxwriter | Fast Excel reading, formatted Excel writing |
| pyreadstat | SPSS / SAS / Stata files |
| zarr, netCDF4 | Chunked and scientific arrays |
| s3fs, gcsfs, adlfs | Cloud storage for fsspec |
| psycopg, pymysql, connectorx | Database drivers, fast SQL loading |
| Ibis | One dataframe API compiled to SQL (DuckDB backend) |
| ADBC (SQLite, PostgreSQL) | Arrow tables in and out of databases |

### ML (Machine Learning)

Classical machine learning algorithms.

| Library | Description |
|---------|-------------|
| Scikit-learn | ML algorithms |
| XGBoost | Gradient boosting |
| LightGBM | Fast gradient boosting |
| imbalanced-learn | Imbalanced datasets |
| Optuna | Hyperparameter optimization |
| CatBoost | Gradient boosting with categorical features |
| SHAP | Model explanations |
| MAPIE | Conformal prediction intervals |
| UMAP | Non-linear embeddings |
| skrub, skops | DataFrame preparation, safe model persistence |
| ONNX, onnxruntime, skl2onnx | Export pipelines, serve them without scikit-learn |

### DeepLearn (Neural Networks)

Deep learning frameworks.

| Library | Description |
|---------|-------------|
| PyTorch | Dynamic neural networks |
| TorchVision | Computer vision for PyTorch |
| TorchAudio | Audio for PyTorch |
| TensorFlow | ML platform |
| Keras | High-level neural network API |
| Lightning, torchmetrics | Training loops and metrics |
| Accelerate, einops | Device handling, tensor reshaping |
| TensorBoard | Training dashboards (ONNX comes from ML) |

### Vision (Image Processing)

Computer vision and image manipulation.

| Library | Description |
|---------|-------------|
| Pillow | Image processing |
| OpenCV (headless) | Computer vision |
| scikit-image | Image algorithms |
| imageio | Image I/O |
| Ultralytics | YOLO26 object detection |
| timm, OpenCLIP | Backbones and image–text models |
| kornia | Differentiable image ops and augmentation |
| supervision | Detection annotation and tracking |
| onnxruntime | Fast inference for exported models |

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
| pedalboard | Studio effects, augmentation, audio I/O |
| pyloudnorm | Loudness measurement |
| noisereduce | Spectral-gating denoise |
| praat-parselmouth | Praat phonetics (pitch, formants) |

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
| rasterio, rioxarray | Raster data |
| H3 | Hexagonal spatial indexing |
| mapclassify | Choropleth classification |
| OSMnx | OpenStreetMap street networks |
| contextily, geodatasets | Basemaps, sample data |
| lonboard | GPU-rendered maps |

### TimeSeries (Time Series Analysis)

Time series modeling and forecasting.

| Library | Description |
|---------|-------------|
| tsfresh | Feature extraction |
| sktime | Time series ML |
| pmdarima | Auto-ARIMA |
| Prophet | Forecasting |
| statsforecast, mlforecast | Fast statistical and ML forecasting (Nixtla) |
| skforecast | Forecasting with scikit-learn regressors |
| arch | Volatility (GARCH) models |
| tslearn | Time-series clustering and DTW |

### Optimization (Operations Research)

Linear, integer, and convex optimization, routing, and scheduling.

| Library | Description |
|---------|-------------|
| OR-Tools | CP-SAT, vehicle routing, embedded SCIP/CBC/HiGHS/GLOP/PDLP |
| CVXPY | Convex optimization (Clarabel, OSQP, SCS) |
| Pyomo, PuLP | Algebraic modeling, solved with the bundled CBC command |

### JAX (Accelerated Numerics)

JIT compilation, automatic differentiation, and neural networks on JAX (CPU
jaxlib; `pip install "jax[cuda13]"` for NVIDIA GPUs).

| Library | Description |
|---------|-------------|
| JAX | NumPy-like arrays with jit, grad, and vmap |
| Optax | Gradient-based optimizers |
| Flax NNX, Equinox | Neural network libraries |
| NumPyro | Probabilistic programming on JAX |

### Probabilistic (Bayesian Statistics)

Bayesian modeling on top of the JAX image.

| Library | Description |
|---------|-------------|
| PyMC | Probabilistic programming (PyTensor, compiled with Numba) |
| nutpie | Fast NUTS sampler written in Rust |
| ArviZ | Diagnostics, summaries, and plots |
| Bambi | Regression models from formulas |
| PreliZ | Prior elicitation |

### NLP (Natural Language Processing)

Text processing and language models.

| Library | Description |
|---------|-------------|
| spaCy | Industrial NLP |
| NLTK | Classic NLP toolkit |
| Transformers | Hugging Face models |
| sentence-transformers | Sentence embeddings |
| tokenizers | Fast tokenization |
| datasets, evaluate | Hugging Face datasets and metrics |
| Accelerate, PEFT | Fine-tuning, LoRA adapters |
| rapidfuzz, lingua | Fuzzy matching, language detection |
| tiktoken, SentencePiece | OpenAI tokenizers, subword tokenizer training |
| BERTopic, KeyBERT | Topic modelling, keyword extraction |

### GenAI (Retrieval and Fine-Tuning)

Retrieval-augmented generation, fine-tuning, and LLM clients on top of NLP.

| Library | Description |
|---------|-------------|
| TRL | Supervised fine-tuning, DPO, GRPO |
| FAISS, bm25s | Vector search, keyword search |
| diffusers | Diffusion models |
| openai, anthropic | Clients for hosted APIs and local OpenAI-compatible servers |

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
| jiwer | WER / CER for evaluating ASR |
| silero-vad | Voice activity detection |
| praat-parselmouth | Prosody and pitch analysis |

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
| InsightFace, MediaPipe | Face recognition, face mesh and landmarks |

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

Version `overrides` in the matrix document real constraints (e.g. TensorFlow capping
h5py in `full`); remove an override once the upstream constraint is gone.

## Docker Commands Reference

### Build Commands

```bash
# Build specific target
docker build --target scientific -t ds-scientific .

# Build with no cache
docker build --no-cache --target ml -t ds-ml .

# Build on Python 3.13 (required for the TensorFlow targets: deeplearn, face, full)
docker build --build-arg PYTHON_VERSION=3.13 --target full -t ds-full .
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
jupytext (`just nb`) and CI verifies they stay in sync. Each image ships the
examples of its own target and its parents (`ml` also has the `scientific`
examples), and `full` ships all of them:

| Example | Image | Description |
|---------|-------|-------------|
| `01_numpy_scipy_basics` | `scientific` | NumPy arrays, SciPy statistics |
| `02_pandas_data_analysis` | `scientific` | DataFrame operations |
| `03_matplotlib_seaborn_viz` | `visualization` | Static visualizations |
| `04_plotly_interactive` | `visualization` | Interactive charts |
| `05_bokeh_holoviews` | `visualization` | Bokeh and HoloViews |
| `06_geospatial` | `geospatial` | Maps with Cartopy, GeoPandas, Folium |
| `07_timeseries_analysis` | `timeseries` | Time series, ARIMA, forecasting |
| `08_data_io_serialization` | `dataio` | JSON, XML, Parquet, HDF5 |
| `09_machine_learning` | `ml` | Classification, regression |
| `10_deep_learning_pytorch` | `deeplearn` | PyTorch neural networks |
| `11_deep_learning_tensorflow` | `deeplearn` | TensorFlow and Keras |
| `12_image_processing` | `vision` | PIL, OpenCV, scikit-image |
| `13_object_detection_yolo` | `vision` | YOLO26 object detection |
| `14_nlp_text_analysis` | `nlp` | spaCy, NLTK, sentence-transformers |
| `15_audio_analysis` | `audio` | librosa, torchaudio features |
| `16_altair_panel_viz` | `visualization` | Altair, hvPlot, Panel dashboards |
| `17_scipy_signal_processing` | `scientific` | FFT, filters, spectrograms |
| `18_sqlalchemy_database` | `dataio` | SQLAlchemy ORM, Parquet, HDF5 |
| `19_speech_processing` | `speech` | Whisper ASR, gTTS, torchaudio |
| `20_face_analysis` | `face` | dlib, DeepFace, face-alignment |
| `21_polars_duckdb_xarray` | `scientific` | polars, DuckDB, Arrow, xarray, pint |
| `22_plotnine_tables_datashader` | `visualization` | plotnine, great-tables, itables, datashader, vl-convert |
| `23_modern_data_formats` | `dataio` | Delta Lake, Excel, SPSS, zarr, netCDF, connectorx |
| `24_ml_explain_and_uncertainty` | `ml` | CatBoost, SHAP, MAPIE, UMAP, skrub, skops, skl2onnx |
| `25_lightning_onnx` | `deeplearn` | Lightning, torchmetrics, Accelerate, einops, ONNX |
| `26_vision_backbones_kornia` | `vision` | timm, kornia, supervision, OpenCLIP |
| `27_audio_effects_loudness` | `audio` | pedalboard, pyloudnorm, noisereduce, parselmouth |
| `28_geospatial_raster_h3` | `geospatial` | rasterio, rioxarray, H3, mapclassify, OSMnx, lonboard |
| `29_forecasting_toolkit` | `timeseries` | statsforecast, mlforecast, skforecast, arch, tslearn |
| `30_nlp_toolkit` | `nlp` | rapidfuzz, lingua, datasets, PEFT, KeyBERT, SentencePiece |
| `31_speech_metrics_vad` | `speech` | jiwer, silero-vad, parselmouth |
| `32_dataframe_engines` | `dataio` | One query in pandas, Polars, DuckDB, and Ibis; ADBC |
| `33_lorenz_numba_dask` | `scientific` | Lorenz attractor with SymPy, Numba, FFT, and Dask |
| `34_optimization` | `optimization` | Assignment, OR-Tools routing, CVXPY portfolios, Pyomo MILP |
| `35_jax_jit_autodiff` | `jax` | jit Mandelbrot, grad, Optax, Flax NNX and Equinox |
| `36_bayesian_modeling` | `probabilistic` | PreliZ prior, three NUTS samplers, eight schools, Bambi |
| `37_retrieval_and_generation` | `genai` | BM25 vs embeddings vs hybrid search, RAG with SmolLM2, TRL |

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
| Optimization (LP/MIP, routing, scheduling) | `optimization` |
| JAX, autodiff, differentiable programming | `jax` |
| Bayesian statistics (PyMC) | `probabilistic` |
| Text/NLP work | `nlp` |
| RAG, fine-tuning LLMs, LLM API clients | `genai` |
| Speech recognition/synthesis | `speech` |
| Face detection/recognition | `face` |
| Need everything | `full` |

## Container Details

- **Base Image**: Ubuntu 24.04
- **Python**: 3.14 by default, 3.13 alongside it (via deadsnakes PPA); `deeplearn`, `face`, and `full` are 3.13-only until TensorFlow supports 3.14
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
