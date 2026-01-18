# Data Science Jupyter Notebook Environment

A modular, multi-target Docker environment for data science with Python 3.13. Build only what you need - from a lightweight base image to a comprehensive full environment.

**Package Manager**: [uv](https://docs.astral.sh/uv/) - Fast Python package manager from Astral

## Quick Start

### Build a Specific Target

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
| `full` | ~11GB | Everything combined | standalone |

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
└── nlp

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

### Full (Complete Environment)

All libraries from all targets combined. Use when you need everything.

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
docker run --rm ds-scientific uv run python /home/jupyter/scripts/verify_scientific.py

# Run IPython
docker run --rm -it ds-scientific uv run ipython
```

### Volume Mounts

| Local Directory | Container Path | Purpose |
|-----------------|----------------|---------|
| `./notebooks` | `/home/jupyter/notebooks` | Your notebooks |
| `./data` | `/home/jupyter/data` | Data files |
| `./examples` | `/home/jupyter/examples` | Example scripts |

## Example Files

The `examples/` directory contains Python scripts and Jupyter notebooks:

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
| Need everything | `full` |

## Container Details

- **Base Image**: Ubuntu 24.04
- **Python**: 3.13 (via deadsnakes PPA)
- **Package Manager**: uv
- **User**: `jupyter` (non-root)
- **Working Directory**: `/home/jupyter`
- **Exposed Port**: 8888

## Security Note

Default configuration has **no authentication** for local development. For production:

```bash
docker run -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secret-token \
  ds-scientific \
  uv run jupyter lab --ip=0.0.0.0 --IdentityProvider.token='your-secret-token'
```

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
├── build-all.sh            # Build and test all targets
├── README.md
├── targets/
│   ├── base/
│   │   ├── pyproject.toml
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
│   └── full/
├── examples/
│   ├── *.py                # Python scripts
│   └── *.ipynb             # Jupyter notebooks
├── notebooks/              # Your notebooks (mounted)
└── data/                   # Your data (mounted)
```
