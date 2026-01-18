# Data Science Jupyter Notebook Environment

A comprehensive, Dockerized Jupyter Notebook environment pre-configured with Python 3.13 and essential data science libraries for statistics, analysis, and visualization tasks.

**Package Manager**: [uv](https://docs.astral.sh/uv/) - Fast Python package manager from Astral

## Quick Start

### Build the Docker Image

```bash
docker build -t datascience-notebook .
```

### Run the Container

```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/examples:/home/jupyter/examples \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook
```

Access Jupyter Lab at: **http://localhost:8888**

This mounts local directories into the container, so:
- Changes made outside the container are immediately visible inside
- Work created inside the container persists after the container stops
- No rebuild needed when you modify files

### Directory Mounts Explained

| Local Directory | Container Path | Purpose |
|-----------------|----------------|---------|
| `./notebooks` | `/home/jupyter/notebooks` | Your Jupyter notebooks |
| `./examples` | `/home/jupyter/examples` | Example scripts and notebooks |
| `./data` | `/home/jupyter/data` | Data files for analysis |

### Minimal Run (No Persistence)

```bash
docker run -p 8888:8888 datascience-notebook
```

**Note**: Without volume mounts, all work is lost when the container stops.

## Installed Libraries

### Core Scientific Computing

| Library | Description | Links |
|---------|-------------|-------|
| **NumPy** | Fundamental package for numerical computing with N-dimensional arrays | [Website](https://numpy.org/) \| [Docs](https://numpy.org/doc/stable/) |
| **SciPy** | Scientific computing library for optimization, integration, interpolation, and more | [Website](https://scipy.org/) \| [Docs](https://docs.scipy.org/doc/scipy/) |
| **Pandas** | Data manipulation and analysis with DataFrames | [Website](https://pandas.pydata.org/) \| [Docs](https://pandas.pydata.org/docs/) |
| **IPython** | Enhanced interactive Python shell | [Website](https://ipython.org/) \| [Docs](https://ipython.readthedocs.io/) |
| **Jupyter** | Interactive notebook environment | [Website](https://jupyter.org/) \| [Docs](https://docs.jupyter.org/) |
| **JupyterLab** | Next-generation web-based interface for Jupyter | [Website](https://jupyterlab.readthedocs.io/) |

### Visualization Libraries

| Library | Description | Links |
|---------|-------------|-------|
| **Matplotlib** | Comprehensive 2D/3D plotting library | [Website](https://matplotlib.org/) \| [Gallery](https://matplotlib.org/stable/gallery/) |
| **Seaborn** | Statistical data visualization based on Matplotlib | [Website](https://seaborn.pydata.org/) \| [Gallery](https://seaborn.pydata.org/examples/) |
| **Plotly** | Interactive graphing library for web-based visualizations | [Website](https://plotly.com/python/) \| [Docs](https://plotly.com/python/getting-started/) |
| **Bokeh** | Interactive visualization library for modern web browsers | [Website](https://bokeh.org/) \| [Gallery](https://docs.bokeh.org/en/latest/docs/gallery.html) |
| **HoloViews** | Declarative data visualization | [Website](https://holoviews.org/) \| [Gallery](https://holoviews.org/gallery/) |
| **hvPlot** | High-level plotting API for Pandas, Dask, and more | [Website](https://hvplot.holoviz.org/) |
| **Panel** | Build interactive dashboards and apps | [Website](https://panel.holoviz.org/) |
| **Altair** | Declarative statistical visualization | [Website](https://altair-viz.github.io/) \| [Gallery](https://altair-viz.github.io/gallery/) |

### Geospatial Libraries

| Library | Description | Links |
|---------|-------------|-------|
| **Cartopy** | Geospatial data processing and map projections | [Website](https://scitools.org.uk/cartopy/) \| [Gallery](https://scitools.org.uk/cartopy/docs/latest/gallery/) |
| **GeoPandas** | Extends Pandas for geospatial data | [Website](https://geopandas.org/) \| [Docs](https://geopandas.org/en/stable/docs.html) |
| **Shapely** | Manipulation and analysis of geometric objects | [Website](https://shapely.readthedocs.io/) |
| **PyProj** | Cartographic projections and coordinate transformations | [Website](https://pyproj4.github.io/pyproj/) |
| **Folium** | Interactive Leaflet.js maps | [Website](https://python-visualization.github.io/folium/) |
| **GeoViews** | Geographic data visualization with HoloViews | [Website](https://geoviews.org/) |

### Time Series Analysis

| Library | Description | Links |
|---------|-------------|-------|
| **tsfresh** | Automatic time series feature extraction | [Website](https://tsfresh.readthedocs.io/) \| [Docs](https://tsfresh.readthedocs.io/en/latest/) |
| **sktime** | Unified framework for time series ML | [Website](https://www.sktime.net/) \| [Docs](https://www.sktime.net/en/stable/) |
| **Statsmodels** | Statistical models, hypothesis tests, and data exploration | [Website](https://www.statsmodels.org/) \| [Docs](https://www.statsmodels.org/stable/index.html) |
| **pmdarima** | Auto-ARIMA and statistical analysis | [Website](https://alkaline-ml.com/pmdarima/) |
| **Prophet** | Forecasting tool by Meta for time series with seasonality | [Website](https://facebook.github.io/prophet/) |

### Machine Learning

| Library | Description | Links |
|---------|-------------|-------|
| **Scikit-learn** | Machine learning algorithms and tools | [Website](https://scikit-learn.org/) \| [Docs](https://scikit-learn.org/stable/documentation.html) |
| **XGBoost** | Optimized gradient boosting library | [Website](https://xgboost.readthedocs.io/) \| [Docs](https://xgboost.readthedocs.io/en/stable/) |
| **LightGBM** | Fast gradient boosting framework | [Website](https://lightgbm.readthedocs.io/) \| [Docs](https://lightgbm.readthedocs.io/en/latest/) |

### Deep Learning Frameworks

| Library | Description | Links |
|---------|-------------|-------|
| **PyTorch** | Deep learning framework with dynamic computation graphs | [Website](https://pytorch.org/) \| [Docs](https://pytorch.org/docs/stable/) |
| **TorchVision** | Image datasets, transforms, and models for PyTorch | [Website](https://pytorch.org/vision/) \| [Docs](https://pytorch.org/vision/stable/) |
| **TorchAudio** | Audio processing and datasets for PyTorch | [Website](https://pytorch.org/audio/) \| [Docs](https://pytorch.org/audio/stable/) |
| **TensorFlow** | End-to-end machine learning platform | [Website](https://tensorflow.org/) \| [Docs](https://tensorflow.org/api_docs) |
| **Keras** | High-level neural networks API | [Website](https://keras.io/) \| [Docs](https://keras.io/api/) |

### Image Processing

| Library | Description | Links |
|---------|-------------|-------|
| **Pillow** | Python Imaging Library (PIL) fork for image processing | [Website](https://pillow.readthedocs.io/) \| [Docs](https://pillow.readthedocs.io/en/stable/) |
| **OpenCV** | Computer vision and image processing library (headless) | [Website](https://opencv.org/) \| [Docs](https://docs.opencv.org/) |
| **scikit-image** | Image processing algorithms for SciPy | [Website](https://scikit-image.org/) \| [Docs](https://scikit-image.org/docs/stable/) |
| **imageio** | Read/write images in various formats | [Website](https://imageio.readthedocs.io/) \| [Docs](https://imageio.readthedocs.io/en/stable/) |
| **Ultralytics YOLO** | State-of-the-art object detection (YOLOv8+) | [Website](https://ultralytics.com/) \| [Docs](https://docs.ultralytics.com/) |

### Data I/O and Serialization

| Library | Description | Links |
|---------|-------------|-------|
| **orjson** | Fast JSON library (10x faster than stdlib) | [GitHub](https://github.com/ijl/orjson) |
| **ujson** | Ultra-fast JSON encoder/decoder | [GitHub](https://github.com/ultrajson/ultrajson) |
| **simplejson** | JSON encoder/decoder with extra features | [GitHub](https://github.com/simplejson/simplejson) |
| **lxml** | Fast XML and HTML processing | [Website](https://lxml.de/) |
| **xmltodict** | Work with XML like JSON | [GitHub](https://github.com/martinblech/xmltodict) |
| **BeautifulSoup4** | HTML/XML parsing and scraping | [Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) |
| **PyYAML** | YAML parser and emitter | [Website](https://pyyaml.org/) |
| **openpyxl** | Read/write Excel 2010+ files | [Docs](https://openpyxl.readthedocs.io/) |
| **xlrd** | Read Excel files | [GitHub](https://github.com/python-excel/xlrd) |
| **PyArrow** | Apache Arrow for columnar data | [Website](https://arrow.apache.org/docs/python/) |
| **fastparquet** | Parquet format support | [GitHub](https://github.com/dask/fastparquet) |
| **h5py** | HDF5 binary data format | [Website](https://www.h5py.org/) |
| **PyTables** | HDF5 with advanced indexing | [Website](https://www.pytables.org/) |

### HTTP and API Clients

| Library | Description | Links |
|---------|-------------|-------|
| **Requests** | HTTP library for humans | [Website](https://requests.readthedocs.io/) |
| **HTTPX** | Modern HTTP client with async support | [Website](https://www.python-httpx.org/) |
| **aiohttp** | Async HTTP client/server | [Website](https://docs.aiohttp.org/) |

### Utilities

| Library | Description | Links |
|---------|-------------|-------|
| **Pydantic** | Data validation using Python type hints | [Website](https://docs.pydantic.dev/) |
| **tqdm** | Progress bars for loops | [GitHub](https://github.com/tqdm/tqdm) |
| **Loguru** | Simple and powerful logging | [GitHub](https://github.com/Delgan/loguru) |
| **python-dotenv** | Read .env files | [GitHub](https://github.com/theskumar/python-dotenv) |
| **joblib** | Efficient serialization and parallelism | [Website](https://joblib.readthedocs.io/) |
| **toolz** | Functional programming utilities | [Website](https://toolz.readthedocs.io/) |
| **more-itertools** | Extended iteration tools | [Website](https://more-itertools.readthedocs.io/) |
| **python-dateutil** | Powerful date/time utilities | [Website](https://dateutil.readthedocs.io/) |
| **pytz** | Timezone definitions | [Website](https://pythonhosted.org/pytz/) |
| **pendulum** | Easy datetime manipulation | [Website](https://pendulum.eustace.io/) |

## Example Files

The `examples/` directory contains both Python scripts (`.py`) and Jupyter notebooks (`.ipynb`) demonstrating library usage:

| Example | Description |
|---------|-------------|
| `01_numpy_scipy_basics` | NumPy arrays, SciPy statistics and optimization |
| `02_pandas_data_analysis` | DataFrame operations, grouping, time series |
| `03_matplotlib_seaborn_viz` | Static visualizations and statistical plots |
| `04_plotly_interactive` | Interactive charts, 3D plots, animations |
| `05_bokeh_holoviews` | Bokeh dashboards, HoloViews declarative viz |
| `06_geospatial` | Cartopy maps, GeoPandas, Folium interactive maps |
| `07_timeseries_analysis` | Time series decomposition, ARIMA, feature extraction |
| `08_data_io_serialization` | JSON, XML, YAML, Parquet, HDF5 I/O |
| `09_machine_learning` | Classification, regression, clustering |
| `10_deep_learning_pytorch` | PyTorch tensors, autograd, neural networks, TorchVision, TorchAudio |
| `11_deep_learning_tensorflow` | TensorFlow tensors, Keras models, training loops, callbacks |
| `12_image_processing` | PIL, OpenCV, scikit-image, imageio operations |
| `13_object_detection_yolo` | Ultralytics YOLOv8 object detection |

Each example is available in two formats:
- **`.py`** - Python script (can be run from command line)
- **`.ipynb`** - Jupyter notebook (interactive, with documentation)

### Running Examples

**In Jupyter Lab** (recommended): Open any `.ipynb` file from the examples folder.

**From command line** (inside container):

```bash
# Run a Python script
uv run python examples/01_numpy_scipy_basics.py

# Or run all Python examples
for f in examples/*.py; do uv run python "$f"; done
```

## Docker Commands Reference

### Build

```bash
# Standard build
docker build -t datascience-notebook .

# Build with no cache (for updates)
docker build --no-cache -t datascience-notebook .
```

### Run

```bash
# Test that all packages are installed correctly:
docker run --rm datascience-notebook uv run python /home/jupyter/scripts/verify_imports.py

# Recommended: Run with all volume mounts (live file updates)
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/examples:/home/jupyter/examples \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook

# Run in detached mode with volume mounts
docker run -d -p 8888:8888 --name jupyter \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/examples:/home/jupyter/examples \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook

# Run with custom port
docker run -p 9999:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/examples:/home/jupyter/examples \
  datascience-notebook

# Run with increased memory limit
docker run -p 8888:8888 --memory=4g \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/examples:/home/jupyter/examples \
  datascience-notebook

# Run with environment variables
docker run -p 8888:8888 -e MY_VAR=value \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  datascience-notebook
```

### Run IPython Shell

```bash
# Interactive IPython with volume mounts
docker run --rm -it \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook uv run ipython

# Quick IPython without mounts
docker run --rm -it datascience-notebook uv run ipython
```

## Security Note

The default configuration has **no authentication** for convenience in local development. For production or shared environments, use a token:

```bash
docker run -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secret-token \
  datascience-notebook \
  uv run jupyter lab --ip=0.0.0.0 --IdentityProvider.token='your-secret-token'
```

## Container Details

- **Base Image**: `python:3.13-slim-bookworm`
- **Package Manager**: `uv` (https://docs.astral.sh/uv/)
- **User**: `jupyter` (non-root)
- **Working Directory**: `/home/jupyter`
- **Exposed Port**: `8888`
- **Default Command**: `uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`


### Using a Password

For production use, set a password:

```bash
docker run -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secret-token \
  datascience-notebook uv run jupyter lab --ip=0.0.0.0 --IdentityProvider.token='your-secret-token'
```

### GPU Support

For NVIDIA GPU support, use nvidia-docker:

```bash
docker run --gpus all -p 8888:8888 datascience-notebook
```

Note: Requires NVIDIA Container Toolkit and GPU-compatible packages to be added to Dockerfile.
