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
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/jupyter/notebooks datascience-notebook
```

Access Jupyter Lab at: **http://localhost:8888**

### Run with Persistent Data

```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook
```

## Installed Libraries

### Core Scientific Computing

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **NumPy** | Latest | Fundamental package for numerical computing with N-dimensional arrays | [Website](https://numpy.org/) \| [Docs](https://numpy.org/doc/stable/) |
| **SciPy** | Latest | Scientific computing library for optimization, integration, interpolation, and more | [Website](https://scipy.org/) \| [Docs](https://docs.scipy.org/doc/scipy/) |
| **Pandas** | Latest | Data manipulation and analysis with DataFrames | [Website](https://pandas.pydata.org/) \| [Docs](https://pandas.pydata.org/docs/) |
| **IPython** | Latest | Enhanced interactive Python shell | [Website](https://ipython.org/) \| [Docs](https://ipython.readthedocs.io/) |
| **Jupyter** | Latest | Interactive notebook environment | [Website](https://jupyter.org/) \| [Docs](https://docs.jupyter.org/) |
| **JupyterLab** | Latest | Next-generation web-based interface for Jupyter | [Website](https://jupyterlab.readthedocs.io/) |

### Visualization Libraries

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **Matplotlib** | Latest | Comprehensive 2D/3D plotting library | [Website](https://matplotlib.org/) \| [Gallery](https://matplotlib.org/stable/gallery/) |
| **Seaborn** | Latest | Statistical data visualization based on Matplotlib | [Website](https://seaborn.pydata.org/) \| [Gallery](https://seaborn.pydata.org/examples/) |
| **Plotly** | Latest | Interactive graphing library for web-based visualizations | [Website](https://plotly.com/python/) \| [Docs](https://plotly.com/python/getting-started/) |
| **Bokeh** | Latest | Interactive visualization library for modern web browsers | [Website](https://bokeh.org/) \| [Gallery](https://docs.bokeh.org/en/latest/docs/gallery.html) |
| **HoloViews** | Latest | Declarative data visualization | [Website](https://holoviews.org/) \| [Gallery](https://holoviews.org/gallery/) |
| **hvPlot** | Latest | High-level plotting API for Pandas, Dask, and more | [Website](https://hvplot.holoviz.org/) |
| **Panel** | Latest | Build interactive dashboards and apps | [Website](https://panel.holoviz.org/) |
| **Altair** | Latest | Declarative statistical visualization | [Website](https://altair-viz.github.io/) \| [Gallery](https://altair-viz.github.io/gallery/) |

### Geospatial Libraries

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **Cartopy** | Latest | Geospatial data processing and map projections | [Website](https://scitools.org.uk/cartopy/) \| [Gallery](https://scitools.org.uk/cartopy/docs/latest/gallery/) |
| **GeoPandas** | Latest | Extends Pandas for geospatial data | [Website](https://geopandas.org/) \| [Docs](https://geopandas.org/en/stable/docs.html) |
| **Shapely** | Latest | Manipulation and analysis of geometric objects | [Website](https://shapely.readthedocs.io/) |
| **PyProj** | Latest | Cartographic projections and coordinate transformations | [Website](https://pyproj4.github.io/pyproj/) |
| **Folium** | Latest | Interactive Leaflet.js maps | [Website](https://python-visualization.github.io/folium/) |
| **GeoViews** | Latest | Geographic data visualization with HoloViews | [Website](https://geoviews.org/) |

### Time Series Analysis

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **tsfresh** | Latest | Automatic time series feature extraction | [Website](https://tsfresh.readthedocs.io/) \| [Docs](https://tsfresh.readthedocs.io/en/latest/) |
| **sktime** | Latest | Unified framework for time series ML | [Website](https://www.sktime.net/) \| [Docs](https://www.sktime.net/en/stable/) |
| **Statsmodels** | Latest | Statistical models, hypothesis tests, and data exploration | [Website](https://www.statsmodels.org/) \| [Docs](https://www.statsmodels.org/stable/index.html) |
| **pmdarima** | Latest | Auto-ARIMA and statistical analysis | [Website](https://alkaline-ml.com/pmdarima/) |
| **Prophet** | Latest | Forecasting tool by Meta for time series with seasonality | [Website](https://facebook.github.io/prophet/) |

### Machine Learning

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **Scikit-learn** | Latest | Machine learning algorithms and tools | [Website](https://scikit-learn.org/) \| [Docs](https://scikit-learn.org/stable/documentation.html) |
| **XGBoost** | Latest | Optimized gradient boosting library | [Website](https://xgboost.readthedocs.io/) \| [Docs](https://xgboost.readthedocs.io/en/stable/) |
| **LightGBM** | Latest | Fast gradient boosting framework | [Website](https://lightgbm.readthedocs.io/) \| [Docs](https://lightgbm.readthedocs.io/en/latest/) |

### Data I/O and Serialization

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **orjson** | Latest | Fast JSON library (10x faster than stdlib) | [GitHub](https://github.com/ijl/orjson) |
| **ujson** | Latest | Ultra-fast JSON encoder/decoder | [GitHub](https://github.com/ultrajson/ultrajson) |
| **simplejson** | Latest | JSON encoder/decoder with extra features | [GitHub](https://github.com/simplejson/simplejson) |
| **lxml** | Latest | Fast XML and HTML processing | [Website](https://lxml.de/) |
| **xmltodict** | Latest | Work with XML like JSON | [GitHub](https://github.com/martinblech/xmltodict) |
| **BeautifulSoup4** | Latest | HTML/XML parsing and scraping | [Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) |
| **PyYAML** | Latest | YAML parser and emitter | [Website](https://pyyaml.org/) |
| **openpyxl** | Latest | Read/write Excel 2010+ files | [Docs](https://openpyxl.readthedocs.io/) |
| **xlrd** | Latest | Read Excel files | [GitHub](https://github.com/python-excel/xlrd) |
| **PyArrow** | Latest | Apache Arrow for columnar data | [Website](https://arrow.apache.org/docs/python/) |
| **fastparquet** | Latest | Parquet format support | [GitHub](https://github.com/dask/fastparquet) |
| **h5py** | Latest | HDF5 binary data format | [Website](https://www.h5py.org/) |
| **PyTables** | Latest | HDF5 with advanced indexing | [Website](https://www.pytables.org/) |

### HTTP and API Clients

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **Requests** | Latest | HTTP library for humans | [Website](https://requests.readthedocs.io/) |
| **HTTPX** | Latest | Modern HTTP client with async support | [Website](https://www.python-httpx.org/) |
| **aiohttp** | Latest | Async HTTP client/server | [Website](https://docs.aiohttp.org/) |

### Utilities

| Library | Version | Description | Links |
|---------|---------|-------------|-------|
| **Pydantic** | Latest | Data validation using Python type hints | [Website](https://docs.pydantic.dev/) |
| **tqdm** | Latest | Progress bars for loops | [GitHub](https://github.com/tqdm/tqdm) |
| **Loguru** | Latest | Simple and powerful logging | [GitHub](https://github.com/Delgan/loguru) |
| **python-dotenv** | Latest | Read .env files | [GitHub](https://github.com/theskumar/python-dotenv) |
| **joblib** | Latest | Efficient serialization and parallelism | [Website](https://joblib.readthedocs.io/) |
| **toolz** | Latest | Functional programming utilities | [Website](https://toolz.readthedocs.io/) |
| **more-itertools** | Latest | Extended iteration tools | [Website](https://more-itertools.readthedocs.io/) |
| **python-dateutil** | Latest | Powerful date/time utilities | [Website](https://dateutil.readthedocs.io/) |
| **pytz** | Latest | Timezone definitions | [Website](https://pythonhosted.org/pytz/) |
| **pendulum** | Latest | Easy datetime manipulation | [Website](https://pendulum.eustace.io/) |

## Example Files

The `examples/` directory contains boilerplate scripts demonstrating library usage:

| File | Description |
|------|-------------|
| `01_numpy_scipy_basics.py` | NumPy arrays, SciPy statistics and optimization |
| `02_pandas_data_analysis.py` | DataFrame operations, grouping, time series |
| `03_matplotlib_seaborn_viz.py` | Static visualizations and statistical plots |
| `04_plotly_interactive.py` | Interactive charts, 3D plots, animations |
| `05_bokeh_holoviews.py` | Bokeh dashboards, HoloViews declarative viz |
| `06_geospatial.py` | Cartopy maps, GeoPandas, Folium interactive maps |
| `07_timeseries_analysis.py` | Time series decomposition, ARIMA, feature extraction |
| `08_data_io_serialization.py` | JSON, XML, YAML, Parquet, HDF5 I/O |
| `09_machine_learning.py` | Classification, regression, clustering |

### Running Examples

Inside the container:

```bash
# Run an example script
uv run python examples/01_numpy_scipy_basics.py

# Or run all examples
for f in examples/*.py; do uv run python "$f"; done
```

## Verifying Installation

The container includes a verification script to check all package imports:

```bash
# Inside the container
uv run python scripts/verify_imports.py
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
# Basic run
docker run -p 8888:8888 datascience-notebook

# With volume mounts for persistence
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook

# Run in detached mode
docker run -d -p 8888:8888 --name jupyter datascience-notebook

# Run with custom port
docker run -p 9999:8888 datascience-notebook

# Run with increased memory limit
docker run -p 8888:8888 --memory=4g datascience-notebook

# Run with environment variables
docker run -p 8888:8888 -e MY_VAR=value datascience-notebook
```

### Management

```bash
# Stop container
docker stop jupyter

# Start stopped container
docker start jupyter

# Remove container
docker rm jupyter

# View logs
docker logs jupyter

# Execute command in running container
docker exec -it jupyter bash

# Execute Python in container
docker exec -it jupyter uv run python -c "import numpy; print(numpy.__version__)"
```

### Cleanup

```bash
# Remove image
docker rmi datascience-notebook

# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune
```

## Directory Structure

```
.
├── Dockerfile              # Container definition
├── pyproject.toml          # Project dependencies (uv)
├── uv.lock                  # Lock file (generated)
├── README.md               # This file
├── INSTRUCTIONS.md         # Quick start guide
├── notebooks/              # Your Jupyter notebooks (mounted)
├── data/                   # Your datasets (mounted)
├── examples/               # Example scripts
│   ├── output/             # Generated visualizations
│   ├── 01_numpy_scipy_basics.py
│   ├── 02_pandas_data_analysis.py
│   ├── 03_matplotlib_seaborn_viz.py
│   ├── 04_plotly_interactive.py
│   ├── 05_bokeh_holoviews.py
│   ├── 06_geospatial.py
│   ├── 07_timeseries_analysis.py
│   ├── 08_data_io_serialization.py
│   └── 09_machine_learning.py
└── scripts/
    └── verify_imports.py   # Package verification script
```

## Container Details

- **Base Image**: `python:3.13-slim-bookworm`
- **Package Manager**: `uv` (https://docs.astral.sh/uv/)
- **User**: `jupyter` (non-root)
- **Working Directory**: `/home/jupyter`
- **Exposed Port**: `8888`
- **Default Command**: `uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`

## Customization

### Adding More Packages

To add additional packages, edit `pyproject.toml` and rebuild:

```toml
# Add to the dependencies list in pyproject.toml
dependencies = [
    # ... existing packages ...
    "your-package-here",
]
```

Then rebuild the image:

```bash
docker build -t datascience-notebook .
```

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

## Troubleshooting

### Container Won't Start

Check if port 8888 is already in use:
```bash
lsof -i :8888
```

### Import Errors

Run the verification script to identify issues:
```bash
docker exec -it jupyter uv run python scripts/verify_imports.py
```

### Out of Memory

Increase Docker memory limit in Docker Desktop settings or use `--memory` flag.

## License

This Dockerfile and example code are provided under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and feature requests, please open an issue in the repository.
