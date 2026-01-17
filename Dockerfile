# =============================================================================
# Data Science Jupyter Notebook Environment
# =============================================================================
# A comprehensive Jupyter Notebook environment for data science, statistics,
# analysis, and visualization tasks.
#
# Python: 3.13 (latest stable with full ecosystem support)
# Base: Debian Bookworm (slim)
# =============================================================================

FROM python:3.13-slim-bookworm

LABEL maintainer="Data Engineering Team"
LABEL description="Comprehensive Data Science Jupyter Notebook Environment"
LABEL version="1.0"

# =============================================================================
# Environment Configuration
# =============================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_ENABLE_LAB=yes \
    DEBIAN_FRONTEND=noninteractive

# =============================================================================
# System Dependencies
# =============================================================================
# Install system packages required for scientific computing and geospatial libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Required for some scientific packages
    libopenblas-dev \
    liblapack-dev \
    # Required for matplotlib and plotting
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # Required for Cartopy and geospatial libraries
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgdal-dev \
    gdal-bin \
    # Required for HDF5 support (pandas, scipy)
    libhdf5-dev \
    # Required for XML parsing (lxml)
    libxml2-dev \
    libxslt1-dev \
    # Network and compression utilities
    curl \
    wget \
    git \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Create Working Directory and Non-Root User
# =============================================================================
RUN useradd -m -s /bin/bash jupyter && \
    mkdir -p /home/jupyter/notebooks /home/jupyter/data /home/jupyter/examples/output /home/jupyter/scripts && \
    chown -R jupyter:jupyter /home/jupyter

WORKDIR /home/jupyter

# =============================================================================
# Python Package Installation
# =============================================================================
# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Core Scientific Computing
# -----------------------------------------------------------------------------
RUN pip install \
    numpy \
    scipy \
    pandas \
    # IPython and Jupyter
    ipython \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    # Enable widgets in JupyterLab
    jupyterlab-widgets

# -----------------------------------------------------------------------------
# Visualization Libraries
# -----------------------------------------------------------------------------
RUN pip install \
    matplotlib \
    seaborn \
    plotly \
    bokeh \
    holoviews \
    hvplot \
    panel \
    altair

# -----------------------------------------------------------------------------
# Geospatial and Mapping
# -----------------------------------------------------------------------------
RUN pip install \
    cartopy \
    geopandas \
    shapely \
    pyproj \
    folium \
    geoviews

# -----------------------------------------------------------------------------
# Time Series Analysis
# -----------------------------------------------------------------------------
RUN pip install \
    tsfresh \
    sktime \
    statsmodels \
    pmdarima \
    prophet

# -----------------------------------------------------------------------------
# Machine Learning and Statistics
# -----------------------------------------------------------------------------
RUN pip install \
    scikit-learn \
    xgboost \
    lightgbm

# -----------------------------------------------------------------------------
# Data I/O and Serialization
# -----------------------------------------------------------------------------
RUN pip install \
    # JSON handling
    orjson \
    ujson \
    simplejson \
    # XML parsing
    lxml \
    xmltodict \
    beautifulsoup4 \
    # YAML
    pyyaml \
    # Excel support
    openpyxl \
    xlrd \
    # Parquet and Arrow
    pyarrow \
    fastparquet \
    # HDF5
    h5py \
    tables

# -----------------------------------------------------------------------------
# HTTP and API Clients
# -----------------------------------------------------------------------------
RUN pip install \
    requests \
    httpx \
    aiohttp \
    urllib3

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
RUN pip install \
    # Date/time handling
    python-dateutil \
    pytz \
    pendulum \
    # Data validation
    pydantic \
    # Progress bars
    tqdm \
    # Logging
    loguru \
    # Environment variables
    python-dotenv \
    # Caching
    joblib \
    # Functional programming utilities
    toolz \
    more-itertools

# =============================================================================
# Copy Example Files
# =============================================================================
COPY --chown=jupyter:jupyter examples/ /home/jupyter/examples/
COPY --chown=jupyter:jupyter scripts/ /home/jupyter/scripts/

# =============================================================================
# Make scripts executable
# =============================================================================
RUN chmod +x /home/jupyter/scripts/*.py

# =============================================================================
# Final Configuration
# =============================================================================
USER jupyter

# Configure Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py

# Expose Jupyter port
EXPOSE 8888

# Set default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
