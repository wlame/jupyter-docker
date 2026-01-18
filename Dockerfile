# =============================================================================
# Multi-Target Data Science Docker Image
# =============================================================================
# Build specific targets for different use cases:
#
#   docker build --target base -t ds-base .
#   docker build --target scientific -t ds-scientific .
#   docker build --target visualization -t ds-visualization .
#   docker build --target dataio -t ds-dataio .
#   docker build --target ml -t ds-ml .
#   docker build --target deeplearn -t ds-deeplearn .
#   docker build --target vision -t ds-vision .
#   docker build --target audio -t ds-audio .
#   docker build --target geospatial -t ds-geospatial .
#   docker build --target timeseries -t ds-timeseries .
#   docker build --target nlp -t ds-nlp .
#   docker build --target full -t ds-full .
#
# =============================================================================

# =============================================================================
# BASE: Common utilities for all data science work
# =============================================================================
FROM ubuntu:24.04 AS base

LABEL maintainer="wlame"
LABEL description="Data Science Base Image with common utilities"
LABEL version="2.0"

# Environment Configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JUPYTER_ENABLE_LAB=yes \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:$PATH"

# Install Python 3.13 and basic system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # Required for XML parsing
    libxml2-dev \
    libxslt1-dev \
    # Network utilities
    curl \
    wget \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.13 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Create non-root user and directories
RUN useradd -m -s /bin/bash jupyter \
    && mkdir -p /home/jupyter/notebooks /home/jupyter/data /home/jupyter/examples/output /home/jupyter/scripts \
    && chown -R jupyter:jupyter /home/jupyter

WORKDIR /home/jupyter

# Copy base pyproject.toml and install
COPY --chown=jupyter:jupyter targets/base/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/base/verify_imports.py /home/jupyter/scripts/verify_imports.py

RUN uv lock && uv sync --no-install-project

# Copy examples and scripts
COPY --chown=jupyter:jupyter examples/ /home/jupyter/examples/

# Configure Jupyter
USER jupyter
RUN uv run --no-project jupyter lab --generate-config \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.port = 8888" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.token = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.password = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py

EXPOSE 8888
CMD ["uv", "run", "--no-project", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]


# =============================================================================
# SCIENTIFIC: Core numerical computing (inherits from base)
# =============================================================================
FROM base AS scientific

USER root

# Install scientific dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with scientific version (includes all base + scientific deps)
COPY --chown=jupyter:jupyter targets/scientific/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/scientific/verify_imports.py /home/jupyter/scripts/verify_scientific.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# VISUALIZATION: Charts and dashboards (inherits from base)
# =============================================================================
FROM base AS visualization

USER root

# Install visualization dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with visualization version (includes all base + visualization deps)
COPY --chown=jupyter:jupyter targets/visualization/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/visualization/verify_imports.py /home/jupyter/scripts/verify_visualization.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# DATAIO: Data formats and databases (inherits from base)
# =============================================================================
FROM base AS dataio

USER root

# Install dataio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with dataio version (includes all base + dataio deps)
COPY --chown=jupyter:jupyter targets/dataio/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/dataio/verify_imports.py /home/jupyter/scripts/verify_dataio.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# ML: Classical machine learning (inherits from scientific)
# =============================================================================
FROM scientific AS ml

USER root

# Replace pyproject.toml with ml version (includes all base + scientific + ml deps)
COPY --chown=jupyter:jupyter targets/ml/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/ml/verify_imports.py /home/jupyter/scripts/verify_ml.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# DEEPLEARN: Neural networks (inherits from ml)
# =============================================================================
FROM ml AS deeplearn

USER root

# Replace pyproject.toml with deeplearn version (includes all base + scientific + ml + deeplearn deps)
COPY --chown=jupyter:jupyter targets/deeplearn/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/deeplearn/verify_imports.py /home/jupyter/scripts/verify_deeplearn.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# VISION: Image processing (inherits from base, needs numpy)
# =============================================================================
FROM base AS vision

USER root

# Install vision dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with vision version (includes all base + vision deps)
COPY --chown=jupyter:jupyter targets/vision/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/vision/verify_imports.py /home/jupyter/scripts/verify_vision.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# AUDIO: Audio processing (inherits from base, needs torch)
# =============================================================================
FROM base AS audio

USER root

# Install audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with audio version (includes all base + audio deps)
COPY --chown=jupyter:jupyter targets/audio/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/audio/verify_imports.py /home/jupyter/scripts/verify_audio.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# GEOSPATIAL: Maps and GIS (inherits from scientific)
# =============================================================================
FROM scientific AS geospatial

USER root

# Install geospatial dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgdal-dev \
    gdal-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with geospatial version (includes all deps)
COPY --chown=jupyter:jupyter targets/geospatial/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/geospatial/verify_imports.py /home/jupyter/scripts/verify_geospatial.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# TIMESERIES: Time series analysis (inherits from scientific)
# =============================================================================
FROM scientific AS timeseries

USER root

# Replace pyproject.toml with timeseries version (includes all deps)
COPY --chown=jupyter:jupyter targets/timeseries/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/timeseries/verify_imports.py /home/jupyter/scripts/verify_timeseries.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# NLP: Natural language processing (inherits from base, needs torch)
# =============================================================================
FROM base AS nlp

USER root

# Install NLP dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Replace pyproject.toml with nlp version (includes all deps)
COPY --chown=jupyter:jupyter targets/nlp/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/nlp/verify_imports.py /home/jupyter/scripts/verify_nlp.py

# Install with uv lock and sync
RUN rm -f /home/jupyter/uv.lock && uv lock && uv sync --no-install-project

USER jupyter


# =============================================================================
# FULL: Complete data science environment (all libraries)
# =============================================================================
FROM ubuntu:24.04 AS full

LABEL maintainer="wlame"
LABEL description="Complete Data Science Environment with all libraries"
LABEL version="2.0"

# Environment Configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JUPYTER_ENABLE_LAB=yes \
    DEBIAN_FRONTEND=noninteractive \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:$PATH"

# Install all system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Scientific computing
    libopenblas-dev \
    liblapack-dev \
    # Visualization
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # Vision/OpenCV
    libgl1 \
    libglib2.0-0 \
    # Geospatial
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgdal-dev \
    gdal-bin \
    # HDF5
    libhdf5-dev \
    # XML
    libxml2-dev \
    libxslt1-dev \
    # Audio
    libsndfile1 \
    ffmpeg \
    # Network utilities
    curl \
    wget \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.13 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Create non-root user and directories
RUN useradd -m -s /bin/bash jupyter \
    && mkdir -p /home/jupyter/notebooks /home/jupyter/data /home/jupyter/examples/output /home/jupyter/scripts \
    && chown -R jupyter:jupyter /home/jupyter

WORKDIR /home/jupyter

# Copy full pyproject.toml and install all packages
COPY --chown=jupyter:jupyter targets/full/pyproject.toml /home/jupyter/
COPY --chown=jupyter:jupyter targets/full/verify_imports.py /home/jupyter/scripts/verify_imports.py

RUN uv lock && uv sync --no-install-project

# Copy examples and scripts
COPY --chown=jupyter:jupyter examples/ /home/jupyter/examples/
COPY --chown=jupyter:jupyter targets/ /home/jupyter/targets/

# Copy all verification scripts
COPY --chown=jupyter:jupyter targets/base/verify_imports.py /home/jupyter/scripts/verify_base.py
COPY --chown=jupyter:jupyter targets/scientific/verify_imports.py /home/jupyter/scripts/verify_scientific.py
COPY --chown=jupyter:jupyter targets/visualization/verify_imports.py /home/jupyter/scripts/verify_visualization.py
COPY --chown=jupyter:jupyter targets/dataio/verify_imports.py /home/jupyter/scripts/verify_dataio.py
COPY --chown=jupyter:jupyter targets/ml/verify_imports.py /home/jupyter/scripts/verify_ml.py
COPY --chown=jupyter:jupyter targets/deeplearn/verify_imports.py /home/jupyter/scripts/verify_deeplearn.py
COPY --chown=jupyter:jupyter targets/vision/verify_imports.py /home/jupyter/scripts/verify_vision.py
COPY --chown=jupyter:jupyter targets/audio/verify_imports.py /home/jupyter/scripts/verify_audio.py
COPY --chown=jupyter:jupyter targets/geospatial/verify_imports.py /home/jupyter/scripts/verify_geospatial.py
COPY --chown=jupyter:jupyter targets/timeseries/verify_imports.py /home/jupyter/scripts/verify_timeseries.py
COPY --chown=jupyter:jupyter targets/nlp/verify_imports.py /home/jupyter/scripts/verify_nlp.py

# Configure Jupyter
USER jupyter
RUN uv run --no-project jupyter lab --generate-config \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.port = 8888" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.token = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.password = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py

EXPOSE 8888
CMD ["uv", "run", "--no-project", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
