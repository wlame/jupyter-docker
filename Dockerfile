# syntax=docker/dockerfile:1
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
#   docker build --target speech -t ds-speech .
#   docker build --target face -t ds-face .
#   docker build --target full -t ds-full .
#
# Python dependencies come from targets/<name>/pyproject.toml + uv.lock,
# both generated from targets/matrix.toml (see scripts/gen_targets.py).
# Requires BuildKit (cache mounts): DOCKER_BUILDKIT=1 or a buildx builder.
# =============================================================================

# =============================================================================
# BASE: Common utilities for all data science work
# =============================================================================
FROM ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90 AS base

LABEL org.opencontainers.image.authors="wlame" \
      org.opencontainers.image.source="https://github.com/wlame/jupyter-docker" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="ds-base: Base Python environment with common utilities for data science"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JUPYTER_ENABLE_LAB=yes \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install Python 3.13 and basic system dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    # Build essentials (needed for sdist-only packages, e.g. pmdarima)
    build-essential \
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

# Convenience `python` on PATH; /usr/bin/python3 stays the distro 3.12 so
# python3-apt keeps working. The venv (via uv) is the real interpreter.
RUN ln -s /usr/bin/python3.13 /usr/local/bin/python

# Install uv (version-pinned copy from the official distroless image)
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/

# Create the non-root user at UID 1000 (replacing the stock ubuntu user) so
# bind-mounted host directories keep sane ownership.
RUN userdel -r ubuntu \
    && useradd -m -u 1000 -s /bin/bash jupyter \
    && mkdir -p /home/jupyter/notebooks /home/jupyter/data /home/jupyter/examples/output /home/jupyter/scripts \
    && chown -R jupyter:jupyter /home/jupyter

WORKDIR /home/jupyter

# Copy base pyproject.toml + lockfile and install
COPY --chown=jupyter:jupyter targets/base/pyproject.toml targets/base/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/base/verify_imports.py /home/jupyter/scripts/verify_imports.py

# Sync from the committed lockfile as the jupyter user (.venv stays user-writable)
USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Copy examples, tests, and scripts
COPY --chown=jupyter:jupyter examples/ /home/jupyter/examples/
COPY --chown=jupyter:jupyter tests/ /home/jupyter/tests/

# Configure Jupyter.
# No token/password lines: jupyter-server generates a random token per start
# (printed in the logs) and honors the JUPYTER_TOKEN env var for a fixed one.
RUN uv run --no-project jupyter lab --generate-config \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.port = 8888" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py

EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8888/api || exit 1
CMD ["uv", "run", "--no-project", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]


# =============================================================================
# SCIENTIFIC: Core numerical computing (inherits from base)
# =============================================================================
FROM base AS scientific
LABEL org.opencontainers.image.description="ds-scientific: Scientific computing with NumPy, SciPy, and Pandas"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/scientific/pyproject.toml targets/scientific/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/scientific/verify_imports.py /home/jupyter/scripts/verify_scientific.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# VISUALIZATION: Charts and dashboards (inherits from base)
# =============================================================================
FROM base AS visualization
LABEL org.opencontainers.image.description="ds-visualization: Data visualization with Matplotlib, Seaborn, Plotly, and Bokeh"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/visualization/pyproject.toml targets/visualization/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/visualization/verify_imports.py /home/jupyter/scripts/verify_visualization.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# DATAIO: Data formats and databases (inherits from base)
# =============================================================================
FROM base AS dataio
LABEL org.opencontainers.image.description="ds-dataio: Data I/O for Parquet, HDF5, Excel, and databases"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/dataio/pyproject.toml targets/dataio/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/dataio/verify_imports.py /home/jupyter/scripts/verify_dataio.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# ML: Classical machine learning (inherits from scientific)
# =============================================================================
FROM scientific AS ml
LABEL org.opencontainers.image.description="ds-ml: Classical machine learning with scikit-learn, XGBoost, and LightGBM"

COPY --chown=jupyter:jupyter targets/ml/pyproject.toml targets/ml/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/ml/verify_imports.py /home/jupyter/scripts/verify_ml.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# DEEPLEARN: Neural networks (inherits from ml)
# =============================================================================
FROM ml AS deeplearn
LABEL org.opencontainers.image.description="ds-deeplearn: Deep learning with PyTorch and TensorFlow"

COPY --chown=jupyter:jupyter targets/deeplearn/pyproject.toml targets/deeplearn/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/deeplearn/verify_imports.py /home/jupyter/scripts/verify_deeplearn.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# VISION: Image processing (inherits from base, needs numpy)
# =============================================================================
FROM base AS vision
LABEL org.opencontainers.image.description="ds-vision: Computer vision and image processing"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
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

COPY --chown=jupyter:jupyter targets/vision/pyproject.toml targets/vision/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/vision/verify_imports.py /home/jupyter/scripts/verify_vision.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# AUDIO: Audio processing (inherits from base, needs torch)
# =============================================================================
FROM base AS audio
LABEL org.opencontainers.image.description="ds-audio: Audio processing and analysis"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/audio/pyproject.toml targets/audio/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/audio/verify_imports.py /home/jupyter/scripts/verify_audio.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# GEOSPATIAL: Maps and GIS (inherits from scientific)
# =============================================================================
FROM scientific AS geospatial
LABEL org.opencontainers.image.description="ds-geospatial: Geospatial analysis and mapping"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
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

COPY --chown=jupyter:jupyter targets/geospatial/pyproject.toml targets/geospatial/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/geospatial/verify_imports.py /home/jupyter/scripts/verify_geospatial.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# TIMESERIES: Time series analysis (inherits from scientific)
# =============================================================================
FROM scientific AS timeseries
LABEL org.opencontainers.image.description="ds-timeseries: Time series analysis and forecasting"

COPY --chown=jupyter:jupyter targets/timeseries/pyproject.toml targets/timeseries/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/timeseries/verify_imports.py /home/jupyter/scripts/verify_timeseries.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# NLP: Natural language processing (inherits from base, needs torch)
# =============================================================================
FROM base AS nlp
LABEL org.opencontainers.image.description="ds-nlp: Natural language processing"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/nlp/pyproject.toml targets/nlp/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/nlp/verify_imports.py /home/jupyter/scripts/verify_nlp.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# SPEECH: Speech recognition and text-to-speech (inherits from base, needs torch)
# =============================================================================
FROM base AS speech
LABEL org.opencontainers.image.description="ds-speech: Speech recognition and text-to-speech synthesis"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/speech/pyproject.toml targets/speech/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/speech/verify_imports.py /home/jupyter/scripts/verify_speech.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# FACE: Face detection, recognition, and analysis (inherits from base)
# =============================================================================
FROM base AS face
LABEL org.opencontainers.image.description="ds-face: Face detection, recognition, analysis, and generation"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/face/pyproject.toml targets/face/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/face/verify_imports.py /home/jupyter/scripts/verify_face.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project


# =============================================================================
# FULL: Complete data science environment (inherits from base; installs the
# union of every specialized target's system libraries, then the full lockfile)
# =============================================================================
FROM base AS full
LABEL org.opencontainers.image.description="ds-full: Complete data science environment with all libraries"

USER root
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Scientific computing
    gfortran \
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
    # Audio / Speech
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    # Face (dlib compilation)
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/full/pyproject.toml targets/full/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/full/verify_imports.py /home/jupyter/scripts/verify_imports.py
COPY --chown=jupyter:jupyter targets/ /home/jupyter/targets/

# Expose every target's verification script as scripts/verify_<target>.py
RUN for dir in /home/jupyter/targets/*/; do \
        target="$(basename "$dir")"; \
        cp "$dir/verify_imports.py" "/home/jupyter/scripts/verify_$target.py"; \
    done \
    && chown jupyter:jupyter /home/jupyter/scripts/verify_*.py

USER jupyter
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project
