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
#   docker build --target optimization -t ds-optimization .
#   docker build --target jax -t ds-jax .
#   docker build --target probabilistic -t ds-probabilistic .
#   docker build --target nlp -t ds-nlp .
#   docker build --target speech -t ds-speech .
#   docker build --target face -t ds-face .
#   docker build --target full -t ds-full .
#
# Python dependencies come from targets/<name>/pyproject.toml + uv.lock,
# both generated from targets/matrix.toml (see scripts/gen_targets.py).
# Requires BuildKit (cache mounts): DOCKER_BUILDKIT=1 or a buildx builder.
#
# PYTHON_VERSION selects the interpreter for the whole stage chain. Each target
# supports the versions listed for it in the matrix (`just build <target>` and
# CI pick them from there). The default suits most targets; the TensorFlow ones
# (deeplearn, face, full) are 3.13-only, so a raw docker build of those needs
# --build-arg PYTHON_VERSION=3.13 (uv's requires-python check stops it otherwise).
# =============================================================================

ARG PYTHON_VERSION=3.14

# =============================================================================
# BASE: Common utilities for all data science work
# =============================================================================
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55 AS base
ARG PYTHON_VERSION

LABEL org.opencontainers.image.authors="wlame" \
      org.opencontainers.image.source="https://github.com/wlame/jupyter-docker" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.description="ds-base: Base Python environment with common utilities for data science"

# DASK_DISTRIBUTED__DASHBOARD__LINK makes Dask print dashboard links through
# jupyter-server-proxy (installed with Dask), since only Jupyter's port is exposed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JUPYTER_ENABLE_LAB=yes \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python${PYTHON_VERSION} \
    DASK_DISTRIBUTED__DASHBOARD__LINK="/proxy/{port}/status"

# Install the selected Python and basic system dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    # Runtime shared libs the removed build toolchain used to pull in transitively:
    #  - libgomp1 (libgomp.so.1): OpenMP runtime for scikit-learn, xgboost, lightgbm.
    #  - libpython3.X (libpython3.X.so.1.0): needed by extensions that link
    #    libpython directly, e.g. torchcodec's custom-ops lib in audio/speech/full.
    libgomp1 \
    libpython${PYTHON_VERSION} \
    # Network utilities
    curl \
    wget \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# No build toolchain here: every target installs from prebuilt wheels, so the
# runtime images ship no compilers or headers. The only packages that compile
# from source (dlib) live in the face/full builder stages below, which install
# build-essential + cmake + python3.X-dev and hand off just the built venv.

# Convenience `python` on PATH; /usr/bin/python3 stays the distro 3.12 so
# python3-apt keeps working. The venv (via uv) is the real interpreter.
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

# Install uv (version-pinned copy from the official distroless image)
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

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

# Sync from the committed lockfile as the jupyter user (.venv stays user-writable).
# USER takes numeric IDs (1000:1000 = jupyter, 0:0 = root) so runtimes such as
# Kubernetes runAsNonRoot can verify the final user is not root.
USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Copy tests and the build-time helpers every stage uses: model baking and example
# selection (each stage ships only its own and its ancestors' examples).
COPY --chown=jupyter:jupyter tests/ /home/jupyter/tests/
COPY --chown=jupyter:jupyter scripts/bake_models.sh scripts/select_examples.sh /home/jupyter/scripts/

# Ship this target's examples: its own and its ancestors' (targets/base/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/base/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt

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
    CMD ["sh", "-c", "curl -fsS http://localhost:8888/api || exit 1"]
CMD ["uv", "run", "--no-project", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]


# =============================================================================
# SCIENTIFIC: Core numerical computing (inherits from base)
# =============================================================================
FROM base AS scientific
LABEL org.opencontainers.image.description="ds-scientific: Scientific computing with NumPy, SciPy, and Pandas"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/scientific/pyproject.toml targets/scientific/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/scientific/verify_imports.py /home/jupyter/scripts/verify_scientific.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/scientific/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/scientific/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# VISUALIZATION: Charts and dashboards (inherits from base)
# =============================================================================
FROM base AS visualization
LABEL org.opencontainers.image.description="ds-visualization: Data visualization with Matplotlib, Seaborn, Plotly, and Bokeh"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16t64 \
    libjpeg-turbo8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/visualization/pyproject.toml targets/visualization/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/visualization/verify_imports.py /home/jupyter/scripts/verify_visualization.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/visualization/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/visualization/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# DATAIO: Data formats and databases (inherits from base)
# =============================================================================
FROM base AS dataio
LABEL org.opencontainers.image.description="ds-dataio: Data I/O for Parquet, HDF5, Excel, and databases"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-103-1t64 \
    libhdf5-hl-100t64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/dataio/pyproject.toml targets/dataio/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/dataio/verify_imports.py /home/jupyter/scripts/verify_dataio.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/dataio/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/dataio/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# ML: Classical machine learning (inherits from scientific)
# =============================================================================
FROM scientific AS ml
LABEL org.opencontainers.image.description="ds-ml: Classical machine learning with scikit-learn, XGBoost, and LightGBM"

COPY --chown=jupyter:jupyter targets/ml/pyproject.toml targets/ml/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/ml/verify_imports.py /home/jupyter/scripts/verify_ml.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/ml/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/ml/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# DEEPLEARN: Neural networks (inherits from ml)
# =============================================================================
FROM ml AS deeplearn
LABEL org.opencontainers.image.description="ds-deeplearn: Deep learning with PyTorch and TensorFlow"

COPY --chown=jupyter:jupyter targets/deeplearn/pyproject.toml targets/deeplearn/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/deeplearn/verify_imports.py /home/jupyter/scripts/verify_deeplearn.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/deeplearn/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/deeplearn/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# VISION: Image processing (inherits from base, needs numpy)
# =============================================================================
FROM base AS vision
LABEL org.opencontainers.image.description="ds-vision: Computer vision and image processing"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libfreetype6 \
    libpng16-16t64 \
    libjpeg-turbo8 \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/vision/pyproject.toml targets/vision/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/vision/verify_imports.py /home/jupyter/scripts/verify_vision.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Pre-bake model weights so example tests run offline (see scripts/bake_models.sh)
RUN bash /home/jupyter/scripts/bake_models.sh vision

# Ship this target's examples: its own and its ancestors' (targets/vision/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/vision/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# AUDIO: Audio processing (inherits from base, needs torch)
# =============================================================================
FROM base AS audio
LABEL org.opencontainers.image.description="ds-audio: Audio processing and analysis"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libsndfile1 \
    ffmpeg \
    # libatomic.so.1 for pedalboard's native module
    libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/audio/pyproject.toml targets/audio/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/audio/verify_imports.py /home/jupyter/scripts/verify_audio.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/audio/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/audio/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# GEOSPATIAL: Maps and GIS (inherits from scientific)
# =============================================================================
FROM scientific AS geospatial
LABEL org.opencontainers.image.description="ds-geospatial: Geospatial analysis and mapping"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16t64 \
    libgeos-c1t64 \
    libproj25 \
    proj-data \
    proj-bin \
    libgdal34t64 \
    gdal-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/geospatial/pyproject.toml targets/geospatial/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/geospatial/verify_imports.py /home/jupyter/scripts/verify_geospatial.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/geospatial/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/geospatial/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# TIMESERIES: Time series analysis (inherits from scientific)
# =============================================================================
FROM scientific AS timeseries
LABEL org.opencontainers.image.description="ds-timeseries: Time series analysis and forecasting"

COPY --chown=jupyter:jupyter targets/timeseries/pyproject.toml targets/timeseries/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/timeseries/verify_imports.py /home/jupyter/scripts/verify_timeseries.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/timeseries/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/timeseries/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# OPTIMIZATION: LP, MILP, convex, routing, scheduling (inherits from scientific)
# =============================================================================
FROM scientific AS optimization
LABEL org.opencontainers.image.description="ds-optimization: Mathematical optimization: linear, integer, convex, routing, and scheduling"

# The cbc command is the solver PuLP and Pyomo call (OR-Tools embeds its own).
USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/optimization/pyproject.toml targets/optimization/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/optimization/verify_imports.py /home/jupyter/scripts/verify_optimization.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/optimization/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/optimization/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# JAX: JIT, autodiff, Flax, Optax, NumPyro (inherits from scientific; CPU jaxlib)
# =============================================================================
FROM scientific AS jax
LABEL org.opencontainers.image.description="ds-jax: JAX: JIT compilation, automatic differentiation, and accelerated numerics"

COPY --chown=jupyter:jupyter targets/jax/pyproject.toml targets/jax/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/jax/verify_imports.py /home/jupyter/scripts/verify_jax.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/jax/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/jax/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# PROBABILISTIC: PyMC, nutpie, ArviZ, Bambi, PreliZ (inherits from jax)
# =============================================================================
FROM jax AS probabilistic
LABEL org.opencontainers.image.description="ds-probabilistic: Bayesian statistics and probabilistic programming"

COPY --chown=jupyter:jupyter targets/probabilistic/pyproject.toml targets/probabilistic/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/probabilistic/verify_imports.py /home/jupyter/scripts/verify_probabilistic.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Ship this target's examples: its own and its ancestors' (targets/probabilistic/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/probabilistic/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# NLP: Natural language processing (inherits from base, needs torch)
# =============================================================================
FROM base AS nlp
LABEL org.opencontainers.image.description="ds-nlp: Natural language processing"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/nlp/pyproject.toml targets/nlp/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/nlp/verify_imports.py /home/jupyter/scripts/verify_nlp.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Pre-bake model weights so example tests run offline (see scripts/bake_models.sh)
RUN bash /home/jupyter/scripts/bake_models.sh nlp

# Ship this target's examples: its own and its ancestors' (targets/nlp/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/nlp/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# SPEECH: Speech recognition and text-to-speech (inherits from base, needs torch)
# =============================================================================
FROM base AS speech
LABEL org.opencontainers.image.description="ds-speech: Speech recognition and text-to-speech synthesis"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/speech/pyproject.toml targets/speech/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/speech/verify_imports.py /home/jupyter/scripts/verify_speech.py

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

# Pre-bake model weights so example tests run offline (see scripts/bake_models.sh)
RUN bash /home/jupyter/scripts/bake_models.sh speech

# Ship this target's examples: its own and its ancestors' (targets/speech/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/speech/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# FACE: Face detection, recognition, and analysis (inherits from base)
#
# dlib has no wheel and compiles from source, so it is built in a throwaway
# BUILDER stage that carries the C/C++ toolchain (build-essential, cmake) plus
# python3.X-dev for the Python bindings. The published `face` image then copies
# only the finished venv, so no compiler or header package ships in the runtime.
# =============================================================================
FROM base AS face-builder
ARG PYTHON_VERSION

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python${PYTHON_VERSION}-dev \
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

COPY --chown=jupyter:jupyter targets/face/pyproject.toml targets/face/uv.lock /home/jupyter/

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

FROM base AS face
LABEL org.opencontainers.image.description="ds-face: Face detection, recognition, analysis, and generation"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    # PortAudio for sounddevice, which mediapipe imports
    libportaudio2 \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libfreetype6 \
    libpng16-16t64 \
    libjpeg-turbo8 \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/face/pyproject.toml targets/face/uv.lock /home/jupyter/
COPY --chown=jupyter:jupyter targets/face/verify_imports.py /home/jupyter/scripts/verify_face.py

# Take the fully built venv (with the compiled dlib) from the builder. Base sets
# UV_LINK_MODE=copy, so the venv is self-contained and relocatable across stages.
COPY --from=face-builder --chown=jupyter:jupyter /home/jupyter/.venv /home/jupyter/.venv

USER 1000:1000
# Pre-bake face-alignment weights so example tests run offline. DeepFace's
# attribute models are intentionally NOT baked (~1.5 GB, reliably hosted, and
# the example already skips them offline). See scripts/bake_models.sh.
RUN bash /home/jupyter/scripts/bake_models.sh face

# Ship this target's examples: its own and its ancestors' (targets/face/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/face/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt


# =============================================================================
# FULL: Complete data science environment (inherits from base; installs the
# union of every specialized target's system libraries, then the full lockfile)
#
# Like face, `full` compiles dlib from source, so the build toolchain and all
# -dev headers live in a throwaway BUILDER stage; the published `full` image
# copies only the finished venv and installs the runtime shared libraries.
# =============================================================================
FROM base AS full-builder
ARG PYTHON_VERSION

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Build toolchain (dlib compiles from source; also builds pure-python sdists)
    build-essential \
    cmake \
    python${PYTHON_VERSION}-dev \
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
    libgdal-dev \
    # HDF5
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter targets/full/pyproject.toml targets/full/uv.lock /home/jupyter/

USER 1000:1000
RUN --mount=type=cache,target=/home/jupyter/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-install-project

FROM base AS full
LABEL org.opencontainers.image.description="ds-full: Complete data science environment with all libraries"

USER 0:0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Scientific computing
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    # Visualization
    libfreetype6 \
    libpng16-16t64 \
    libjpeg-turbo8 \
    # Vision/OpenCV
    libgl1 \
    libglib2.0-0 \
    # Geospatial
    libgeos-c1t64 \
    libproj25 \
    proj-data \
    proj-bin \
    libgdal34t64 \
    gdal-bin \
    # HDF5
    libhdf5-103-1t64 \
    libhdf5-hl-100t64 \
    # Audio / Speech
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    libatomic1 \
    # Face (mediapipe -> sounddevice)
    libportaudio2 \
    # Optimization (the solver PuLP and Pyomo call)
    coinor-cbc \
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

# Take the fully built venv (with the compiled dlib) from the builder.
COPY --from=full-builder --chown=jupyter:jupyter /home/jupyter/.venv /home/jupyter/.venv

USER 1000:1000
# Pre-bake every target's model weights so example tests run offline
RUN bash /home/jupyter/scripts/bake_models.sh full

# Ship this target's examples: its own and its ancestors' (targets/full/examples.txt).
RUN --mount=type=bind,source=examples,target=/tmp/examples \
    --mount=type=bind,source=targets/full/examples.txt,target=/tmp/examples.txt \
    bash /home/jupyter/scripts/select_examples.sh /tmp/examples /tmp/examples.txt
