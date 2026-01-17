# =============================================================================
# Data Science Jupyter Notebook Environment
# =============================================================================
# A comprehensive Jupyter Notebook environment for data science, statistics,
# analysis, and visualization tasks.
#
# Python: 3.13 (latest stable with full ecosystem support)
# Package Manager: uv (https://docs.astral.sh/uv/)
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
    JUPYTER_ENABLE_LAB=yes \
    DEBIAN_FRONTEND=noninteractive \
    # uv configuration
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

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
# Install uv via official installer
# =============================================================================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# =============================================================================
# Create Working Directory and Non-Root User
# =============================================================================
RUN useradd -m -s /bin/bash jupyter && \
    mkdir -p /home/jupyter/notebooks /home/jupyter/data /home/jupyter/examples/output /home/jupyter/scripts && \
    chown -R jupyter:jupyter /home/jupyter

WORKDIR /home/jupyter

# =============================================================================
# Python Package Installation with uv
# =============================================================================
# Copy project definition
COPY --chown=jupyter:jupyter pyproject.toml /home/jupyter/

# Initialize uv project and install dependencies
RUN uv lock && uv sync

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
RUN uv run jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = False" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/jupyter/.jupyter/jupyter_lab_config.py

# Expose Jupyter port
EXPOSE 8888

# Set default command
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
