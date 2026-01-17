# Quick Start Instructions

## Prerequisites

- Docker installed and running
- At least 4GB of available disk space (image size is ~3-4GB)
- Port 8888 available (or choose a different port)

## Build the Image

```bash
# Navigate to the project directory
cd /path/to/jupy2

# Build the Docker image
docker build -t datascience-notebook .
```

**Note:** First build takes 5-10 minutes to download and install all dependencies.

## Run the Container

### Basic Usage

```bash
docker run -p 8888:8888 datascience-notebook
```

Access Jupyter Lab at: **http://localhost:8888**

### With Persistent Storage (Recommended)

Save your notebooks and data between container restarts:

```bash
docker run -p 8888:8888 \
  -v $(pwd)/notebooks:/home/jupyter/notebooks \
  -v $(pwd)/data:/home/jupyter/data \
  datascience-notebook
```

### Run in Background

```bash
# Start in background
docker run -d -p 8888:8888 --name jupyter datascience-notebook

# View logs
docker logs jupyter

# Stop
docker stop jupyter

# Start again
docker start jupyter

# Remove container
docker rm jupyter
```

### Custom Port

If port 8888 is already in use:

```bash
docker run -p 9999:8888 datascience-notebook
```

Then access at: **http://localhost:9999**

## Verify Installation

Test that all packages are installed correctly:

```bash
docker run --rm datascience-notebook uv run python /home/jupyter/scripts/verify_imports.py
```

## Run Example Scripts

```bash
# Run a specific example
docker run --rm datascience-notebook uv run python /home/jupyter/examples/01_numpy_scipy_basics.py

# Run inside running container
docker exec -it jupyter uv run python /home/jupyter/examples/01_numpy_scipy_basics.py
```

## Access Container Shell

```bash
# If container is running
docker exec -it jupyter bash

# One-off shell
docker run --rm -it datascience-notebook bash
```

## Package Management with uv

This container uses [uv](https://docs.astral.sh/uv/) for package management. All Python commands should be run with `uv run`:

```bash
# Run Python
uv run python script.py

# Run Jupyter
uv run jupyter lab

# Check installed packages
uv tree
```

### Adding Packages at Runtime

To add packages temporarily in a running container:

```bash
docker exec -it jupyter uv add package-name
```

For permanent changes, edit `pyproject.toml` and rebuild the image.

## Security Note

The default configuration has **no authentication** for convenience in local development. For production or shared environments, use a token:

```bash
docker run -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secret-token \
  datascience-notebook \
  uv run jupyter lab --ip=0.0.0.0 --IdentityProvider.token='your-secret-token'
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using port 8888
lsof -i :8888

# Or use a different port
docker run -p 9999:8888 datascience-notebook
```

### Out of Memory

```bash
# Increase memory limit
docker run -p 8888:8888 --memory=4g datascience-notebook
```

### Network Issues During Build

If you encounter network errors during build, retry the command. Intermittent connectivity issues are common.

```bash
# Retry build
docker build -t datascience-notebook .
```

### Clean Build (No Cache)

If you need to rebuild from scratch:

```bash
docker build --no-cache -t datascience-notebook .
```
