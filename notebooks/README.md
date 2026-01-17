# Notebooks

Place your Jupyter notebooks (`.ipynb` files) in this directory.

This directory is mounted to `/home/jupyter/notebooks` inside the container, providing persistent storage for your work.

## Usage

When running the container with volume mount:

```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/jupyter/notebooks datascience-notebook
```

Any notebooks created or modified in Jupyter Lab will be saved here and persist after the container stops.
