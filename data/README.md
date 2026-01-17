# Data

Place your datasets in this directory.

This directory is mounted to `/home/jupyter/data` inside the container, making your data files accessible from Jupyter notebooks.

## Usage

When running the container with volume mount:

```bash
docker run -p 8888:8888 -v $(pwd)/data:/home/jupyter/data datascience-notebook
```

## Supported Formats

The environment supports reading various data formats:

- **CSV/TSV**: `pandas.read_csv()`
- **Excel**: `pandas.read_excel()` (xlsx, xls)
- **JSON**: `pandas.read_json()`, `orjson`, `ujson`
- **Parquet**: `pandas.read_parquet()`
- **HDF5**: `pandas.read_hdf()`, `h5py`
- **Feather**: `pandas.read_feather()`
- **XML**: `lxml`, `xmltodict`
- **YAML**: `yaml.safe_load()`

## Example

```python
import pandas as pd

# Read data from mounted directory
df = pd.read_csv('/home/jupyter/data/my_dataset.csv')
```
