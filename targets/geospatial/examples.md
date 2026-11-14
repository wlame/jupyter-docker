<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:geospatial`

Geospatial analysis and mapping.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [06_geospatial](06_geospatial.ipynb) | Geospatial Data Analysis | `geospatial` |
| [28_geospatial_raster_h3](28_geospatial_raster_h3.ipynb) | Rasters, Hexagons, and Street Networks: rasterio, rioxarray, H3, mapclassify, OSMnx | `geospatial` |
| [01_numpy_scipy_basics](01_numpy_scipy_basics.ipynb) | NumPy and SciPy Basics | `scientific` |
| [02_pandas_data_analysis](02_pandas_data_analysis.ipynb) | Pandas Data Analysis | `scientific` |
| [17_scipy_signal_processing](17_scipy_signal_processing.ipynb) | SciPy: Signal Processing and Optimization | `scientific` |
| [21_polars_duckdb_xarray](21_polars_duckdb_xarray.ipynb) | Modern Tabular and Array Tools: polars, DuckDB, Arrow, xarray, pint | `scientific` |
| [33_lorenz_numba_dask](33_lorenz_numba_dask.ipynb) | Lorenz Attractor: SymPy, Numba, FFT, and Dask | `scientific` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
