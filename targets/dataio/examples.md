<!-- GENERATED FILE — do not edit by hand. Source: targets/matrix.toml and the example titles. -->
# Examples in `ghcr.io/wlame/jupyter-docker:dataio`

Data I/O for Parquet, HDF5, Excel, and databases.

Open a notebook and run it top to bottom (**Run > Run All Cells**). Files an
example writes go to `output/` in this folder. Your own data belongs in
`/home/jupyter/data`: mount a host folder there with
`-v "$(pwd)/data:/home/jupyter/data"`.

| Notebook | What it covers | Image it comes from |
|---|---|---|
| [08_data_io_serialization](08_data_io_serialization.ipynb) | Data I/O and Serialization | `dataio` |
| [18_sqlalchemy_database](18_sqlalchemy_database.ipynb) | SQLAlchemy ORM and Columnar Data Formats | `dataio` |
| [23_modern_data_formats](23_modern_data_formats.ipynb) | Modern Data Formats: Delta Lake, Excel, SPSS, Zarr, netCDF, Fast SQL Loading | `dataio` |
| [32_dataframe_engines](32_dataframe_engines.ipynb) | One Query, Four Engines: pandas, Polars, DuckDB, Ibis, and ADBC | `dataio` |
| [49_messy_csv_to_parquet](49_messy_csv_to_parquet.ipynb) | From a Messy CSV to Clean, Validated Parquet | `dataio` |

Each notebook has a matching `.py` file with the same code, which you can run
with `python <name>.py` or open as a notebook through Jupytext.
