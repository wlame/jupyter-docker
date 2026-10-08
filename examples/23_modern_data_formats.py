#!/usr/bin/env python3
"""
Modern Data Formats: Delta Lake, Excel, SPSS, Zarr, netCDF, Fast SQL Loading
=============================================================================
Round-trips a dataset through the formats data work runs into: Delta Lake
tables (deltalake, no Spark), formatted Excel (xlsxwriter) read back with the
fast calamine engine, SPSS/SAS/Stata files (pyreadstat), chunked arrays (zarr),
netCDF (netCDF4), and database-to-DataFrame loading with connectorx.

The cloud filesystems (s3fs, gcsfs, adlfs) and database drivers (psycopg,
pymysql) need real endpoints, so this example only names them: every
fsspec-aware reader accepts `s3://`, `gs://`, and `abfs://` URLs once the
matching package is installed, and SQLAlchemy uses psycopg/pymysql through
`postgresql+psycopg://` and `mysql+pymysql://` URLs.

deltalake:  https://delta-io.github.io/delta-rs/
calamine:   https://github.com/dimastbk/python-calamine
pyreadstat: https://github.com/Roche/pyreadstat
zarr:       https://zarr.readthedocs.io/
netCDF4:    https://unidata.github.io/netcdf4-python/
connectorx: https://sfu-db.github.io/connector-x/
"""

import os
import shutil
import sqlite3

import connectorx as cx
import duckdb
import netCDF4
import numpy as np
import pandas as pd
import polars as pl
import pyreadstat
import zarr
from deltalake import DeltaTable, write_deltalake

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

survey = pd.DataFrame({
    'respondent': np.arange(1, 501),
    'region': rng.choice(['north', 'south', 'east', 'west'], size=500),
    'age': rng.integers(18, 80, size=500),
    'satisfaction': rng.integers(1, 6, size=500),
    'income': rng.normal(42_000, 9_000, size=500).round(0),
})

# =============================================================================
# Delta Lake — versioned tables without Spark
# =============================================================================
print("=" * 60)
print("deltalake: Versioned Table")
print("=" * 60)

delta_path = os.path.join(OUTPUT_DIR, 'delta_survey')
shutil.rmtree(delta_path, ignore_errors=True)
write_deltalake(delta_path, survey.iloc[:250])
write_deltalake(delta_path, survey.iloc[250:], mode='append')
table = DeltaTable(delta_path)
print(f"Delta version: {table.version()}, rows: {table.to_pandas().shape[0]}")
first_version = DeltaTable(delta_path, version=0).to_pandas()
print(f"Time travel to version 0: {len(first_version)} rows")

# DuckDB queries the Arrow table deltalake hands over, by its variable name.
arrow_table = table.to_pyarrow_table()
by_region = duckdb.sql(
    "SELECT region, round(avg(satisfaction), 2) AS satisfaction FROM arrow_table GROUP BY region ORDER BY region"
).df()
print(by_region.to_string(index=False))

# =============================================================================
# Excel — write with xlsxwriter, read back with calamine
# =============================================================================
print("\n" + "=" * 60)
print("Excel: xlsxwriter + calamine")
print("=" * 60)

xlsx_path = os.path.join(OUTPUT_DIR, 'survey_report.xlsx')
with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
    survey.to_excel(writer, sheet_name='responses', index=False)
    workbook = writer.book
    sheet = writer.sheets['responses']
    money = workbook.add_format({'num_format': '#,##0'})
    sheet.set_column('E:E', 12, money)
    sheet.conditional_format('D2:D501', {'type': '3_color_scale'})
read_back = pd.read_excel(xlsx_path, engine='calamine')
print(f"Read {len(read_back)} rows with the calamine engine; columns match: "
      f"{list(read_back.columns) == list(survey.columns)}")

# =============================================================================
# SPSS — pyreadstat with value labels
# =============================================================================
print("\n" + "=" * 60)
print("pyreadstat: SPSS with Value Labels")
print("=" * 60)

sav_path = os.path.join(OUTPUT_DIR, 'survey.sav')
pyreadstat.write_sav(
    survey,
    sav_path,
    column_labels={'satisfaction': 'Overall satisfaction (1-5)'},
    variable_value_labels={'satisfaction': {1: 'very low', 2: 'low', 3: 'neutral', 4: 'high', 5: 'very high'}},
)
frame, meta = pyreadstat.read_sav(sav_path, apply_value_formats=True)
print(f"Column label: {meta.column_names_to_labels['satisfaction']}")
print(frame['satisfaction'].value_counts().sort_index().to_string())

# =============================================================================
# Zarr — chunked N-D arrays
# =============================================================================
print("\n" + "=" * 60)
print("zarr: Chunked Array Store")
print("=" * 60)

zarr_path = os.path.join(OUTPUT_DIR, 'measurements.zarr')
shutil.rmtree(zarr_path, ignore_errors=True)
store = zarr.open_array(zarr_path, mode='w', shape=(365, 100, 100), chunks=(30, 50, 50), dtype='float32')
store[:] = rng.normal(size=(365, 100, 100)).astype('float32')
reopened = zarr.open_array(zarr_path, mode='r')
print(f"Shape {reopened.shape}, chunks {reopened.chunks}, January mean {float(reopened[:31].mean()):.4f}")

# =============================================================================
# netCDF — self-describing scientific files
# =============================================================================
print("\n" + "=" * 60)
print("netCDF4: Write and Read")
print("=" * 60)

nc_path = os.path.join(OUTPUT_DIR, 'temperature.nc')
with netCDF4.Dataset(nc_path, 'w') as nc:
    nc.createDimension('time', 12)
    nc.createDimension('station', 5)
    temp = nc.createVariable('temperature', 'f4', ('time', 'station'))
    temp.units = 'degC'
    temp[:] = 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, 12))[:, None] + rng.normal(size=(12, 5))
with netCDF4.Dataset(nc_path) as nc:
    values = nc.variables['temperature']
    print(f"Variable shape {values.shape}, units {values.units}, max {float(values[:].max()):.2f}")

# =============================================================================
# connectorx — fast SQL into DataFrames
# =============================================================================
print("\n" + "=" * 60)
print("connectorx: SQL to DataFrame")
print("=" * 60)

db_path = os.path.join(OUTPUT_DIR, 'survey.sqlite')
if os.path.exists(db_path):
    os.remove(db_path)
with sqlite3.connect(db_path) as conn:
    survey.to_sql('survey', conn, index=False)
query = 'SELECT region, COUNT(*) AS n, AVG(income) AS income FROM survey GROUP BY region'
loaded = cx.read_sql(f'sqlite://{os.path.abspath(db_path)}', query, return_type='polars')
print(loaded.sort('region'))
assert isinstance(loaded, pl.DataFrame)

print("\nDone.")
