#!/usr/bin/env python3
# # Modern Tabular and Array Tools: polars, DuckDB, Arrow, xarray, pint
#
# Demonstrates the fast DataFrame and labeled-array stack next to pandas:
# lazy polars queries, SQL over in-memory frames with DuckDB, zero-copy Arrow
# interchange, labeled N-D arrays with xarray, physical units with pint, and the
# numexpr/bottleneck accelerators pandas picks up automatically.
#
# - [polars](https://docs.pola.rs/)
# - [DuckDB](https://duckdb.org/docs/api/python/overview)
# - [PyArrow](https://arrow.apache.org/docs/python/)
# - [xarray](https://docs.xarray.dev/)
# - [pint](https://pint.readthedocs.io/)

import os

import bottleneck as bn
import duckdb
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
import pint
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=0)

# ## Synthetic Sales Data

n_rows = 20_000
sales = pd.DataFrame({
    'day': pd.date_range('2026-01-01', periods=n_rows, freq='h'),
    'store': rng.choice(['north', 'south', 'east', 'west'], size=n_rows),
    'units': rng.poisson(lam=12, size=n_rows),
    'price': rng.uniform(2.0, 20.0, size=n_rows).round(2),
})
sales['revenue'] = sales['units'] * sales['price']
print(f"Rows: {len(sales):,}  Columns: {list(sales.columns)}")

# ## polars: Lazy Aggregation

frame = pl.from_pandas(sales)
per_store = (
    frame.lazy()
    .filter(pl.col('units') > 0)
    .group_by('store')
    .agg(
        pl.col('revenue').sum().alias('revenue'),
        pl.col('units').mean().alias('mean_units'),
        pl.len().alias('orders'),
    )
    .sort('revenue', descending=True)
    .collect()
)
print(per_store)

# ## DuckDB: SQL over DataFrames and Parquet

parquet_path = os.path.join(OUTPUT_DIR, 'duckdb_sales.parquet')
pq.write_table(pa.Table.from_pandas(sales, preserve_index=False), parquet_path)

con = duckdb.connect()
# con.sql() returns a relation, which converts straight to polars, pandas, or Arrow.
monthly = con.sql(
    """
    SELECT date_trunc('month', day) AS month, store, sum(revenue) AS revenue
    FROM read_parquet(?)
    GROUP BY ALL
    ORDER BY month, store
    """,
    params=[parquet_path],
).pl()
print(monthly.head(8))

# DuckDB also queries in-scope DataFrames by name (here the polars frame).
top_store = con.execute("SELECT store FROM per_store ORDER BY revenue DESC LIMIT 1").fetchone()[0]
print(f"Top store (queried from the polars frame): {top_store}")

# ## PyArrow: Zero-Copy Interchange

table = per_store.to_arrow()
print(f"Arrow schema: {table.schema}")
round_trip = pl.from_arrow(table)
print(f"Round trip equal: {round_trip.equals(per_store)}")

# ## xarray: Labeled Temperature Grid

times = pd.date_range('2026-01-01', periods=365, freq='D')
lat = np.linspace(-60, 60, 25)
lon = np.linspace(-180, 175, 72)
season = 10 * np.sin(2 * np.pi * (times.dayofyear.to_numpy() - 80) / 365)
temperature = (
    25 - 0.4 * np.abs(lat)[None, :, None]
    + season[:, None, None] * np.sign(lat)[None, :, None]
    + rng.normal(scale=1.5, size=(len(times), len(lat), len(lon)))
)
grid = xr.DataArray(
    temperature,
    dims=('time', 'lat', 'lon'),
    coords={'time': times, 'lat': lat, 'lon': lon},
    name='temperature',
    attrs={'units': 'degC'},
)
monthly_zonal = grid.resample(time='MS').mean().mean('lon')
print(f"Grid shape: {dict(grid.sizes)} -> monthly zonal mean {dict(monthly_zonal.sizes)}")

fig, ax = plt.subplots(figsize=(9, 4))
monthly_zonal.plot(ax=ax, x='time', y='lat', cmap='coolwarm')
ax.set_title('Monthly zonal-mean temperature (synthetic)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xarray_zonal_mean.png'), dpi=120)
plt.show()
print("Saved: xarray_zonal_mean.png")

# ## pint: Units-Aware Arithmetic

ureg = pint.UnitRegistry()
distance = 42.195 * ureg.kilometer
duration = 2 * ureg.hour + 1 * ureg.minute + 9 * ureg.second
pace = (duration / distance).to(ureg.minute / ureg.kilometer)
speed = (distance / duration).to(ureg.meter / ureg.second)
print(f"Marathon pace: {pace:.2f~P}, speed: {speed:.2f~P}")

# ## numexpr and bottleneck

a = rng.normal(size=1_000_000)
b = rng.normal(size=1_000_000)
fused = ne.evaluate('2 * a**2 + 3 * b - 1')
print(f"numexpr matches NumPy: {np.allclose(fused, 2 * a**2 + 3 * b - 1)}")

window = bn.move_mean(sales['revenue'].to_numpy(), window=24, min_count=1)
print(f"bottleneck 24h moving mean, last value: {window[-1]:.2f}")

print("\nDone.")
