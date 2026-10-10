#!/usr/bin/env python3
"""
One Query, Four Engines: pandas, Polars, DuckDB, Ibis, and ADBC
===============================================================
Writes one million synthetic orders to a Parquet file once, then answers the
same question four ways: eager pandas, the lazy Polars API, DuckDB SQL, and
Ibis (one dataframe API compiled to SQL, here executed by DuckDB). The four
answers are checked against each other and timed. The last section moves an
Arrow table into SQLite and back with ADBC, which keeps data columnar instead
of converting it row by row.

For PostgreSQL the same ADBC calls work with
`adbc_driver_postgresql.dbapi.connect("postgresql://user:pass@host/db")`.

pandas:  https://pandas.pydata.org/docs/
Polars:  https://docs.pola.rs/
DuckDB:  https://duckdb.org/docs/api/python/overview
Ibis:    https://ibis-project.org/
ADBC:    https://arrow.apache.org/adbc/
"""

import json
import os
import time
from datetime import datetime

import adbc_driver_sqlite.dbapi as sqlite_adbc
import duckdb
import ibis
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from ibis import _

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ORDERS = 1_000_000
START = datetime(2026, 1, 1)
REGIONS = ['north', 'south', 'east', 'west', 'central']
CATEGORIES = ['books', 'games', 'garden', 'kitchen', 'music', 'office', 'sports', 'toys']
STATUSES = ['shipped', 'returned', 'cancelled']
TIMING_RUNS = 3

rng = np.random.default_rng(seed=0)

# =============================================================================
# One Parquet file shared by every engine
# =============================================================================
print("=" * 60)
print("Synthetic Orders Written Once to Parquet")
print("=" * 60)

order_seconds = rng.integers(0, 730 * 24 * 3600, size=N_ORDERS)
orders = pa.table({
    'order_id': np.arange(N_ORDERS, dtype=np.int64),
    'order_date': np.datetime64('2025-01-01T00:00:00', 'us') + order_seconds.astype('timedelta64[s]'),
    'region': rng.choice(REGIONS, size=N_ORDERS),
    'category': rng.choice(CATEGORIES, size=N_ORDERS),
    'quantity': rng.integers(1, 11, size=N_ORDERS),
    'unit_price': rng.uniform(2.0, 120.0, size=N_ORDERS).round(2),
    'discount': rng.choice([0.0, 0.05, 0.1, 0.2, 0.3], size=N_ORDERS),
    'status': rng.choice(STATUSES, size=N_ORDERS, p=[0.85, 0.1, 0.05]),
})
parquet_path = os.path.join(OUTPUT_DIR, 'orders.parquet')
pq.write_table(orders, parquet_path)
print(f"Rows: {orders.num_rows:,}  Size on disk: {os.path.getsize(parquet_path) / 1e6:.1f} MB")
print("Question: revenue, order count, and mean discount per region and category")
print(f"for shipped orders since {START:%Y-%m-%d}, highest revenue first.")


# =============================================================================
# The same query, four ways
# =============================================================================
def query_pandas(path: str) -> pd.DataFrame:
    """Eager pandas: load everything, then filter, derive, and aggregate."""
    frame = pd.read_parquet(path)
    shipped = frame[(frame['status'] == 'shipped') & (frame['order_date'] >= START)]
    shipped = shipped.assign(revenue=shipped['quantity'] * shipped['unit_price'] * (1 - shipped['discount']))
    return (
        shipped.groupby(['region', 'category'], as_index=False)
        .agg(revenue=('revenue', 'sum'), orders=('order_id', 'count'), mean_discount=('discount', 'mean'))
        .sort_values('revenue', ascending=False, ignore_index=True)
    )


def query_polars(path: str) -> pl.DataFrame:
    """Lazy Polars: the optimizer pushes the filter and column selection into the scan."""
    return (
        pl.scan_parquet(path)
        .filter((pl.col('status') == 'shipped') & (pl.col('order_date') >= START))
        .with_columns(revenue=pl.col('quantity') * pl.col('unit_price') * (1 - pl.col('discount')))
        .group_by('region', 'category')
        .agg(
            pl.col('revenue').sum(),
            pl.len().alias('orders'),
            pl.col('discount').mean().alias('mean_discount'),
        )
        .sort('revenue', descending=True)
        .collect()
    )


DUCKDB_SQL = """
SELECT region, category,
       sum(quantity * unit_price * (1 - discount)) AS revenue,
       count(*) AS orders,
       avg(discount) AS mean_discount
FROM read_parquet(?)
WHERE status = 'shipped' AND order_date >= ?
GROUP BY region, category
ORDER BY revenue DESC
"""


def query_duckdb(path: str) -> pd.DataFrame:
    """DuckDB SQL straight over the Parquet file."""
    return duckdb.connect().execute(DUCKDB_SQL, [path, START]).df()


def ibis_expression(path: str) -> ibis.Table:
    """Ibis: a lazy dataframe expression that compiles to SQL for any backend."""
    orders_table = ibis.duckdb.connect().read_parquet(path)
    return (
        orders_table.filter((_.status == 'shipped') & (_.order_date >= START))
        .mutate(revenue=_.quantity * _.unit_price * (1 - _.discount))
        .group_by(['region', 'category'])
        .aggregate(revenue=_.revenue.sum(), orders=_.count(), mean_discount=_.discount.mean())
        .order_by(ibis.desc('revenue'))
    )


def query_ibis(path: str) -> pd.DataFrame:
    """Run the Ibis expression on DuckDB and return pandas."""
    return ibis_expression(path).to_pandas()


ENGINES = {
    'pandas': query_pandas,
    'polars (lazy)': query_polars,
    'duckdb (SQL)': query_duckdb,
    'ibis (on duckdb)': query_ibis,
}


def normalized(result: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Common shape for comparing engines: pandas, fixed dtypes, sorted by key."""
    frame = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    frame = frame.astype({'orders': 'int64', 'revenue': 'float64', 'mean_discount': 'float64'})
    return frame.sort_values(['region', 'category'], ignore_index=True)


print("\n" + "=" * 60)
print("Ibis compiles the expression to SQL")
print("=" * 60)
print(ibis.to_sql(ibis_expression(parquet_path)))

print("\n" + "=" * 60)
print("Answers agree across engines")
print("=" * 60)
results = {name: normalized(query(parquet_path)) for name, query in ENGINES.items()}
reference = results['pandas']
for name, frame in results.items():
    same_keys = frame[['region', 'category']].equals(reference[['region', 'category']])
    same_counts = frame['orders'].equals(reference['orders'])
    same_values = np.allclose(frame[['revenue', 'mean_discount']], reference[['revenue', 'mean_discount']], rtol=1e-9)
    if not (same_keys and same_counts and same_values):
        raise SystemExit(f"{name} disagrees with pandas")
    print(f"  {name:18} {len(frame)} groups, matches pandas")

answer = query_polars(parquet_path)
print(answer.head(5))
answer.write_csv(os.path.join(OUTPUT_DIR, 'engines_answer.csv'))
print("Saved: engines_answer.csv")

# =============================================================================
# Timing (best of several runs; each run reads the Parquet file again)
# =============================================================================
print("\n" + "=" * 60)
print(f"Timing, best of {TIMING_RUNS} runs")
print("=" * 60)


def best_seconds(query, path: str) -> float:
    """Fastest wall-clock time of TIMING_RUNS calls."""
    durations = []
    for _run in range(TIMING_RUNS):
        started = time.perf_counter()
        query(path)
        durations.append(time.perf_counter() - started)
    return min(durations)


timings = {name: round(best_seconds(query, parquet_path), 4) for name, query in ENGINES.items()}
fastest = min(timings.values())
for name, seconds in sorted(timings.items(), key=lambda item: item[1]):
    print(f"  {name:18} {seconds * 1000:8.1f} ms   {seconds / fastest:5.1f}x")
with open(os.path.join(OUTPUT_DIR, 'engines_timings.json'), 'w') as f:
    json.dump({'rows': N_ORDERS, 'seconds': timings}, f, indent=2)
print("Saved: engines_timings.json")
print("pandas reads every column into memory first; the other three prune columns and")
print("filter inside the Parquet scan, then aggregate in parallel native code.")

# =============================================================================
# ADBC: Arrow tables in and out of a database without row conversion
# =============================================================================
print("\n" + "=" * 60)
print("ADBC: Arrow In, Arrow Out (SQLite)")
print("=" * 60)

sqlite_path = os.path.join(OUTPUT_DIR, 'engines.sqlite')
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)

with sqlite_adbc.connect(sqlite_path) as conn:
    with conn.cursor() as cursor:
        ingested = cursor.adbc_ingest('region_category', answer.to_arrow(), mode='create')
        cursor.execute(
            'SELECT region, sum(revenue) AS revenue, sum(orders) AS orders '
            'FROM region_category GROUP BY region ORDER BY revenue DESC'
        )
        by_region = cursor.fetch_arrow_table()
    conn.commit()

print(f"Ingested {ingested} rows as an Arrow table; query returned {type(by_region).__name__}:")
print(pl.from_arrow(by_region))
if sum(by_region['orders'].to_pylist()) != int(answer['orders'].sum()):
    raise SystemExit("ADBC round trip lost orders")
print(f"Saved: {os.path.basename(sqlite_path)}")

print("\nDone.")
