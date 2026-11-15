#!/usr/bin/env python3
# # One API for Local and Cloud Files
#
# fsspec gives every storage system the same interface: a URL picks the backend
# (`file://`, `memory://`, `s3://`, `gs://`, `abfs://`), and pandas, Polars,
# DuckDB, and plain Python read and write through it. Code written against a
# local folder then runs unchanged against a bucket. This notebook works through
# local and in-memory storage offline, shows compression and caching, and runs
# the same steps on S3, Google Cloud Storage, or Azure when you set a URL.
#
# Cloud access uses each provider's usual credentials from the environment
# (AWS_ACCESS_KEY_ID / AWS_PROFILE, GOOGLE_APPLICATION_CREDENTIALS,
# AZURE_STORAGE_CONNECTION_STRING). Nothing here stores keys.
#
# - [fsspec](https://filesystem-spec.readthedocs.io/)
# - [s3fs](https://s3fs.readthedocs.io/) · [gcsfs](https://gcsfs.readthedocs.io/) · [adlfs](https://github.com/fsspec/adlfs)

import json
import os

import duckdb
import fsspec
import numpy as np
import pandas as pd
import polars as pl

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set one of these to a writable prefix, e.g. 's3://my-bucket/scratch'.
CLOUD_URLS = {
    's3': os.environ.get('S3_URL'),
    'gcs': os.environ.get('GCS_URL'),
    'azure': os.environ.get('AZURE_URL'),  # abfs://container/path
}
rng = np.random.default_rng(0)
sales = pd.DataFrame({
    'date': pd.date_range('2026-01-01', periods=5_000, freq='h'),
    'store': rng.choice(['north', 'south', 'east', 'west'], size=5_000),
    'amount': rng.gamma(2.0, 30.0, size=5_000).round(2),
})

# ## The same code for every location
#
# `store_and_summarize` only sees a URL prefix. It writes partitioned Parquet,
# lists what it wrote, and reads it back three ways.


def store_and_summarize(prefix: str) -> dict:
    """Write sales as one Parquet file per store under prefix, then read it back."""
    fs, root = fsspec.core.url_to_fs(prefix)
    fs.makedirs(root, exist_ok=True)
    for store, part in sales.groupby('store'):
        fs.makedirs(f'{root}/store={store}', exist_ok=True)  # local disks need the folder; buckets ignore it
        with fs.open(f'{root}/store={store}/part-0.parquet', 'wb') as f:
            part.drop(columns='store').to_parquet(f, index=False)
    files = sorted(fs.glob(f'{root}/store=*/*.parquet'))
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    urls = [fs.unstrip_protocol(path) for path in files]
    # pandas: reads through fsspec directly from the URL.
    pandas_total = sum(pd.read_parquet(url)['amount'].sum() for url in urls)
    # Polars: give it file objects opened by fsspec.
    with fs.open(files[0], 'rb') as f:
        first = pl.read_parquet(f)
    return {'protocol': protocol, 'files': len(files), 'pandas_total': round(float(pandas_total), 2),
            'first_file_rows': first.height}


local = store_and_summarize(os.path.join(OUTPUT_DIR, 'lake'))
memory = store_and_summarize('memory://scratch/lake')
print(f"local : {local}")
print(f"memory: {memory}")
assert local['pandas_total'] == memory['pandas_total'] == round(float(sales['amount'].sum()), 2)

# ## DuckDB on the same files
#
# DuckDB reads Hive-style folders (`store=north/`) and turns the folder name into
# a column. With the httpfs extension it reads `s3://` URLs the same way.

by_store = duckdb.sql(
    "SELECT store, count(*) AS orders, round(sum(amount), 2) AS revenue "
    "FROM read_parquet(?, hive_partitioning = true) GROUP BY store ORDER BY store",
    params=[os.path.join(OUTPUT_DIR, 'lake', 'store=*', '*.parquet')],
).df()
print(by_store.to_string(index=False))

# ## Compression and caching through the URL
#
# Compression is inferred from the file name. A `simplecache::` prefix keeps a
# local copy of a remote file, so the second read costs no download.

csv_url = 'memory://scratch/sales.csv.gz'
sales.to_csv(csv_url, index=False, compression='infer')
compressed = fsspec.filesystem('memory').size('/scratch/sales.csv.gz')
print(f"gzip CSV in memory: {compressed / 1024:.0f} KB (uncompressed about {len(sales.to_csv(index=False)) / 1024:.0f} KB)")

cache_dir = os.path.join(OUTPUT_DIR, 'fsspec_cache')
for attempt in range(2):
    with fsspec.open(f'simplecache::{csv_url}', 'rb', compression='gzip',
                     simplecache={'cache_storage': cache_dir}) as f:
        rows = len(pd.read_csv(f))
print(f"read {rows} rows twice; the second read comes from {cache_dir}")

# ## The cloud, when configured
#
# Each block runs only when its URL is set. Credentials come from the provider's
# standard environment variables or config files.

results = {'local': local, 'memory': memory}
for name, url in CLOUD_URLS.items():
    if not url:
        print(f"{name}: skipped (set {name.upper()}_URL to a writable prefix to try it)")
        continue
    try:
        results[name] = store_and_summarize(url.rstrip('/') + '/jupyter-docker-example')
        print(f"{name}: {results[name]}")
    except Exception as error:  # report and continue: one provider's error shouldn't hide the rest
        print(f"{name}: failed with {type(error).__name__}: {error}")

with open(os.path.join(OUTPUT_DIR, 'files_anywhere.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: files_anywhere.json")
