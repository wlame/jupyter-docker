#!/usr/bin/env python3
# # Working with Databases: SQL Safely, Arrow Fast, Servers Optional
#
# Three habits make database work in notebooks reliable: keep connection details
# out of the code, never build SQL by pasting strings together, and move data in
# bulk instead of row by row. This notebook practises all three on SQLite (built
# in, so it runs offline), then repeats the round trip on PostgreSQL and MySQL
# when you point it at a server.
#
# Set `DATABASE_URL` (postgresql://user:pass@host:5432/db) or `MYSQL_URL`
# (mysql://user:pass@host:3306/db) in the environment to include those sections.
# For a quick local server: `docker run -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres`.
#
# - [SQLAlchemy](https://docs.sqlalchemy.org/)
# - [psycopg](https://www.psycopg.org/psycopg3/docs/)
# - [ADBC](https://arrow.apache.org/adbc/)
# - [PyMySQL](https://pymysql.readthedocs.io/)

import json
import os
import sqlite3
import time
from urllib.parse import urlparse

import adbc_driver_sqlite.dbapi as sqlite_adbc
import numpy as np
import pandas as pd
import pyarrow as pa
from psycopg import sql
from sqlalchemy import create_engine, text

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSTGRES_URL = os.environ.get('DATABASE_URL')
MYSQL_URL = os.environ.get('MYSQL_URL')
ROWS = 200_000
rng = np.random.default_rng(0)
events = pa.table({
    'event_id': np.arange(ROWS, dtype=np.int64),
    'user_id': rng.integers(1, 5_000, size=ROWS),
    'kind': rng.choice(['view', 'click', 'purchase'], size=ROWS, p=[0.8, 0.17, 0.03]),
    'amount': np.where(rng.random(ROWS) < 0.03, rng.gamma(2.0, 25.0, size=ROWS), 0.0).round(2),
})
summary: dict[str, object] = {}

# ## Never paste values into SQL
#
# Values go in as parameters, and the driver sends them separately from the SQL
# text. Table and column names cannot be parameters, so psycopg's `sql` module
# quotes them as identifiers. This works without a server.

user_input = "click'; DROP TABLE events; --"
query = sql.SQL('SELECT {columns} FROM {table} WHERE kind = %s').format(
    columns=sql.SQL(', ').join(map(sql.Identifier, ['user_id', 'amount'])),
    table=sql.Identifier('analytics', 'events'),
)
print(f"Composed SQL: {query.as_string(None)}")
print(f"The value {user_input!r} travels as a parameter, never as SQL text.")

# ## Bulk load: row by row versus Arrow
#
# ADBC hands the database a whole Arrow table at once instead of one INSERT per
# row. On local SQLite, inside one transaction, the row-by-row path is still
# fairly quick, so the gap is modest; against a server over a network every row
# is a round trip, and bulk loading wins by orders of magnitude.

sqlite_path = os.path.join(OUTPUT_DIR, 'events.sqlite')
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)

started = time.perf_counter()
with sqlite3.connect(sqlite_path) as conn:
    conn.execute('CREATE TABLE events_rows (event_id INTEGER, user_id INTEGER, kind TEXT, amount REAL)')
    for row in events.to_pylist():
        conn.execute('INSERT INTO events_rows VALUES (?, ?, ?, ?)', tuple(row.values()))
row_seconds = time.perf_counter() - started

started = time.perf_counter()
with sqlite_adbc.connect(sqlite_path) as conn:
    with conn.cursor() as cursor:
        cursor.adbc_ingest('events', events, mode='create')
    conn.commit()
arrow_seconds = time.perf_counter() - started
print(f"{ROWS:,} rows into local SQLite: one INSERT per row {row_seconds:.2f} s, "
      f"Arrow ingest {arrow_seconds:.2f} s ({row_seconds / arrow_seconds:.0f}x faster)")
summary['sqlite_speedup'] = round(row_seconds / arrow_seconds, 1)

# ## Portable SQL with SQLAlchemy
#
# The engine URL is the only database-specific part. The same query text and
# parameters run on SQLite here and on PostgreSQL or MySQL below.

REPORT_SQL = text(
    'SELECT kind, COUNT(*) AS events, COUNT(DISTINCT user_id) AS users, ROUND(SUM(amount), 2) AS revenue '
    'FROM events WHERE amount >= :min_amount OR kind <> :skip GROUP BY kind ORDER BY events DESC'
)


def report(url: str) -> pd.DataFrame:
    """Run the report query through SQLAlchemy and return a DataFrame."""
    engine = create_engine(url)
    with engine.connect() as conn:
        return pd.read_sql(REPORT_SQL, conn, params={'min_amount': 0, 'skip': 'none'})


sqlite_report = report(f'sqlite:///{sqlite_path}')
print(sqlite_report.to_string(index=False))
summary['sqlite'] = sqlite_report.to_dict('records')


def safe_url(url: str) -> str:
    """The URL with its password hidden, for printing."""
    parts = urlparse(url)
    return url.replace(parts.password, '***') if parts.password else url


# ## PostgreSQL, when configured
#
# psycopg for ordinary queries, ADBC's PostgreSQL driver for the bulk load (it
# uses COPY under the hood).

if POSTGRES_URL:
    import adbc_driver_postgresql.dbapi as pg_adbc
    import psycopg

    try:
        with pg_adbc.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cursor:
                cursor.adbc_ingest('events', events, mode='replace')
            conn.commit()
        with psycopg.connect(POSTGRES_URL) as conn:
            count = conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier('events'))).fetchone()[0]
        pg_report = report(POSTGRES_URL.replace('postgresql://', 'postgresql+psycopg://', 1))
        print(f"PostgreSQL at {safe_url(POSTGRES_URL)}: {count:,} rows")
        print(pg_report.to_string(index=False))
        summary['postgres'] = pg_report.to_dict('records')
    except Exception as error:  # a wrong URL or a stopped server should not hide the rest
        print(f"PostgreSQL at {safe_url(POSTGRES_URL)} failed: {type(error).__name__}: {error}")
else:
    print("PostgreSQL: skipped (set DATABASE_URL to try it)")

# ## MySQL, when configured

if MYSQL_URL:
    try:
        engine_url = MYSQL_URL.replace('mysql://', 'mysql+pymysql://', 1)
        create_engine(engine_url).connect().close()
        with create_engine(engine_url).begin() as conn:
            events.slice(0, 50_000).to_pandas().to_sql('events', conn, if_exists='replace', index=False,
                                                      chunksize=5_000, method='multi')
        mysql_report = report(engine_url)
        print(f"MySQL at {safe_url(MYSQL_URL)} (first 50,000 rows):")
        print(mysql_report.to_string(index=False))
        summary['mysql'] = mysql_report.to_dict('records')
    except Exception as error:
        print(f"MySQL at {safe_url(MYSQL_URL)} failed: {type(error).__name__}: {error}")
else:
    print("MySQL: skipped (set MYSQL_URL to try it)")

with open(os.path.join(OUTPUT_DIR, 'databases_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("Saved: events.sqlite, databases_summary.json")
