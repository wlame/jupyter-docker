#!/usr/bin/env python3
# # From a Messy CSV to Clean, Validated Parquet
#
# Real exports are messy: mixed date formats, numbers with currency signs and
# thousands separators, inconsistent spelling, duplicates, and impossible values.
# This notebook generates such a file, cleans it with pandas, validates every
# row against a Pydantic model (keeping the rejects with their reasons), writes
# the clean rows to Parquet, and reports on them with DuckDB SQL.
#
# To clean your own export, put it in `/home/jupyter/data/` and set `DATA_FILE`;
# adjust the `Order` model to match its columns.
#
# - [pandas](https://pandas.pydata.org/docs/)
# - [Pydantic](https://docs.pydantic.dev/)
# - [DuckDB](https://duckdb.org/docs/)

import json
import os
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

# Outputs go to ./output next to the notebook (the kernel's working directory).
OUTPUT_DIR = os.path.abspath('output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = None  # e.g. '/home/jupyter/data/export.csv'
rng = np.random.default_rng(0)

# ## Generate a messy export
#
# 400 orders with the kinds of problems a spreadsheet export usually has.

if DATA_FILE:
    raw_path = DATA_FILE
else:
    n = 400
    countries = rng.choice(['Germany', 'germany ', 'DE', 'France', 'FR', 'Spain', 'spain', 'Italy'], size=n)
    dates = pd.date_range('2026-01-01', periods=n, freq='19h')
    date_strings = [d.strftime(rng.choice(['%Y-%m-%d', '%d/%m/%Y', '%d.%m.%Y'])) for d in dates]
    amounts = [rng.choice(['€{:,.2f}', '{:.2f}', '{:,.2f} EUR']).format(a) for a in rng.gamma(2.0, 60.0, size=n)]
    raw = pd.DataFrame({
        'Order ID': [f'ORD-{i:05d}' for i in range(n)],
        'Order Date': date_strings,
        'Country': countries,
        'Amount': amounts,
        'Quantity': rng.integers(1, 6, size=n).astype(str),
        'Email': [f'customer{i % 150}@example.com' for i in range(n)],
    })
    raw.loc[rng.choice(n, 12, replace=False), 'Quantity'] = rng.choice(['', 'n/a', '-1', 'two'], size=12)
    raw.loc[rng.choice(n, 6, replace=False), 'Email'] = 'not-an-email'
    raw = pd.concat([raw, raw.sample(15, random_state=0)])  # duplicated rows
    raw_path = os.path.join(OUTPUT_DIR, 'orders_messy.csv')
    raw.to_csv(raw_path, index=False)

raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
print(f"{len(raw)} rows; first lines of the raw file:")
print(raw.head(6).to_string(index=False))

# ## Clean the columns
#
# Each step fixes one kind of mess and is easy to adjust for your own data.

COUNTRY_CODES = {'germany': 'DE', 'de': 'DE', 'france': 'FR', 'fr': 'FR', 'spain': 'ES', 'italy': 'IT'}
clean = (
    raw.rename(columns=lambda c: c.strip().lower().replace(' ', '_'))
    .drop_duplicates()
    .assign(
        order_date=lambda d: pd.to_datetime(d['order_date'], format='mixed', dayfirst=True).dt.date,
        country=lambda d: d['country'].str.strip().str.lower().map(COUNTRY_CODES),
        amount=lambda d: pd.to_numeric(d['amount'].str.replace(r'[€,]|EUR', '', regex=True).str.strip(),
                                       errors='coerce'),
        quantity=lambda d: pd.to_numeric(d['quantity'], errors='coerce'),
    )
)
print(f"After de-duplication: {len(clean)} rows")

# ## Validate every row
#
# The model states what a valid order is. Rows that break a rule go to a reject
# file together with the reason, instead of disappearing silently.


class Order(BaseModel):
    """One valid order."""

    order_id: str = Field(pattern=r'^ORD-\d{5}$')
    order_date: date
    country: str = Field(min_length=2, max_length=2)
    amount: float = Field(gt=0, lt=10_000)
    quantity: int = Field(ge=1, le=100)
    email: str

    @field_validator('email')
    @classmethod
    def email_has_domain(cls, value: str) -> str:
        """A minimal email check: something@domain.tld."""
        local, _, domain = value.partition('@')
        if not local or '.' not in domain:
            raise ValueError("not an email address")
        return value.lower()


valid, rejects = [], []
for record in clean.replace({np.nan: None}).to_dict('records'):
    try:
        valid.append(Order.model_validate(record).model_dump())
    except ValidationError as error:
        rejects.append({**record, 'reason': '; '.join(f"{e['loc'][0]}: {e['msg']}" for e in error.errors())})
orders = pd.DataFrame(valid)
rejected = pd.DataFrame(rejects)
print(f"Valid {len(orders)}, rejected {len(rejected)}")
print(rejected[['order_id', 'reason']].head(6).to_string(index=False))

# ## Write Parquet and report with DuckDB
#
# Parquet keeps the cleaned types (dates, numbers) and compresses well; DuckDB
# queries the file directly with SQL.

parquet_path = os.path.join(OUTPUT_DIR, 'orders_clean.parquet')
orders.to_parquet(parquet_path, index=False)
rejected.to_csv(os.path.join(OUTPUT_DIR, 'orders_rejected.csv'), index=False)
print(f"CSV {os.path.getsize(raw_path) / 1024:.0f} KB -> Parquet {os.path.getsize(parquet_path) / 1024:.0f} KB")

report = duckdb.sql(
    """
    SELECT country,
           count(*) AS orders,
           round(sum(amount * quantity), 2) AS revenue,
           round(avg(amount), 2) AS avg_amount,
           count(DISTINCT email) AS customers
    FROM read_parquet(?)
    GROUP BY country
    ORDER BY revenue DESC
    """,
    params=[parquet_path],
).df()
print(report.to_string(index=False))

with open(os.path.join(OUTPUT_DIR, 'orders_summary.json'), 'w') as f:
    json.dump({'raw_rows': len(raw), 'valid': len(orders), 'rejected': len(rejected),
               'revenue_by_country': report.set_index('country')['revenue'].to_dict()}, f, indent=2)
print("Saved: orders_clean.parquet, orders_rejected.csv, orders_summary.json")
