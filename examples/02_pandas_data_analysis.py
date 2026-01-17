#!/usr/bin/env python3
"""
Pandas Data Analysis
====================
Demonstrates data manipulation and analysis with Pandas.

Pandas: https://pandas.pydata.org/
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# =============================================================================
# Creating DataFrames
# =============================================================================
print("=" * 60)
print("Creating DataFrames")
print("=" * 60)

# From dictionary
data = {
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
    "salary": [75000, 65000, 85000, 70000, 72000],
    "start_date": pd.date_range("2020-01-01", periods=5, freq="3ME"),
}
df = pd.DataFrame(data)
print("Employee DataFrame:")
print(df)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

# =============================================================================
# Data Selection and Filtering
# =============================================================================
print("\n" + "=" * 60)
print("Data Selection and Filtering")
print("=" * 60)

# Select columns
print("Names column:")
print(df["name"])

# Filter rows
engineers = df[df["department"] == "Engineering"]
print("\nEngineers only:")
print(engineers)

# Multiple conditions
high_earners = df[(df["salary"] > 70000) & (df["age"] < 35)]
print("\nHigh earners under 35:")
print(high_earners)

# Using query
result = df.query("department == 'Marketing' and salary > 60000")
print("\nMarketing with salary > 60000:")
print(result)

# =============================================================================
# Aggregation and Grouping
# =============================================================================
print("\n" + "=" * 60)
print("Aggregation and Grouping")
print("=" * 60)

# Basic aggregation
print("Salary statistics:")
print(df["salary"].describe())

# Group by
dept_stats = df.groupby("department").agg(
    {"salary": ["mean", "min", "max", "count"], "age": "mean"}
)
print("\nStatistics by department:")
print(dept_stats)

# Custom aggregation
summary = df.groupby("department").apply(
    lambda x: pd.Series(
        {"avg_salary": x["salary"].mean(), "total_employees": len(x)}
    )
)
print("\nCustom summary by department:")
print(summary)

# =============================================================================
# Data Transformation
# =============================================================================
print("\n" + "=" * 60)
print("Data Transformation")
print("=" * 60)

# Add new columns
df["salary_category"] = pd.cut(
    df["salary"], bins=[0, 70000, 80000, float("inf")], labels=["Low", "Medium", "High"]
)

df["years_employed"] = (datetime.now() - df["start_date"]).dt.days / 365

print("DataFrame with new columns:")
print(df)

# Apply functions
df["salary_normalized"] = df.groupby("department")["salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("\nWith normalized salary:")
print(df[["name", "department", "salary", "salary_normalized"]])

# =============================================================================
# Pivot Tables
# =============================================================================
print("\n" + "=" * 60)
print("Pivot Tables")
print("=" * 60)

# Create sample sales data
sales_data = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=12, freq="ME"),
        "region": ["North", "South"] * 6,
        "product": ["A", "B", "A", "B"] * 3,
        "sales": np.random.randint(1000, 5000, 12),
    }
)

pivot = pd.pivot_table(
    sales_data, values="sales", index="region", columns="product", aggfunc="sum"
)
print("Sales pivot table:")
print(pivot)

# =============================================================================
# Time Series Operations
# =============================================================================
print("\n" + "=" * 60)
print("Time Series Operations")
print("=" * 60)

# Create time series
dates = pd.date_range("2024-01-01", periods=100, freq="D")
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

print("Time series head:")
print(ts.head(10))

# Resampling
weekly = ts.resample("W").mean()
print("\nWeekly resampled (first 5):")
print(weekly.head())

# Rolling statistics
rolling = ts.rolling(window=7).mean()
print("\n7-day rolling mean (last 5):")
print(rolling.tail())

# =============================================================================
# Data Cleaning
# =============================================================================
print("\n" + "=" * 60)
print("Data Cleaning")
print("=" * 60)

# Create messy data
messy = pd.DataFrame(
    {
        "A": [1, 2, np.nan, 4, 5],
        "B": [np.nan, 2, 3, np.nan, 5],
        "C": ["x", "y", "z", "x", "y"],
    }
)
print("Messy data:")
print(messy)

# Check for missing values
print(f"\nMissing values:\n{messy.isnull().sum()}")

# Fill missing values
filled = messy.fillna({"A": messy["A"].mean(), "B": 0})
print("\nFilled data:")
print(filled)

# Drop duplicates
with_dups = pd.DataFrame({"x": [1, 1, 2], "y": [1, 1, 3]})
print(f"\nWith duplicates:\n{with_dups}")
print(f"Without duplicates:\n{with_dups.drop_duplicates()}")

print("\n" + "=" * 60)
print("Pandas example complete!")
print("=" * 60)
