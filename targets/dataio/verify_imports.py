#!/usr/bin/env python3
"""Verify all dataio target imports are working correctly.

GENERATED FILE — do not edit by hand.
Source of truth: targets/matrix.toml (regenerate: python3 scripts/gen_targets.py).
"""

import sys

IMPORTS = [
    # --- base ---
    ("aiohttp", "aiohttp"),
    ("bs4", "beautifulsoup4"),
    ("httpx", "httpx"),
    ("IPython", "ipython"),
    ("joblib", "joblib"),
    ("jupyter", "jupyter"),
    ("jupyter_resource_usage", "jupyter-resource-usage"),
    ("jupyterlab", "jupyterlab"),
    ("jupyterlab_git", "jupyterlab-git"),
    ("jupyterlab_lsp", "jupyterlab-lsp"),
    ("jupytext", "jupytext"),
    ("loguru", "loguru"),
    ("lxml", "lxml"),
    ("more_itertools", "more-itertools"),
    ("nbdime", "nbdime"),
    ("orjson", "orjson"),
    ("pendulum", "pendulum"),
    ("pip", "pip"),
    ("pydantic", "pydantic"),
    ("pytest", "pytest"),
    ("pytest_timeout", "pytest-timeout"),
    ("dateutil", "python-dateutil"),
    ("dotenv", "python-dotenv"),
    ("pylsp", "python-lsp-server"),
    ("pytz", "pytz"),
    ("yaml", "pyyaml"),
    ("requests", "requests"),
    ("simplejson", "simplejson"),
    ("toolz", "toolz"),
    ("tqdm", "tqdm"),
    ("ujson", "ujson"),
    ("xmltodict", "xmltodict"),
    # --- dataio ---
    ("adlfs", "adlfs"),
    ("connectorx", "connectorx"),
    ("deltalake", "deltalake"),
    ("duckdb", "duckdb"),
    ("fastparquet", "fastparquet"),
    ("gcsfs", "gcsfs"),
    ("h5py", "h5py"),
    ("netCDF4", "netcdf4"),
    ("numpy", "numpy"),
    ("openpyxl", "openpyxl"),
    ("pandas", "pandas"),
    ("polars", "polars"),
    ("psycopg", "psycopg"),
    ("psycopg_binary", "psycopg-binary"),
    ("pyarrow", "pyarrow"),
    ("pymysql", "pymysql"),
    ("pyreadstat", "pyreadstat"),
    ("python_calamine", "python-calamine"),
    ("s3fs", "s3fs"),
    ("sqlalchemy", "sqlalchemy"),
    ("tables", "tables"),
    ("xlrd", "xlrd"),
    ("xlsxwriter", "xlsxwriter"),
    ("zarr", "zarr"),
]


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying DATAIO target imports")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for module_name, package_name in IMPORTS:
        try:
            __import__(module_name)
            print(f"  \u2713 {package_name}")
            passed += 1
        except Exception as e:  # any import-time failure counts, not only ImportError
            message = f"{type(e).__name__}: {e}"
            print(f"  \u2717 {package_name}: {message}")
            failed += 1
            errors.append((package_name, message))

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFailed imports:")
        for pkg, err in errors:
            print(f"  - {pkg}: {err}")
        sys.exit(1)
    else:
        print("\nAll dataio imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
