#!/usr/bin/env python3
"""Verify all deeplearn target imports are working correctly.

GENERATED FILE — do not edit by hand.
Source of truth: targets/matrix.toml (regenerate: python3 scripts/gen_targets.py).
"""

import sys

IMPORTS = [
    # --- first: torch family, before anything that may load TensorFlow ---
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("torchvision", "torchvision"),
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
    # --- scientific ---
    ("bottleneck", "bottleneck"),
    ("duckdb", "duckdb"),
    ("matplotlib", "matplotlib"),
    ("numexpr", "numexpr"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pint", "pint"),
    ("polars", "polars"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("sympy", "sympy"),
    ("xarray", "xarray"),
    # --- ml ---
    ("catboost", "catboost"),
    ("imblearn", "imbalanced-learn"),
    ("lightgbm", "lightgbm"),
    ("mapie", "mapie"),
    ("optuna", "optuna"),
    ("sklearn", "scikit-learn"),
    ("shap", "shap"),
    ("skops", "skops"),
    ("skrub", "skrub"),
    ("umap", "umap-learn"),
    ("xgboost", "xgboost"),
    # --- deeplearn ---
    ("accelerate", "accelerate"),
    ("einops", "einops"),
    ("keras", "keras"),
    ("lightning", "lightning"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("tensorboard", "tensorboard"),
    ("tensorflow", "tensorflow"),
    ("torchmetrics", "torchmetrics"),
]


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying DEEPLEARN target imports")
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
        print("\nAll deeplearn imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
