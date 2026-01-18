#!/usr/bin/env python3
"""Verify all imports for the FULL target (all libraries)."""

import sys
import os

# Suppress TensorFlow warnings during import verification
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# All imports organized by target
IMPORTS = {
    "base": [
        ("IPython", "ipython"),
        ("jupyter", "jupyter"),
        ("jupyterlab", "jupyterlab"),
        ("orjson", "orjson"),
        ("ujson", "ujson"),
        ("simplejson", "simplejson"),
        ("lxml", "lxml"),
        ("xmltodict", "xmltodict"),
        ("bs4", "beautifulsoup4"),
        ("yaml", "pyyaml"),
        ("requests", "requests"),
        ("httpx", "httpx"),
        ("aiohttp", "aiohttp"),
        ("pydantic", "pydantic"),
        ("tqdm", "tqdm"),
        ("loguru", "loguru"),
        ("dotenv", "python-dotenv"),
        ("joblib", "joblib"),
        ("toolz", "toolz"),
        ("more_itertools", "more-itertools"),
        ("dateutil", "python-dateutil"),
        ("pytz", "pytz"),
        ("pendulum", "pendulum"),
    ],
    "scientific": [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("statsmodels", "statsmodels"),
        ("sympy", "sympy"),
    ],
    "visualization": [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("bokeh", "bokeh"),
        ("holoviews", "holoviews"),
        ("hvplot", "hvplot"),
        ("panel", "panel"),
        ("altair", "altair"),
    ],
    "dataio": [
        ("pyarrow", "pyarrow"),
        ("fastparquet", "fastparquet"),
        ("h5py", "h5py"),
        ("tables", "tables"),
        ("openpyxl", "openpyxl"),
        ("xlrd", "xlrd"),
        ("sqlalchemy", "sqlalchemy"),
    ],
    "ml": [
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("imblearn", "imbalanced-learn"),
        ("optuna", "optuna"),
    ],
    "deeplearn": [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
    ],
    "vision": [
        ("PIL", "pillow"),
        ("cv2", "opencv-python-headless"),
        ("skimage", "scikit-image"),
        ("imageio", "imageio"),
        ("ultralytics", "ultralytics"),
    ],
    "audio": [
        ("torchaudio", "torchaudio"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("pydub", "pydub"),
        ("audioread", "audioread"),
    ],
    "geospatial": [
        ("cartopy", "cartopy"),
        ("geopandas", "geopandas"),
        ("shapely", "shapely"),
        ("pyproj", "pyproj"),
        ("folium", "folium"),
        ("geoviews", "geoviews"),
    ],
    "timeseries": [
        ("tsfresh", "tsfresh"),
        ("sktime", "sktime"),
        ("pmdarima", "pmdarima"),
        ("prophet", "prophet"),
    ],
    "nlp": [
        ("spacy", "spacy"),
        ("nltk", "nltk"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence-transformers"),
        ("tokenizers", "tokenizers"),
    ],
}


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying FULL target imports (all libraries)")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    all_errors = []

    for target_name, imports in IMPORTS.items():
        print(f"\n[{target_name.upper()}]")
        passed = 0
        failed = 0

        for module_name, package_name in imports:
            try:
                __import__(module_name)
                print(f"  ✓ {package_name}")
                passed += 1
            except ImportError as e:
                print(f"  ✗ {package_name}: {e}")
                failed += 1
                all_errors.append((target_name, package_name, str(e)))

        total_passed += passed
        total_failed += failed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if total_failed > 0:
        print("\nFailed imports:")
        for target, pkg, err in all_errors:
            print(f"  [{target}] {pkg}: {err}")
        sys.exit(1)
    else:
        print("\nAll imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
