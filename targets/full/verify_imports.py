#!/usr/bin/env python3
"""Verify all full target imports are working correctly.

GENERATED FILE — do not edit by hand.
Source of truth: targets/matrix.toml (regenerate: python3 scripts/gen_targets.py).
"""

import sys

IMPORTS = [
    # --- first: torch family, before anything that may load TensorFlow ---
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("torchvision", "torchvision"),
    ("torchcodec", "torchcodec"),
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
    ("rich", "rich"),
    ("simplejson", "simplejson"),
    ("tenacity", "tenacity"),
    ("toolz", "toolz"),
    ("tqdm", "tqdm"),
    ("ujson", "ujson"),
    ("xmltodict", "xmltodict"),
    # --- scientific ---
    ("bokeh", "bokeh"),
    ("bottleneck", "bottleneck"),
    ("dask.dataframe", "dask"),
    ("distributed", "distributed"),
    ("duckdb", "duckdb"),
    ("jupyter_server_proxy", "jupyter-server-proxy"),
    ("matplotlib", "matplotlib"),
    ("numba", "numba"),
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
    # --- visualization ---
    ("altair", "altair"),
    ("datashader", "datashader"),
    ("great_tables", "great-tables"),
    ("holoviews", "holoviews"),
    ("hvplot", "hvplot"),
    ("ipympl", "ipympl"),
    ("itables", "itables"),
    ("jupyter_bokeh", "jupyter-bokeh"),
    ("panel", "panel"),
    ("plotly", "plotly"),
    ("plotnine", "plotnine"),
    ("seaborn", "seaborn"),
    ("vl_convert", "vl-convert-python"),
    # --- dataio ---
    ("adbc_driver_manager", "adbc-driver-manager"),
    ("adbc_driver_postgresql", "adbc-driver-postgresql"),
    ("adbc_driver_sqlite", "adbc-driver-sqlite"),
    ("adlfs", "adlfs"),
    ("connectorx", "connectorx"),
    ("deltalake", "deltalake"),
    ("fastparquet", "fastparquet"),
    ("gcsfs", "gcsfs"),
    ("h5py", "h5py"),
    ("ibis.backends.duckdb", "ibis-framework"),
    ("netCDF4", "netcdf4"),
    ("openpyxl", "openpyxl"),
    ("psycopg", "psycopg"),
    ("psycopg_binary", "psycopg-binary"),
    ("pymysql", "pymysql"),
    ("pyreadstat", "pyreadstat"),
    ("python_calamine", "python-calamine"),
    ("s3fs", "s3fs"),
    ("sqlalchemy", "sqlalchemy"),
    ("tables", "tables"),
    ("xlrd", "xlrd"),
    ("xlsxwriter", "xlsxwriter"),
    ("zarr", "zarr"),
    # --- ml ---
    ("catboost", "catboost"),
    ("imblearn", "imbalanced-learn"),
    ("lightgbm", "lightgbm"),
    ("mapie", "mapie"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("optuna", "optuna"),
    ("sklearn", "scikit-learn"),
    ("shap", "shap"),
    ("skl2onnx", "skl2onnx"),
    ("skops", "skops"),
    ("skrub", "skrub"),
    ("umap", "umap-learn"),
    ("xgboost", "xgboost"),
    # --- deeplearn ---
    ("accelerate", "accelerate"),
    ("einops", "einops"),
    ("keras", "keras"),
    ("lightning", "lightning"),
    ("tensorboard", "tensorboard"),
    ("tensorflow", "tensorflow"),
    ("torchmetrics", "torchmetrics"),
    # --- vision ---
    ("imageio", "imageio"),
    ("kornia", "kornia"),
    ("open_clip", "open-clip-torch"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("skimage", "scikit-image"),
    ("supervision", "supervision"),
    ("timm", "timm"),
    ("ultralytics", "ultralytics"),
    # --- audio ---
    ("audioread", "audioread"),
    ("librosa", "librosa"),
    ("noisereduce", "noisereduce"),
    ("pedalboard", "pedalboard"),
    ("parselmouth", "praat-parselmouth"),
    ("pydub", "pydub"),
    ("pyloudnorm", "pyloudnorm"),
    ("soundfile", "soundfile"),
    # --- geospatial ---
    ("cartopy", "cartopy"),
    ("contextily", "contextily"),
    ("folium", "folium"),
    ("geodatasets", "geodatasets"),
    ("geopandas", "geopandas"),
    ("geoviews", "geoviews"),
    ("h3", "h3"),
    ("lonboard", "lonboard"),
    ("mapclassify", "mapclassify"),
    ("osmnx", "osmnx"),
    ("pyproj", "pyproj"),
    ("rasterio", "rasterio"),
    ("rioxarray", "rioxarray"),
    ("shapely", "shapely"),
    # --- timeseries ---
    ("arch", "arch"),
    ("mlforecast", "mlforecast"),
    ("pmdarima", "pmdarima"),
    ("prophet", "prophet"),
    ("skforecast", "skforecast"),
    ("sktime", "sktime"),
    ("statsforecast", "statsforecast"),
    ("tsfresh", "tsfresh"),
    ("tslearn", "tslearn"),
    # --- optimization ---
    ("cvxpy", "cvxpy"),
    ("ortools.constraint_solver.pywrapcp", "ortools"),
    ("pulp", "pulp"),
    ("pyomo.environ", "pyomo"),
    # --- nlp ---
    ("bertopic", "bertopic"),
    ("datasets", "datasets"),
    ("en_core_web_sm", "en-core-web-sm"),
    ("evaluate", "evaluate"),
    ("keybert", "keybert"),
    ("lingua", "lingua-language-detector"),
    ("nltk", "nltk"),
    ("peft", "peft"),
    ("rapidfuzz", "rapidfuzz"),
    ("sentence_transformers", "sentence-transformers"),
    ("sentencepiece", "sentencepiece"),
    ("spacy", "spacy"),
    ("tiktoken", "tiktoken"),
    ("tokenizers", "tokenizers"),
    ("transformers", "transformers"),
    # --- speech ---
    ("TTS", "coqui-tts"),
    ("faster_whisper", "faster-whisper"),
    ("gtts", "gtts"),
    ("jiwer", "jiwer"),
    ("whisper", "openai-whisper"),
    ("piper", "piper-tts"),
    ("pyannote.audio", "pyannote-audio"),
    ("silero_vad", "silero-vad"),
    ("speechbrain", "speechbrain"),
    ("speech_recognition", "speechrecognition"),
    # --- face ---
    ("deepface", "deepface"),
    ("diffusers", "diffusers"),
    ("dlib", "dlib"),
    ("face_alignment", "face-alignment"),
    ("insightface", "insightface"),
    ("mediapipe", "mediapipe"),
    ("mtcnn", "mtcnn"),
    ("retinaface", "retina-face"),
    ("tf_keras", "tf-keras"),
]


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying FULL target imports")
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
        print("\nAll full imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
