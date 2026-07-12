#!/usr/bin/env python3
"""Verify all full target imports are working correctly.

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
    ("jupyterlab", "jupyterlab"),
    ("loguru", "loguru"),
    ("lxml", "lxml"),
    ("more_itertools", "more-itertools"),
    ("orjson", "orjson"),
    ("pendulum", "pendulum"),
    ("pip", "pip"),
    ("pydantic", "pydantic"),
    ("pytest", "pytest"),
    ("pytest_timeout", "pytest-timeout"),
    ("dateutil", "python-dateutil"),
    ("dotenv", "python-dotenv"),
    ("pytz", "pytz"),
    ("yaml", "pyyaml"),
    ("requests", "requests"),
    ("simplejson", "simplejson"),
    ("toolz", "toolz"),
    ("tqdm", "tqdm"),
    ("ujson", "ujson"),
    ("xmltodict", "xmltodict"),
    # --- scientific ---
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("sympy", "sympy"),
    # --- visualization ---
    ("altair", "altair"),
    ("bokeh", "bokeh"),
    ("holoviews", "holoviews"),
    ("hvplot", "hvplot"),
    ("panel", "panel"),
    ("plotly", "plotly"),
    ("seaborn", "seaborn"),
    # --- dataio ---
    ("fastparquet", "fastparquet"),
    ("h5py", "h5py"),
    ("openpyxl", "openpyxl"),
    ("pyarrow", "pyarrow"),
    ("sqlalchemy", "sqlalchemy"),
    ("tables", "tables"),
    ("xlrd", "xlrd"),
    # --- ml ---
    ("imblearn", "imbalanced-learn"),
    ("lightgbm", "lightgbm"),
    ("optuna", "optuna"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    # --- deeplearn ---
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("torchvision", "torchvision"),
    ("keras", "keras"),
    ("tensorflow", "tensorflow"),
    # --- vision ---
    ("imageio", "imageio"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("skimage", "scikit-image"),
    ("ultralytics", "ultralytics"),
    # --- audio ---
    ("torchcodec", "torchcodec"),
    ("audioread", "audioread"),
    ("librosa", "librosa"),
    ("numba", "numba"),
    ("pydub", "pydub"),
    ("soundfile", "soundfile"),
    # --- geospatial ---
    ("cartopy", "cartopy"),
    ("folium", "folium"),
    ("geopandas", "geopandas"),
    ("geoviews", "geoviews"),
    ("pyproj", "pyproj"),
    ("shapely", "shapely"),
    # --- timeseries ---
    ("pmdarima", "pmdarima"),
    ("prophet", "prophet"),
    ("sktime", "sktime"),
    ("tsfresh", "tsfresh"),
    # --- nlp ---
    ("en_core_web_sm", "en-core-web-sm"),
    ("nltk", "nltk"),
    ("sentence_transformers", "sentence-transformers"),
    ("spacy", "spacy"),
    ("tokenizers", "tokenizers"),
    ("transformers", "transformers"),
    # --- speech ---
    ("TTS", "coqui-tts"),
    ("faster_whisper", "faster-whisper"),
    ("gtts", "gtts"),
    ("whisper", "openai-whisper"),
    ("piper", "piper-tts"),
    ("pyannote.audio", "pyannote-audio"),
    ("speechbrain", "speechbrain"),
    ("speech_recognition", "speechrecognition"),
    # --- face ---
    ("deepface", "deepface"),
    ("diffusers", "diffusers"),
    ("dlib", "dlib"),
    ("face_alignment", "face-alignment"),
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
        except ImportError as e:
            print(f"  \u2717 {package_name}: {e}")
            failed += 1
            errors.append((package_name, str(e)))

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
