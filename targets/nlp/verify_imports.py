#!/usr/bin/env python3
"""Verify all nlp target imports are working correctly.

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
    # --- nlp ---
    ("torch", "torch"),
    ("en_core_web_sm", "en-core-web-sm"),
    ("matplotlib", "matplotlib"),
    ("nltk", "nltk"),
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers"),
    ("spacy", "spacy"),
    ("tokenizers", "tokenizers"),
    ("transformers", "transformers"),
]


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying NLP target imports")
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
        print("\nAll nlp imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
