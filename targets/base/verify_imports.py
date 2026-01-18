#!/usr/bin/env python3
"""Verify all base target imports are working correctly."""

import sys

IMPORTS = [
    # Interactive Python
    ("IPython", "ipython"),
    ("jupyter", "jupyter"),
    ("jupyterlab", "jupyterlab"),

    # JSON libraries
    ("orjson", "orjson"),
    ("ujson", "ujson"),
    ("simplejson", "simplejson"),

    # XML/HTML
    ("lxml", "lxml"),
    ("xmltodict", "xmltodict"),
    ("bs4", "beautifulsoup4"),

    # YAML
    ("yaml", "pyyaml"),

    # HTTP
    ("requests", "requests"),
    ("httpx", "httpx"),
    ("aiohttp", "aiohttp"),

    # Validation
    ("pydantic", "pydantic"),

    # Utilities
    ("tqdm", "tqdm"),
    ("loguru", "loguru"),
    ("dotenv", "python-dotenv"),
    ("joblib", "joblib"),
    ("toolz", "toolz"),
    ("more_itertools", "more-itertools"),

    # Date/time
    ("dateutil", "python-dateutil"),
    ("pytz", "pytz"),
    ("pendulum", "pendulum"),
]


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying BASE target imports")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for module_name, package_name in IMPORTS:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
            passed += 1
        except ImportError as e:
            print(f"  ✗ {package_name}: {e}")
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
        print("\nAll base imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
