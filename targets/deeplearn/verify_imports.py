#!/usr/bin/env python3
"""Verify all deeplearn target imports are working correctly."""

import sys
import os

# Suppress TensorFlow warnings during import verification
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMPORTS = [
    # PyTorch
    ("torch", "torch"),
    ("torchvision", "torchvision"),

    # TensorFlow
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
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
        print("\nAll deeplearn imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
