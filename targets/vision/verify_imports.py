#!/usr/bin/env python3
"""Verify all vision target imports are working correctly."""

import sys

IMPORTS = [
    # Image processing
    ("PIL", "pillow"),
    ("cv2", "opencv-python-headless"),
    ("skimage", "scikit-image"),
    ("imageio", "imageio"),

    # Object detection
    ("ultralytics", "ultralytics"),
]


def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying VISION target imports")
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
        print("\nAll vision imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
