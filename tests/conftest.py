"""
Pytest configuration for example smoke tests.

Each test runs a .py example script as a subprocess and verifies:
  1. Exit code is 0
  2. Expected output files were created (for examples that produce them)
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest

# Inside Docker the examples live at /home/jupyter/examples/.
# When running from the project root on the host, compute the path relative to this file.
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
EXAMPLES_DIR = _REPO_ROOT / 'examples'
OUTPUT_DIR = EXAMPLES_DIR / 'output'


def run_example(
    name: str,
    expected_outputs: list[str] | None = None,
    timeout: int = 180,
) -> None:
    """
    Run an example script and assert it passes.

    Removes ``expected_outputs`` before running so their existence afterward
    proves the script actively created them in this run.

    Parameters
    ----------
    name:
        Filename of the example, e.g. ``'01_numpy_scipy_basics.py'``.
    expected_outputs:
        List of filenames (relative to ``output/``) that must be created.
    timeout:
        Maximum seconds to allow the script to run.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete stale copies of expected outputs so we can verify fresh creation.
    for fname in (expected_outputs or []):
        target = OUTPUT_DIR / fname
        if target.exists():
            target.unlink()

    env = os.environ.copy()
    # Force non-interactive matplotlib backend for all examples.
    env['MPLBACKEND'] = 'Agg'

    try:
        result = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / name)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(EXAMPLES_DIR),
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{name} timed out after {timeout}s")

    if result.returncode != 0:
        pytest.fail(
            f"{name} exited with code {result.returncode}\n"
            f"--- stdout (last 3000 chars) ---\n{result.stdout[-3000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )

    missing = [f for f in (expected_outputs or []) if not (OUTPUT_DIR / f).exists()]
    if missing:
        pytest.fail(
            f"{name} ran successfully but did not create expected output files:\n"
            + "".join(f"  - {f}\n" for f in missing)
        )
