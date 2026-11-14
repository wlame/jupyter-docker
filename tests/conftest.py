"""
Pytest configuration for example smoke tests.

Each test executes an example notebook (.ipynb) top to bottom in a Jupyter kernel,
the way a user runs it in JupyterLab, and verifies:
  1. Every cell runs without an error
  2. Expected output files were created (for examples that produce them)
"""

import subprocess
import sys
import tempfile
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
    Execute an example notebook and assert it passes.

    Removes ``expected_outputs`` before running so their existence afterward
    proves the notebook actively created them in this run.

    Parameters
    ----------
    name:
        Filename of the example source, e.g. ``'01_numpy_scipy_basics.py'``; the
        paired ``.ipynb`` is what runs.
    expected_outputs:
        List of filenames (relative to ``output/``) that must be created.
    timeout:
        Maximum seconds to allow the whole notebook to run.
    """
    notebook = EXAMPLES_DIR / Path(name).with_suffix('.ipynb')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Delete stale copies of expected outputs so we can verify fresh creation.
    for fname in (expected_outputs or []):
        target = OUTPUT_DIR / fname
        if target.exists():
            target.unlink()

    with tempfile.TemporaryDirectory() as executed_dir:
        # The kernel starts in the notebook's directory, as in JupyterLab.
        command = [
            sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=-1', f'--output-dir={executed_dir}', str(notebook),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=str(EXAMPLES_DIR),
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"{notebook.name} timed out after {timeout}s")

    if result.returncode != 0:
        pytest.fail(
            f"{notebook.name} failed (exit code {result.returncode})\n"
            f"--- nbconvert output (last 4000 chars) ---\n{result.stderr[-4000:]}"
        )

    missing = [f for f in (expected_outputs or []) if not (OUTPUT_DIR / f).exists()]
    if missing:
        pytest.fail(
            f"{notebook.name} ran successfully but did not create expected output files:\n"
            + "".join(f"  - {f}\n" for f in missing)
        )
