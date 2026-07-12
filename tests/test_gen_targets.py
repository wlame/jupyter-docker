"""
Tests for scripts/gen_targets.py — the matrix → targets generator.

These tests run on the host (no Docker image required) and carry no target
mark, so the in-container example suites (`pytest -m <target>`) skip them.
"""

import ast
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
GEN = REPO_ROOT / 'scripts' / 'gen_targets.py'
MATRIX_PATH = REPO_ROOT / 'targets' / 'matrix.toml'

# Inside the Docker images only tests/ and examples/ are shipped — the matrix
# and generator stay in the repo. Skip at module level so in-container pytest
# collection (which imports this file even for deselected tests) succeeds.
if not MATRIX_PATH.exists() or not GEN.exists():
    pytest.skip(
        "generator sources not present (running inside a target image)",
        allow_module_level=True,
    )


def load_matrix() -> dict:
    with open(MATRIX_PATH, 'rb') as f:
        return tomllib.load(f)


def dependency_names(target: str, root: Path = REPO_ROOT) -> set[str]:
    """Package names declared in a target's generated pyproject.toml."""
    with open(root / 'targets' / target / 'pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
    return {re.split(r'==', dep)[0] for dep in data['project']['dependencies']}


def verified_packages(target: str, root: Path = REPO_ROOT) -> set[str]:
    """Package names covered by a target's generated verify_imports.py."""
    tree = ast.parse((root / 'targets' / target / 'verify_imports.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'IMPORTS':
            return {elt.elts[1].value for elt in node.value.elts}
    raise AssertionError(f"no IMPORTS list found for target {target}")


MATRIX = load_matrix()
ALL_TARGETS = list(MATRIX['targets'])
CHILD_PARENT = [
    (name, spec['parent']) for name, spec in MATRIX['targets'].items() if spec['parent']
]


def test_check_mode_passes_on_current_repo():
    result = subprocess.run(
        [sys.executable, str(GEN), '--check'], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('child,parent', CHILD_PARENT)
def test_child_dependency_set_is_superset_of_parent(child, parent):
    missing = dependency_names(parent) - dependency_names(child)
    assert not missing, f"{child} is missing parent ({parent}) packages: {sorted(missing)}"


def test_full_target_contains_every_package():
    everything = set(MATRIX['packages'])
    missing = everything - dependency_names('full')
    assert not missing, f"full is missing: {sorted(missing)}"


@pytest.mark.parametrize('target', ALL_TARGETS)
def test_verify_imports_covers_every_declared_dependency(target):
    deps = dependency_names(target)
    verified = verified_packages(target)
    assert deps == verified, (
        f"{target}: unverified={sorted(deps - verified)}, phantom={sorted(verified - deps)}"
    )


@pytest.mark.parametrize('target', ALL_TARGETS)
def test_every_package_version_is_exact_or_url_sourced(target):
    with open(REPO_ROOT / 'targets' / target / 'pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
    sources = data.get('tool', {}).get('uv', {}).get('sources', {})
    loose = [
        dep for dep in data['project']['dependencies']
        if '==' not in dep and dep not in sources
    ]
    assert not loose, f"{target}: unpinned dependencies without a source: {loose}"


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Minimal copy of the repo that gen_targets.py can operate on."""
    shutil.copytree(
        REPO_ROOT / 'targets',
        tmp_path / 'targets',
        ignore=shutil.ignore_patterns('.venv', '__pycache__'),
    )
    return tmp_path


def test_check_mode_detects_manual_edit(repo_copy: Path):
    pyproject = repo_copy / 'targets' / 'base' / 'pyproject.toml'
    pyproject.write_text(pyproject.read_text() + '\n# manual edit\n')

    result = subprocess.run(
        [sys.executable, str(GEN), '--check', '--root', str(repo_copy)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert 'STALE' in result.stdout


def test_write_mode_restores_drifted_file_and_is_idempotent(repo_copy: Path):
    pyproject = repo_copy / 'targets' / 'base' / 'pyproject.toml'
    original = pyproject.read_text()
    pyproject.write_text(original + '\n# manual edit\n')

    write = subprocess.run(
        [sys.executable, str(GEN), '--root', str(repo_copy)], capture_output=True, text=True
    )
    check = subprocess.run(
        [sys.executable, str(GEN), '--check', '--root', str(repo_copy)],
        capture_output=True,
        text=True,
    )

    assert write.returncode == 0
    assert pyproject.read_text() == original
    assert check.returncode == 0, check.stdout + check.stderr
