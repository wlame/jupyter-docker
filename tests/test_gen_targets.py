"""
Tests for scripts/gen_targets.py — the matrix → targets generator.

These tests run on the host (no Docker image required) and carry no target
mark, so the in-container example suites (`pytest -m <target>`) skip them.
"""

import ast
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

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
    return {re.split(r'\[|==', dep)[0] for dep in data['project']['dependencies']}


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
        [sys.executable, str(GEN), '--check'], capture_output=True, text=True, check=False
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
        check=False,
    )

    assert result.returncode == 1
    assert 'STALE' in result.stdout


def test_write_mode_restores_drifted_file_and_is_idempotent(repo_copy: Path):
    pyproject = repo_copy / 'targets' / 'base' / 'pyproject.toml'
    original = pyproject.read_text()
    pyproject.write_text(original + '\n# manual edit\n')

    write = subprocess.run(
        [sys.executable, str(GEN), '--root', str(repo_copy)], capture_output=True, text=True, check=False
    )
    check = subprocess.run(
        [sys.executable, str(GEN), '--check', '--root', str(repo_copy)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert write.returncode == 0
    assert pyproject.read_text() == original
    assert check.returncode == 0, check.stdout + check.stderr


PYTHON_MATRIX_TOML = """
[settings]
python = ["3.14", "3.13"]
exclude-newer = "2026-01-01T00:00:00Z"

[targets.base]
parent = ""
description = "root that inherits [settings] python"

[targets.narrow]
parent = "base"
description = "narrows the parent's list"
python = ["3.13"]

[targets.grandchild]
parent = "narrow"
description = "inherits the narrowed list"

[packages]
"""


def run_gen(*args: str, root: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GEN), *args, '--root', str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def python_matrix_root(tmp_path: Path) -> Path:
    """A synthetic repo whose matrix exercises Python-version inheritance."""
    (tmp_path / 'targets').mkdir()
    (tmp_path / 'targets' / 'matrix.toml').write_text(PYTHON_MATRIX_TOML)
    return tmp_path


def test_python_matrix_lists_every_target():
    result = run_gen('--python-matrix')
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) == set(ALL_TARGETS)


def test_python_matrix_inherits_from_settings_and_parents(python_matrix_root: Path):
    result = run_gen('--python-matrix', root=python_matrix_root)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        'base': ['3.14', '3.13'],
        'narrow': ['3.13'],
        'grandchild': ['3.13'],
    }


def test_python_versions_prints_default_first(python_matrix_root: Path):
    result = run_gen('--python-versions', 'base', root=python_matrix_root)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['3.14', '3.13']


def test_python_versions_rejects_unknown_target(python_matrix_root: Path):
    result = run_gen('--python-versions', 'missing', root=python_matrix_root)

    assert result.returncode != 0
    assert 'unknown target' in result.stderr


def pyproject(target: str) -> dict:
    with open(REPO_ROOT / 'targets' / target / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)


PYTHON_MATRIX = json.loads(run_gen('--python-matrix').stdout)


@pytest.mark.parametrize('target', ALL_TARGETS)
def test_requires_python_admits_exactly_the_target_versions(target):
    versions = PYTHON_MATRIX[target]
    spec = SpecifierSet(pyproject(target)['project']['requires-python'])
    minors = sorted(int(v.split('.')[1]) for v in versions)

    assert all(f'{v}.0' in spec for v in versions), (target, str(spec))
    assert f'3.{minors[0] - 1}.0' not in spec, (target, str(spec))
    assert f'3.{minors[-1] + 1}.0' not in spec, (target, str(spec))


@pytest.mark.parametrize('target', ALL_TARGETS)
def test_required_environments_list_each_target_version(target):
    environments = pyproject(target)['tool']['uv']['required-environments']
    listed = [re.search(r"python_version == '([\d.]+)'", env).group(1) for env in environments]

    assert sorted(listed) == sorted(PYTHON_MATRIX[target])
    assert all("sys_platform == 'linux'" in env and "platform_machine == 'x86_64'" in env for env in environments)


@pytest.mark.parametrize('child,parent', CHILD_PARENT)
def test_child_python_versions_are_subset_of_parent(child, parent):
    extra = set(PYTHON_MATRIX[child]) - set(PYTHON_MATRIX[parent])
    assert not extra, f"{child} lists Python {sorted(extra)} that its parent {parent} does not build"


@pytest.mark.parametrize(
    'edit,expected',
    [
        # a child may not add a version its parent lacks
        (('python = ["3.13"]\n\n[targets.grandchild]', 'python = ["3.13"]\n\n[targets.grandchild]\npython = ["3.14"]'), 'grandchild'),
        # every version must appear in [settings]
        (('python = ["3.13"]\n\n[targets.grandchild]', 'python = ["3.12"]\n\n[targets.grandchild]'), '3.12'),
        # [settings] must declare the list
        (('python = ["3.14", "3.13"]\nexclude-newer', 'exclude-newer'), 'settings'),
    ],
    ids=['child-outside-parent', 'version-not-in-settings', 'settings-missing-python'],
)
def test_matrix_with_invalid_python_versions_is_rejected(python_matrix_root: Path, edit, expected):
    matrix = python_matrix_root / 'targets' / 'matrix.toml'
    old, new = edit
    assert old in matrix.read_text()
    matrix.write_text(matrix.read_text().replace(old, new))

    result = run_gen('--python-matrix', root=python_matrix_root)

    assert result.returncode != 0
    assert expected in result.stderr


@pytest.mark.parametrize('target', ALL_TARGETS)
def test_lock_environments_cover_linux_images_and_apple_silicon(target):
    environments = pyproject(target)['tool']['uv']['environments']
    assert "sys_platform == 'linux'" in environments
    assert any("sys_platform == 'darwin'" in env and 'arm64' in env for env in environments)


def test_verify_script_reports_non_import_errors_and_keeps_going(python_matrix_root: Path, tmp_path: Path):
    matrix = python_matrix_root / 'targets' / 'matrix.toml'
    matrix.write_text(matrix.read_text() + """
[packages."boom"]
version = "1.0"
module = "boom"
introduced-by = ["base"]

[packages."stdlib-json"]
version = "1.0"
module = "json"
introduced-by = ["base"]
""")
    modules = tmp_path / 'modules'
    modules.mkdir()
    (modules / 'boom.py').write_text("raise FileNotFoundError('package.json missing')\n")

    generated = run_gen(root=python_matrix_root)
    verify = subprocess.run(
        [sys.executable, str(python_matrix_root / 'targets' / 'base' / 'verify_imports.py')],
        capture_output=True,
        text=True,
        check=False,
        env={'PYTHONPATH': str(modules), 'PATH': ''},
    )

    assert generated.returncode == 0, generated.stderr
    assert verify.returncode == 1
    assert 'boom: FileNotFoundError: package.json missing' in verify.stdout
    assert '1 passed, 1 failed' in verify.stdout


VERIFY_FIRST_MODULES = {spec['module'] for spec in MATRIX['packages'].values() if spec.get('verify-first')}


@pytest.mark.parametrize('target', ALL_TARGETS)
def test_verify_script_imports_torch_family_before_everything_else(target):
    tree = ast.parse((REPO_ROOT / 'targets' / target / 'verify_imports.py').read_text())
    pairs = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'IMPORTS')
    modules = [elt.elts[0].value for elt in pairs.elts]
    flags = [m in VERIFY_FIRST_MODULES for m in modules]

    assert flags == sorted(flags, reverse=True), f"{target}: a non-torch module loads before the torch family"


def test_package_exclude_newer_becomes_a_uv_exception(python_matrix_root: Path):
    matrix = python_matrix_root / 'targets' / 'matrix.toml'
    matrix.write_text(matrix.read_text() + """
[packages."fresh-fix"]
version = "1.0"
module = "fresh_fix"
introduced-by = ["narrow"]
exclude-newer = "2026-02-01T00:00:00Z"
""")

    result = run_gen(root=python_matrix_root)
    narrow = tomllib.loads((python_matrix_root / 'targets' / 'narrow' / 'pyproject.toml').read_text())
    base = tomllib.loads((python_matrix_root / 'targets' / 'base' / 'pyproject.toml').read_text())

    assert result.returncode == 0, result.stderr
    assert narrow['tool']['uv']['exclude-newer-package'] == {'fresh-fix': '2026-02-01T00:00:00Z'}
    assert 'exclude-newer-package' not in base['tool']['uv']


def test_package_extras_render_in_the_requirement(python_matrix_root: Path):
    matrix = python_matrix_root / 'targets' / 'matrix.toml'
    matrix.write_text(matrix.read_text() + """
[packages."frame-lib"]
version = "2.0"
module = "frame_lib"
introduced-by = ["base"]
extras = ["sql", "arrow"]
""")

    result = run_gen(root=python_matrix_root)
    base = tomllib.loads((python_matrix_root / 'targets' / 'base' / 'pyproject.toml').read_text())

    assert result.returncode == 0, result.stderr
    assert 'frame-lib[sql,arrow]==2.0' in base['project']['dependencies']
    assert 'frame-lib' in verified_packages('base', root=python_matrix_root)


@pytest.mark.parametrize('extras', ['"sql"', '[""]', '[1]'], ids=['string', 'empty-name', 'number'])
def test_package_extras_must_be_a_list_of_names(python_matrix_root: Path, extras):
    matrix = python_matrix_root / 'targets' / 'matrix.toml'
    matrix.write_text(matrix.read_text() + f"""
[packages."frame-lib"]
version = "2.0"
module = "frame_lib"
introduced-by = ["base"]
extras = {extras}
""")

    result = run_gen('--python-matrix', root=python_matrix_root)

    assert result.returncode != 0
    assert 'extras must be a list of extra names' in result.stderr


@pytest.mark.parametrize(
    'old,new',
    [
        ('exclude-newer = "2026-01-01T00:00:00Z"', 'exclude-newer = "2026-01-01"'),
        ('[packages]\n', '[packages."fresh-fix"]\nversion = "1.0"\nmodule = "fresh_fix"\nintroduced-by = ["base"]\nexclude-newer = "2026-02-01"\n'),
    ],
    ids=['settings-bare-date', 'package-bare-date'],
)
def test_exclude_newer_must_be_a_utc_timestamp(python_matrix_root: Path, old, new):
    matrix = python_matrix_root / 'targets' / 'matrix.toml'
    assert old in matrix.read_text()
    matrix.write_text(matrix.read_text().replace(old, new))

    result = run_gen('--python-matrix', root=python_matrix_root)

    assert result.returncode != 0
    assert 'not a UTC timestamp' in result.stderr
