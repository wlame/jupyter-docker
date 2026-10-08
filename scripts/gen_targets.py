#!/usr/bin/env python3
"""Generate per-target pyproject.toml and verify_imports.py from targets/matrix.toml.

The matrix is the single source of truth for which package (at which version)
belongs to which Docker target. This script materializes it into the files the
Dockerfile actually consumes, so the generated files can never drift from the
matrix — CI runs it with --check to enforce that.

Usage:
    python3 scripts/gen_targets.py          # rewrite generated files in place
    python3 scripts/gen_targets.py --check  # exit 1 if any file would change
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

GENERATED_HEADER_TOML = """\
# -----------------------------------------------------------------------------
# GENERATED FILE — do not edit by hand.
# Source of truth: targets/matrix.toml
# Regenerate with:  python3 scripts/gen_targets.py
# -----------------------------------------------------------------------------
"""

GENERATED_HEADER_PY = '''\
#!/usr/bin/env python3
"""Verify all {target} target imports are working correctly.

GENERATED FILE — do not edit by hand.
Source of truth: targets/matrix.toml (regenerate: python3 scripts/gen_targets.py).
"""
'''

VERIFY_RUNNER = '''

def verify_imports():
    """Verify all imports and report results."""
    print("=" * 60)
    print("Verifying {target_upper} target imports")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for module_name, package_name in IMPORTS:
        try:
            __import__(module_name)
            print(f"  \\u2713 {{package_name}}")
            passed += 1
        except ImportError as e:
            print(f"  \\u2717 {{package_name}}: {{e}}")
            failed += 1
            errors.append((package_name, str(e)))

    print("=" * 60)
    print(f"Results: {{passed}} passed, {{failed}} failed")
    print("=" * 60)

    if failed > 0:
        print("\\nFailed imports:")
        for pkg, err in errors:
            print(f"  - {{pkg}}: {{err}}")
        sys.exit(1)
    else:
        print("\\nAll {target} imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    verify_imports()
'''


def load_matrix(root: Path) -> dict:
    """Load and validate targets/matrix.toml."""
    with open(root / 'targets' / 'matrix.toml', 'rb') as f:
        matrix = tomllib.load(f)

    targets = matrix['targets']
    packages = matrix['packages']
    errors = []
    for name, target in targets.items():
        parent = target['parent']
        if parent and parent not in targets:
            errors.append(f"target {name}: unknown parent {parent!r}")
    for pkg, spec in packages.items():
        if 'module' not in spec:
            errors.append(f"package {pkg}: missing module")
        for t in spec.get('introduced-by', []):
            if t not in targets:
                errors.append(f"package {pkg}: unknown target {t!r} in introduced-by")
        for t in spec.get('overrides', {}):
            if t not in targets:
                errors.append(f"package {pkg}: unknown target {t!r} in overrides")
    if errors:
        raise SystemExit("matrix.toml is invalid:\n  " + "\n  ".join(errors))
    return matrix


def lineage(target: str, targets: dict) -> list[str]:
    """Ancestor chain from the root down to (and including) the target itself."""
    chain = []
    cursor = target
    while cursor:
        chain.append(cursor)
        cursor = targets[cursor]['parent']
    return list(reversed(chain))


def target_order(targets: dict) -> list[str]:
    """All targets in matrix declaration order (which is tree order)."""
    return list(targets)


def materialize(target: str, matrix: dict) -> dict[str, dict]:
    """Resolve the full package set for a target.

    Returns {package: {'version': str|None, 'module': str, 'group': str}} where
    group is the lineage target that introduces the package (used for section
    comments in the generated files). The 'full' target gets every package,
    grouped by its first introducer in tree order.
    """
    targets = matrix['targets']
    packages = matrix['packages']
    chain = target_order(targets) if target == 'full' else lineage(target, targets)
    chain = [t for t in chain if t != 'full']

    result: dict[str, dict] = {}
    for pkg, spec in packages.items():
        introducers = spec.get('introduced-by', [])
        group = next((t for t in chain if t in introducers), None)
        if group is None and target != 'full':
            continue
        version = spec.get('overrides', {}).get(target, spec.get('version'))
        result[pkg] = {
            'version': version,
            'module': spec['module'],
            'group': group or introducers[0],
            'source-url': spec.get('source-url'),
            'verify-first': spec.get('verify-first', False),
        }
    return result


def excluded_dependencies(target: str, matrix: dict) -> list[str]:
    """Transitive packages force-excluded for a target (union along the lineage).

    The 'full' target unions every target's exclusions, mirroring how it unions
    every target's packages.
    """
    targets = matrix['targets']
    chain = target_order(targets) if target == 'full' else lineage(target, targets)
    names: list[str] = []
    for t in chain + ([target] if target == 'full' else []):
        for pkg in targets[t].get('exclude-dependencies', []):
            if pkg not in names:
                names.append(pkg)
    return names


def grouped(mat: dict[str, dict], chain: list[str]) -> list[tuple[str, list[str]]]:
    """Package names grouped by introducing target, in lineage order.

    Within a group, verify-first packages sort ahead of the rest: torch native
    modules must be imported before TensorFlow in the same process.
    """
    def order(pkg: str) -> tuple[int, str]:
        return (0 if mat[pkg].get('verify-first') else 1, pkg)

    return [
        (t, sorted((p for p, spec in mat.items() if spec['group'] == t), key=order))
        for t in chain
        if any(spec['group'] == t for spec in mat.values())
    ]


def render_pyproject(target: str, matrix: dict) -> str:
    """Render the pyproject.toml content for one target."""
    targets = matrix['targets']
    settings = matrix['settings']
    mat = materialize(target, matrix)
    chain = target_order(targets) if target == 'full' else lineage(target, targets)
    chain = [t for t in chain if t != 'full']

    lines = [GENERATED_HEADER_TOML]
    lines += [
        '[project]',
        f'name = "datascience-{target}"',
        'version = "1.0.0"',
        f'description = "{targets[target]["description"]}"',
        f'requires-python = "{settings["requires-python"]}"',
        '',
        'dependencies = [',
    ]
    for group, pkgs in grouped(mat, chain):
        lines.append(f'    # --- {group} ---')
        for pkg in pkgs:
            version = mat[pkg]['version']
            lines.append(f'    "{pkg}=={version}",' if version else f'    "{pkg}",')
    lines += [']', '']

    lines += [
        '[tool.uv]',
        '# Supply-chain guard: never resolve packages published after this date',
        f'exclude-newer = "{settings["exclude-newer"]}"',
    ]
    excluded = excluded_dependencies(target, matrix)
    if excluded:
        lines += [
            '# Transitive dependencies excluded via a never-true marker',
            '# (see exclude-dependencies comments in targets/matrix.toml)',
            'override-dependencies = [',
        ]
        lines += [f'    "{pkg} ; sys_platform == \'never\'",' for pkg in excluded]
        lines.append(']')
    lines.append('')
    sourced = {p: s['source-url'] for p, s in mat.items() if s['source-url']}
    if sourced:
        lines.append('[tool.uv.sources]')
        for pkg, url in sorted(sourced.items()):
            lines.append(f'{pkg} = {{ url = "{url}" }}')
        lines.append('')

    lines += [
        '[build-system]',
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        '',
        '[tool.hatch.build.targets.wheel]',
        'packages = []',
    ]
    return '\n'.join(lines) + '\n'


def render_verify(target: str, matrix: dict) -> str:
    """Render the verify_imports.py content for one target."""
    targets = matrix['targets']
    mat = materialize(target, matrix)
    chain = target_order(targets) if target == 'full' else lineage(target, targets)
    chain = [t for t in chain if t != 'full']

    lines = [GENERATED_HEADER_PY.format(target=target), 'import sys', '', 'IMPORTS = [']
    for group, pkgs in grouped(mat, chain):
        lines.append(f'    # --- {group} ---')
        for pkg in pkgs:
            lines.append(f'    ("{mat[pkg]["module"]}", "{pkg}"),')
    lines.append(']')
    body = '\n'.join(lines)
    runner = VERIFY_RUNNER.format(target=target, target_upper=target.upper())
    return body + '\n' + runner


def generate(root: Path, check: bool) -> int:
    """Write (or verify) all generated files; return the number of stale files."""
    matrix = load_matrix(root)
    stale = 0
    for target in matrix['targets']:
        for filename, content in (
            ('pyproject.toml', render_pyproject(target, matrix)),
            ('verify_imports.py', render_verify(target, matrix)),
        ):
            path = root / 'targets' / target / filename
            current = path.read_text() if path.exists() else None
            if current == content:
                continue
            stale += 1
            if check:
                print(f"STALE: {path.relative_to(root)}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                print(f"wrote {path.relative_to(root)}")
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check',
        action='store_true',
        help='verify generated files are current instead of writing them',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help='repository root containing targets/matrix.toml',
    )
    args = parser.parse_args()

    stale = generate(args.root, check=args.check)
    if args.check and stale:
        print(f"\n{stale} generated file(s) out of date — run: python3 scripts/gen_targets.py")
        sys.exit(1)
    if args.check:
        print("all generated files match targets/matrix.toml")
    elif stale == 0:
        print("all generated files already up to date")


if __name__ == '__main__':
    main()
