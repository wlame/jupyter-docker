# =============================================================================
# Dev entrypoint for jupyter-docker. Run `just` to list recipes.
# =============================================================================

image_prefix := "ds"
jupytext_version := "1.19.4"
pytest_version := "9.1.1"

set shell := ["bash", "-uc"]

# List available recipes
default:
    @just --list --unsorted

# ── Generation (targets/matrix.toml is the source of truth) ─────────────────

# Regenerate per-target pyproject.toml / verify_imports.py from the matrix
gen:
    python3 scripts/gen_targets.py

# Verify generated files match the matrix (CI gate)
gen-check:
    python3 scripts/gen_targets.py --check

# Re-resolve every target's uv.lock (run after editing the matrix)
lock:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in targets/*/; do
        echo "── ${d}"
        (cd "${d}" && uv lock --python 3.13)
    done

# Verify all lockfiles are up to date (CI gate)
lock-check:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in targets/*/; do
        echo "── ${d}"
        (cd "${d}" && uv lock --check)
    done

# Regenerate example notebooks from their .py sources
nb:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/[0-9]*.py; do
        uvx --with jupytext=={{jupytext_version}} jupytext --to notebook "${f}"
    done

# Verify notebooks are in sync with their .py sources (CI gate).
# Compares the ipynb→py round-trip against the committed .py, because
# regenerating notebooks directly is not byte-stable (random cell ids).
nb-check:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/[0-9]*.py; do
        nb="${f%.py}.ipynb"
        if [ ! -f "${nb}" ]; then echo "MISSING notebook: ${nb}"; exit 1; fi
        if ! uvx --with jupytext=={{jupytext_version}} jupytext \
                --to py:light --output - "${nb}" 2>/dev/null | diff -q - "${f}" >/dev/null; then
            echo "OUT OF SYNC: ${nb} vs ${f} — run: just nb"
            exit 1
        fi
        echo "in sync: ${nb}"
    done

# ── Quality gates ────────────────────────────────────────────────────────────

# Lint Python always; shell/Dockerfile linters run when installed (CI enforces both)
lint:
    uvx ruff check scripts/ tests/
    @if command -v shellcheck >/dev/null; then \
        shellcheck build-all.sh scripts/bake_models.sh; \
    else \
        echo "shellcheck not installed — skipped (CI enforces it)"; \
    fi
    @if command -v hadolint >/dev/null; then \
        hadolint --ignore DL3008 Dockerfile; \
    else \
        echo "hadolint not installed — skipped (CI enforces it)"; \
    fi

# Run the generator test suite on the host (no Docker needed)
test-gen:
    uv run --no-project --python 3.13 --with pytest=={{pytest_version}} \
        python -m pytest tests/test_gen_targets.py -q

# Every fast no-Docker gate that CI tier 0 enforces
ci: gen-check lock-check nb-check lint test-gen

# ── Docker ───────────────────────────────────────────────────────────────────

# Build one target image as ds-<target>
build target:
    DOCKER_BUILDKIT=1 docker build --target {{target}} -t {{image_prefix}}-{{target}} .

# Build and test every target (see build-all.sh for options)
build-all *args:
    ./build-all.sh {{args}}

# Run import verification + example tests against a built image
test target:
    ./build-all.sh --test-only {{target}}

# Start a target's Jupyter Lab with the standard volume mounts
run target port="8888":
    docker run --rm -p {{port}}:8888 \
        -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
        -v "$(pwd)/data:/home/jupyter/data" \
        {{image_prefix}}-{{target}}

# Scan a built image for HIGH/CRITICAL CVEs (requires trivy)
scan target:
    trivy image --severity HIGH,CRITICAL --ignore-unfixed {{image_prefix}}-{{target}}
