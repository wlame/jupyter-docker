---
tags:
  - reference
  - cli
---

# Dev & build command reference

Every command you need to develop, generate, build, test, scan, and document
`jupyter-docker`. The [`just`](https://github.com/casey/just) recipes are the dev
entrypoint; they wrap the two shell scripts (`build-all.sh`, `scripts/bake_models.sh`)
and the target generator (`scripts/gen_targets.py`) documented below.

!!! note "Image naming"
    The `justfile` sets `image_prefix := "ds"`, so a locally built target `T` is
    tagged `ds-T` (for example `ds-scientific`). Published images use a different
    name — `ghcr.io/wlame/jupyter-docker:<target>` — see
    [Deployment](../operations/deployment.md).

## `just` recipes

Run `just` with no arguments to list every recipe. Each recipe below shows its
arguments and the underlying command it runs (with `justfile` variables resolved:
`image_prefix = ds`, `jupytext_version = 1.19.4`, `pytest_version = 9.1.1`, `ruff_version = 0.16.10`, `uv_version = 0.12.23`).

| Recipe | Args | Description | Underlying command |
|---|---|---|---|
| `default` | — | List available recipes | `just --list --unsorted` |
| `gen` | — | Regenerate per-target `pyproject.toml` / `verify_imports.py` from the matrix | `python3 scripts/gen_targets.py` |
| `gen-check` | — | Verify generated files match the matrix (CI gate) | `python3 scripts/gen_targets.py --check` |
| `lock` | — | Re-resolve every target's `uv.lock` (run after editing the matrix) | `uvx uv@0.12.23 lock` in each `targets/*/` |
| `lock-check` | — | Verify all lockfiles are up to date (CI gate) | `uvx uv@0.12.23 lock --check` in each `targets/*/` |
| `nb` | — | Regenerate example notebooks from their `.py` sources | `uvx --with jupytext==1.19.4 jupytext --to notebook --update <f>` for each `examples/[0-9]*.py` (`--update` keeps existing cell IDs, so unchanged notebooks stay byte-identical) |
| `nb-check` | — | Verify notebooks are in sync with their `.py` sources (CI gate) | round-trips each `.ipynb` back through `jupytext --to py:light` and `diff`s it against the committed `.py` |
| `lint` | — | Lint Python always; shell/Dockerfile linters run when installed (CI enforces both) | `uvx ruff@0.16.10 check scripts/ tests/` (rules from `ruff.toml`); then `shellcheck build-all.sh scripts/bake_models.sh` and `hadolint --ignore DL3008 Dockerfile` if installed |
| `test-gen` | — | Run the generator test suite on the host (no Docker needed) | `uv run --no-project --python 3.13 --with pytest==9.1.1 python -m pytest tests/test_gen_targets.py -q` |
| `ci` | — | Every fast no-Docker gate that CI tier 0 enforces | runs `gen-check`, `lock-check`, `nb-check`, `lint`, `test-gen` |
| `python-matrix` | — | Print every target's Python versions as JSON (the CI build matrix) | `python3 scripts/gen_targets.py --python-matrix` |
| `build` | `target`, `python` (default: the target's first matrix version) | Build one target as `ds-<target>` and `ds-<target>-py<X.Y>`; refuses a Python the target doesn't support | `DOCKER_BUILDKIT=1 docker build --build-arg PYTHON_VERSION=<python> --target <target> -t ds-<target> -t ds-<target>-py<python> .` |
| `build-all` | `*args` | Build and test every target (see `build-all.sh` for options) | `./build-all.sh <args>` |
| `test` | `target` | Run import verification + example tests against a built image | `./build-all.sh --test-only <target>` |
| `run` | `target`, `port` (default `8888`) | Start a target's Jupyter Lab with the standard volume mounts | `docker run --rm -p <port>:8888 -v "$(pwd)/notebooks:/home/jupyter/notebooks" -v "$(pwd)/data:/home/jupyter/data" ds-<target>` |
| `scan` | `target` | Scan a built image for HIGH/CRITICAL CVEs (requires trivy) | `trivy image --severity HIGH,CRITICAL --ignore-unfixed ds-<target>` |
| `docs-build` | — | Build the docs site (strict — fails on any warning) | `uv run --no-project --with-requirements=requirements-docs.txt mkdocs build --strict` |
| `docs-serve` | — | Serve the docs locally with live reload | `uv run --no-project --with-requirements=requirements-docs.txt mkdocs serve` |

Common invocations:

```bash
just                      # list all recipes
just ci                   # run every fast no-Docker gate
just gen                  # regenerate target files after editing targets/matrix.toml
just build scientific     # build ds-scientific on its default Python
just build scientific 3.13  # build ds-scientific on Python 3.13
just run scientific       # start JupyterLab on port 8888
just run scientific 9000  # start JupyterLab on host port 9000
just test scientific      # import + example tests for a built image
just scan scientific      # scan the image for HIGH/CRITICAL CVEs
```

!!! tip "Match CI locally"
    `just ci` runs exactly the fast gates CI tier 0 enforces (`gen-check`,
    `lock-check`, `nb-check`, `lint`, `test-gen`), so a green `just ci` means those
    checks will pass in CI too.

## `build-all.sh`

Builds and tests Docker targets. The `just build-all` and `just test` recipes both
delegate to this script.

```bash
./build-all.sh [OPTIONS] [TARGETS...]
```

| Flag | Description |
|---|---|
| `--build-only` | Build images without running tests |
| `--test-only` | Run tests on existing images only |
| `--python=X.Y` | Build for this Python version (default: each target's first matrix version); a target that doesn't support it fails its build |
| `--help`, `-h` | Show the usage message and exit |

With no flags the script both builds and tests. `TARGETS...` are positional
target names; the three most common are `base`, `scientific`, and `full`. **If no
target is given, all targets are built and tested** in dependency order.

??? note "All targets (dependency order)"
    ```
    base
    scientific
    visualization
    dataio
    ml
    deeplearn
    vision
    audio
    geospatial
    timeseries
    nlp
    speech
    face
    full
    ```

An unknown option or target name prints the usage message and exits `1`.

!!! warning "`--build-only` and `--test-only` are mutually exclusive"
    Passing both is a hard error — nothing would run, so the script fails loudly
    with `Error: --build-only and --test-only are mutually exclusive.` and exits
    `1` rather than reporting a false success.

Examples (from the script header):

```bash
./build-all.sh              # Build and test all targets
./build-all.sh --build-only # Build without testing
./build-all.sh --test-only  # Test existing images only
./build-all.sh base ml      # Build and test specific targets
./build-all.sh --python=3.13 ml  # Build for a specific Python version
```

### What the test phase runs

For each target, testing runs two steps against the built `ds-<target>` image:

1. **Import verification** — a fast smoke test that imports every package the
   target adds. It runs the generated `verify_<target>.py` (the `full` and `base`
   targets use `verify_imports.py`):

    ```bash
    docker run --rm ds-<target> uv run --no-project python /home/jupyter/scripts/verify_<target>.py
    ```

2. **Example smoke tests** — runs the example scripts under pytest, selecting the
   target's mark (`full` runs every target's mark):

    ```bash
    docker run --rm -e HF_HUB_OFFLINE=1 ds-<target> \
      uv run --no-project python -m pytest /home/jupyter/tests/ -m "<mark>" -v --timeout=300
    ```

    `HF_HUB_OFFLINE=1` forces the pre-baked HuggingFace cache to be used with no
    network call, so a Hub outage cannot flake the run. It is set only for this
    test invocation, not baked into the image.

!!! note "pytest exit code 5 counts as success"
    Exit code `5` means "no tests collected" (a mark that isn't populated yet) and
    is treated as a pass. The `base` target has no example tests, so step 2 is
    skipped entirely for it.

## `scripts/gen_targets.py`

Generates each target's `pyproject.toml` and `verify_imports.py` from
[`targets/matrix.toml`](configuration.md), which is the single source of truth for which package (at
which version) belongs to which target. Wrapped by the `just gen` and `just gen-check`
recipes.

```bash
python3 scripts/gen_targets.py [--check] [--python-matrix] [--python-versions TARGET] [--root PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--check` | off (writes files) | Verify generated files are current instead of writing them |
| `--python-matrix` | off | Print every target's Python versions as JSON and exit (what CI's `plan` job reads) |
| `--python-versions TARGET` | — | Print one target's Python versions, space-separated, default first, and exit |
| `--root PATH` | repository root (parent of `scripts/`) | Repository root containing `targets/matrix.toml` |

`--check` is the drift gate: it writes nothing and exits `1` if any generated file
would change, printing each stale path. This is what CI (and `just gen-check`) run
so the generated files can never drift from the matrix.

Examples (from the script docstring):

```bash
python3 scripts/gen_targets.py          # rewrite generated files in place
python3 scripts/gen_targets.py --check  # exit 1 if any file would change
```

## `scripts/bake_models.sh`

Pre-bakes model weights into an image so the example smoke tests run fully offline.
Each model lands in its library's default cache under `/home/jupyter`, so no runtime
environment variable is needed.

```bash
bake_models.sh <target>
```

The single positional argument selects which weights to download:

| Target | What it downloads |
|---|---|
| `vision` | YOLO26n weights (`yolo26n.pt`) into `~/.cache/ultralytics/`, verified against a pinned SHA-256 |
| `nlp` | NLTK resources (punkt, stopwords, wordnet, taggers, …) into `~/nltk_data`, plus the `all-MiniLM-L6-v2` sentence-transformers model |
| `speech` | Whisper `tiny` model |
| `face` | face-alignment detector + landmark nets (s3fd, 2DFAN) |
| `full` | Everything above — runs the `vision`, `nlp`, `speech`, and `face` bakes in sequence |

Any other target value is accepted and simply prints `no models to bake for target:
<target>` and exits `0`.

!!! note "Invoked at build time, not from the justfile"
    This script has no `just` recipe. The Dockerfile runs it during the build
    (while the network is available) for the model-bearing stages, e.g.
    `RUN bash /home/jupyter/scripts/bake_models.sh vision`. Only the standalone YOLO
    weight file is checksum-verified; the library-managed caches (NLTK, HuggingFace,
    Whisper, torch hub) rely on each library's own integrity checks.

## Related pages

- [Image targets](targets.md) — every target, its libraries, and its tag
- [Configuration](configuration.md) — environment variables and build settings
- [Extend the matrix](../use-cases/extend-the-matrix.md) — add a package or target via `targets/matrix.toml`
- [Contributing](../contributing.md) — dev setup and the PR checklist
