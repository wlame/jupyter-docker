# AGENTS.md — orientation for coding agents

Multi-target Docker image family for data science: one `Dockerfile` with 14 build
targets (Jupyter Lab on Ubuntu 24.04, Python 3.13 via deadsnakes, uv for packages),
published to `ghcr.io/wlame/jupyter-docker:<target>`.

## The one rule that matters

**`targets/matrix.toml` is the single source of truth for Python dependencies.**
Every `targets/<name>/pyproject.toml` and `targets/<name>/verify_imports.py` is
GENERATED from it by `scripts/gen_targets.py`. Never edit those files by hand —
edit the matrix, then:

```bash
just gen     # regenerate pyprojects + verify scripts
just lock    # re-resolve all 14 committed uv.lock files
just ci      # every fast gate CI enforces (gen/lock/nb checks, lint, tests)
```

CI (`consistency` job) fails if generated files or lockfiles drift from the matrix.

## Core concepts

- **Target**: one Docker build stage = one curated Python environment.
  Inheritance tree (also in `[targets.*]` of the matrix): `base` → everything;
  `scientific` → `ml` → `deeplearn`; `scientific` → `geospatial`/`timeseries`;
  the rest inherit `base` directly. `full` is `FROM base` and installs the union
  of every package and every system library.
- **Package entry** (`[packages."name"]` in the matrix): `version`, `module`
  (import name for verification), `introduced-by` (targets that add it — all
  descendants inherit it), optional `overrides` (per-target version pins with a
  comment explaining the constraint), optional `source-url` (direct wheel URL,
  e.g. the spaCy model).
- **Committed lockfiles**: `targets/<t>/uv.lock`; the Dockerfile runs
  `uv sync --locked` — images are reproducible and never resolve at build time.
- **Verification**: each image gets `scripts/verify_<t>.py` which imports every
  declared package; `build-all.sh --test-only <t>` runs it plus the pytest
  examples marked `<t>` in `tests/test_examples.py`.

## The constraint web (why some versions are held back)

Documented as comments in the matrix; do not "fix" them without checking:

- `numpy < 2.5`: required by numba 0.66 and sktime 1.0 (matrix holds 2.4.x).
- `scikit-learn < 1.8` and `pandas < 3`: required by sktime — `timeseries` and
  `full` carry overrides; other targets run sklearn 1.9 / pandas 3.
- `tokenizers <= 0.23.0`: capped by transformers 5.13 (0.23.0 was never
  released, so 0.22.x is the effective ceiling).
- `h5py < 3.15` in `full` only: tensorflow 2.21's cap; dataio ships 3.16.
- `torchcodec` pairs with torch minor versions and needs FFmpeg shared libs —
  only in `audio`/`speech`/`full` (their stages install ffmpeg).
- `[tool.uv] exclude-newer` (generated into every pyproject): resolution refuses
  packages published in the last ~7 days; bump the date in `[settings]` when
  upgrading.
- Python 3.14 is blocked by spacy and tensorflow (no cp314 wheels yet).

## Build / run / test

```bash
just                    # list all recipes
just build scientific   # docker build one target (BuildKit required: cache mounts)
just test scientific    # verify imports + run marked example tests in the image
just run scientific     # start Jupyter Lab with standard mounts
./build-all.sh          # build + test everything (see --help)
```

There is no way to fully verify image changes without Docker; CI
(`.github/workflows/ci.yml`) is the arbiter. Jobs are tiered: lint+consistency →
base → per-target builds with registry layer-cache (`ghcr.io/.../buildcache:<scope>`),
Trivy scan (report-only), and pushes on main (`:target` + immutable
`:target-<sha>`), plus a weekly scheduled rebuild.

## Examples and tests

- `examples/NN_*.py` are the source of truth; the paired `.ipynb` files are
  jupytext-generated (`just nb`, checked by `just nb-check` via ipynb→py
  round-trip, since notebook regeneration is not byte-stable).
- Examples must write outputs to `OUTPUT_DIR` (derived from `__file__`), never
  hardcoded paths.
- Tests marked `slow` touch the network (model downloads, gTTS). Example 19's
  gTTS section skips loudly on network failure — never add placeholder outputs
  to make a test pass.
- `tests/test_gen_targets.py` (unmarked, host-run) locks in the generator
  invariants: child ⊇ parent, full ⊇ everything, verify-scripts cover all
  declared deps, everything pinned or URL-sourced.

## Gotchas

- **Import torch before TensorFlow in the same process** — the reverse order
  segfaults (C++ symbol clash between their bundled runtimes). The matrix marks
  torch-family packages `verify-first = true` so generated verify scripts order
  them correctly, and example 20 runs face-alignment (torch) before DeepFace
  (TensorFlow). Keep that ordering in any new example mixing both stacks.

- The container user is `jupyter`, UID 1000; `uv sync` runs as that user so the
  venv stays writable (`pip install` works in notebooks). Keep it that way.
- Jupyter auth: a random token is generated per start (`JUPYTER_TOKEN` overrides).
  Do not reintroduce `token = ''` into the config.
- `.dockerignore` uses Docker semantics: bare names only match at the context
  root — nested excludes need `**/` (e.g. `**/.venv`).
- `build-all.sh` relies on `set -e`-safe exit-code capture (`|| pytest_exit=$?`);
  keep that pattern when editing.
- Commit messages: single imperative sentence ending with a period, no
  conventional-commit prefixes, no trailers.
