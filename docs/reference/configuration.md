---
tags:
  - configuration
  - reference
---

# Configuration

`targets/matrix.toml` is the **single source of truth** for every Python
dependency in every image target. You never edit a per-target `pyproject.toml` or
`verify_imports.py` by hand — those are generated from the matrix by
[`scripts/gen_targets.py`](cli.md). Change a version, add a package, or hold a dependency
back in exactly one place: this file.

The matrix has three sections:

- `[settings]` — global values stamped into every generated `pyproject.toml`.
- `[targets.<name>]` — one Docker build target: its parent, description, and any
  transitive packages to force-exclude.
- `[packages."<name>"]` — one dependency: its pin, import name, which targets add
  it, and any per-target version holds.

!!! note "Generated files are never edited directly"
    Every `targets/<name>/pyproject.toml` and `targets/<name>/verify_imports.py`
    carries a `GENERATED FILE — do not edit by hand` header. Edit the matrix, then
    run `just gen`. CI fails if the generated files or lockfiles drift from the
    matrix.

## `[settings]`

Global values applied to every target. Both are emitted verbatim into each
generated `pyproject.toml`.

| Key | Meaning | Example |
|-----|---------|---------|
| `requires-python` | Minimum Python version; emitted into each pyproject's `[project]` as `requires-python`. | `">=3.13"` |
| `exclude-newer` | Supply-chain guard emitted into `[tool.uv]`. `uv` refuses to resolve any package published after this instant, so a freshly published malicious release can't slip in. Bump it when upgrading. Write a full UTC timestamp: uv reads a bare date in the machine's local timezone, so lockfiles would differ between machines and CI. | `"2026-07-05T00:00:00Z"` |

```toml
[settings]
requires-python = ">=3.13"
# Supply-chain guard: never resolve packages published after this instant.
exclude-newer = "2026-07-05T00:00:00Z"
```

## `[targets.<name>]`

One entry per Docker build target. The section name (`base`, `ml`, `full`, …) is
the target name and the published image tag (`ghcr.io/wlame/jupyter-docker:<name>`).

| Key | Meaning | Example |
|-----|---------|---------|
| `parent` | The target this one inherits from. A child gets **all** of its parent's packages plus its own. Use `""` for a root — only `base` and `full` are roots (`full` is `FROM base` and installs the union of every package). | `"scientific"` |
| `description` | Human-readable summary. Becomes the image `LABEL` and the generated pyproject's `description`. | `"Classical machine learning with scikit-learn, XGBoost, and LightGBM"` |
| `exclude-dependencies` | Transitive packages to force out of resolution. Rendered into `[tool.uv]` as `override-dependencies` entries carrying the never-true marker `sys_platform == 'never'`, so uv drops them on every platform. Unioned along the lineage (and across all targets for `full`). | `["nvidia-nccl-cu12"]` |

```toml
[targets.ml]
parent = "scientific"
description = "Classical machine learning with scikit-learn, XGBoost, and LightGBM"
# xgboost 3.3 pulls nvidia-nccl-cu12 (distributed-GPU only); it collides with
# torch's nvidia-nccl-cu13 wheels in deeplearn and is dead weight on CPU images.
exclude-dependencies = ["nvidia-nccl-cu12"]
```

An `exclude-dependencies` entry becomes, in the generated `pyproject.toml`:

```toml
[tool.uv]
# ...
override-dependencies = [
    "nvidia-nccl-cu12 ; sys_platform == 'never'",
]
```

## `[packages."<name>"]`

One entry per dependency. The section name is the exact package name uv installs
(`beautifulsoup4`, `scikit-learn`, `opencv-python-headless`, …).

| Key | Meaning | Example |
|-----|---------|---------|
| `version` | Pinned version. Emitted as `name==version` in every target that gets the package. Optional **only** when `source-url` is set. | `"1.9.0"` |
| `module` | Import name used by the generated `verify_imports.py` — often differs from the package name (`scikit-learn` → `sklearn`, `opencv-python-headless` → `cv2`, `pyannote-audio` → `pyannote.audio`). | `"sklearn"` |
| `introduced-by` | List of targets that add the package. Every **descendant** of a listed target inherits it. `full` always gets every package regardless. | `["ml", "timeseries"]` |
| `overrides` | Sub-table `[packages."<name>".overrides]` pinning a different version for specific targets. Remove an override once the constraint that forced it is gone. | `[packages."pandas".overrides]` |
| `source-url` | Direct wheel URL rendered into `[tool.uv.sources]` instead of a version pin. Used for the spaCy model `en-core-web-sm`. | `"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"` |
| `verify-first` | `true` for the torch family. Sorts the package ahead of the rest within its group in the verify scripts, so torch native modules import before TensorFlow in one process. | `true` |

!!! warning "Import torch before TensorFlow"
    Importing TensorFlow before torch in the same process segfaults (a C++ symbol
    clash between their bundled runtimes). `verify-first = true` on the
    torch-family packages is what keeps the generated verify scripts in the safe
    order — do not remove it.

### A simple package

`module` differs from the package name here — `beautifulsoup4` is imported as
`bs4`, which is why every package must declare it explicitly:

```toml
[packages."beautifulsoup4"]
version = "4.15.0"         # pinned; emitted as beautifulsoup4==4.15.0
module = "bs4"             # import name checked by verify_imports.py
introduced-by = ["base"]   # added in base, so every target inherits it
```

### A package with overrides

`scikit-learn` runs at 1.9.0 on the `ml` stack, but the `sktime` constraint in
`timeseries` and `full` forces those two targets back to 1.7.2:

```toml
[packages."scikit-learn"]
version = "1.9.0"                       # default pin (ml and its descendants)
module = "sklearn"                      # import name ≠ package name
introduced-by = ["ml", "timeseries"]    # added in both ml and timeseries
# sktime constrains scikit-learn below 1.8; affected targets hold 1.7.2.
[packages."scikit-learn".overrides]
timeseries = "1.7.2"                    # timeseries holds the older release
full = "1.7.2"                          # full merges the sktime stack, same hold
```

## How it's consumed

`scripts/gen_targets.py` reads the matrix and materializes, for each of the 14
targets, a `pyproject.toml` (dependency list, `exclude-newer`,
`override-dependencies`, `[tool.uv.sources]`) and a `verify_imports.py` (every
declared `module`, torch-family first).

```bash
just gen                              # or: python3 scripts/gen_targets.py
python3 scripts/gen_targets.py --check  # or: just gen-check  (CI drift gate)
```

`just gen` rewrites the generated files in place. The `--check` form writes
nothing and exits non-zero if any generated file would differ — this is the CI
gate that guarantees the committed `pyproject.toml`/`verify_imports.py` files
always match the matrix.

The loader validates the matrix before generating and raises `SystemExit` listing
every problem it finds:

- a target whose `parent` isn't a defined target (`unknown parent`),
- a package missing its `module` key (`missing module`),
- a target name in `introduced-by` or `overrides` that isn't defined
  (`unknown target`).

After editing the matrix, re-resolve the lockfiles too (`just lock`), since the
Dockerfile builds with `uv sync --locked`.

## The constraint web

Several versions are deliberately held back so that stacks which share a process
can coexist. These are **not** stale pins to be bumped on sight — each exists
because a downstream library caps a dependency.

| Constraint | Reason | Where in the matrix |
|-----------|--------|---------------------|
| `numpy < 2.5` | `numba` 0.66 and `sktime` 1.0 both require it. | `numpy` pinned to `2.4.6`. |
| `scikit-learn < 1.8` | `sktime` cap. | `scikit-learn` override → `1.7.2` on `timeseries` and `full`. |
| `pandas < 3` | `sktime` cap. | `pandas` override → `2.3.3` on `timeseries` and `full`. |
| `tokenizers <= 0.23.0` | `transformers` 5.13 cap (0.23.0 was never released, so 0.22.x is the effective ceiling). | `tokenizers` pinned to `0.22.2`. |
| `transformers < 5` in speech/full | `coqui-tts` needs it (`isin_mps_friendly` removed in 5.0). | `transformers` override → `4.57.6` on `speech` and `full`; `nlp` stays on `5.13.0`. |
| `sentence-transformers` held on 4-compatible release | Its 5.3+ requires `transformers` 5, which `full` can't ship. | `sentence-transformers` override → `5.2.0` on `full`. |
| `h5py < 3.15` in full | `tensorflow` 2.21 cap; only `full` merges both stacks. | `h5py` override → `3.14.0` on `full`; `dataio` ships `3.16.0`. |
| `opencv-python-headless` 4.13 in face/full | OpenCV 5 dropped the bundled haarcascade files `deepface` needs. | `opencv-python-headless` override → `4.13.0.90` on `face` and `full`. |
| GUI `opencv-python` excluded on vision/face | It double-installs `cv2` over the pinned headless build (same paths, corrupted mix). | `exclude-dependencies = ["opencv-python"]` on `vision` and `face`. |
| `torchcodec` + FFmpeg only in audio/speech/full | `torchaudio` ≥ 2.10 delegates load/save to `torchcodec`, which needs FFmpeg shared libs present only in those stages. | `torchcodec` `introduced-by = ["audio", "speech"]` (and `full`). |
| `nvidia-nccl-cu12` excluded on ml/timeseries | `xgboost` 3.3 pulls it (distributed-GPU only); dead weight on CPU and it clashes with torch's `cu13` wheels. | `exclude-dependencies = ["nvidia-nccl-cu12"]` on `ml` and `timeseries`. |
| Python 3.14 blocked | `tensorflow` 2.21 (in `deeplearn`, `face`, `full`) has no cp314 wheels, and every stage shares the `base` interpreter. | `requires-python = ">=3.13"` in `[settings]`; Python 3.13 in the Dockerfile `base` stage. |

!!! warning "These holds are deliberate — verify before 'fixing'"
    Every pin and exclusion above exists because a specific library caps a
    dependency; the matrix documents each with an inline comment. Before raising
    one, confirm the capping library has actually loosened its requirement.
    Removing a hold blindly reintroduces a resolution conflict or a runtime
    segfault. Bumping the whole matrix forward also means bumping `exclude-newer`
    in `[settings]`.

## Related pages

- [Image targets](targets.md) — every target, its libraries, and its tag
- [Architecture](../concepts/architecture.md) — how the targets form an inheritance tree
- [CLI & recipes](cli.md) — `just gen`, `just lock`, and the build/test workflow
- [Contributing](../contributing.md) — editing the matrix and regenerating files
