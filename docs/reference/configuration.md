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

Global values applied to every target.

| Key | Meaning | Example |
|-----|---------|---------|
| `python` | Python versions targets are built for, newest first. The first is the **default**: the version behind the plain `:<target>` image tag. A target can narrow the list (see below). Each version becomes one `required-environments` entry, and the list sets the generated `requires-python` (`>=3.13,<3.15` for two versions, `==3.13.*` for one). | `["3.14", "3.13"]` |
| `exclude-newer` | Supply-chain guard emitted into `[tool.uv]`. `uv` refuses to resolve any package published after this instant, so a freshly published malicious release can't slip in. Bump it when upgrading. Write a full UTC timestamp: uv reads a bare date in the machine's local timezone, so lockfiles would differ between machines and CI. | `"2026-07-05T00:00:00Z"` |

```toml
[settings]
python = ["3.14", "3.13"]
# Supply-chain guard: never resolve packages published after this instant.
exclude-newer = "2026-10-01T00:00:00Z"
```

## `[targets.<name>]`

One entry per Docker build target. The section name (`base`, `ml`, `full`, …) is
the target name and the published image tag (`ghcr.io/wlame/jupyter-docker:<name>`).

| Key | Meaning | Example |
|-----|---------|---------|
| `parent` | The target this one inherits from. A child gets **all** of its parent's packages plus its own. Use `""` for a root — only `base` and `full` are roots (`full` is `FROM base` and installs the union of every package). | `"scientific"` |
| `description` | Human-readable summary. Becomes the image `LABEL` and the generated pyproject's `description`. | `"Classical machine learning with scikit-learn, XGBoost, and LightGBM"` |
| `python` | Optional. Narrows the Python versions this target (and its descendants) is built for. Every version must appear in `[settings] python`, and a child may list only versions its parent builds, because a child stage is built `FROM` its parent's image. The generator rejects a matrix that breaks either rule. | `["3.13"]` |
| `exclude-dependencies` | Transitive packages to force out of resolution. Rendered into `[tool.uv]` as `override-dependencies` entries carrying the never-true marker `sys_platform == 'never'`, so uv drops them on every platform. Unioned along the lineage (and across all targets for `full`). | `["opencv-python"]` |

```toml
[targets.vision]
parent = "base"
description = "Computer vision and image processing"
# ultralytics requires GUI opencv-python, which double-installs cv2 over our
# pinned opencv-python-headless (same paths, corrupted mix).
exclude-dependencies = ["opencv-python"]
```

An `exclude-dependencies` entry becomes, in the generated `pyproject.toml`:

```toml
[tool.uv]
# ...
override-dependencies = [
    "opencv-python ; sys_platform == 'never'",
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

`opencv-python-headless` runs at OpenCV 5 in `vision`, but DeepFace needs the
Haar cascade files OpenCV 5 no longer ships, so `face` and `full` stay on 4.13:

```toml
[packages."opencv-python-headless"]
version = "5.0.0.93"                    # default pin (vision)
module = "cv2"                          # import name ≠ package name
introduced-by = ["vision", "face"]      # added in both vision and face
# OpenCV 5 removed the bundled haarcascade files that deepface requires;
# targets shipping deepface stay on the last 4.x release.
[packages."opencv-python-headless".overrides]
face = "4.13.0.90"                      # face ships deepface
full = "4.13.0.90"                      # full merges the face stack, same hold
```

## How it's consumed

`scripts/gen_targets.py` reads the matrix and materializes, for each of the 14
targets, a `pyproject.toml` (dependency list, `requires-python`, `exclude-newer`,
`environments`, `required-environments`, `override-dependencies`,
`[tool.uv.sources]`) and a `verify_imports.py` (every declared `module`,
torch-family first).

`environments` limits each lockfile to Linux (the images) and Apple-silicon macOS
(local `uv sync` for development). Without it, uv resolves for every platform, and a
cap a library declares only for Windows, emscripten, or Intel macOS can make an
otherwise valid Linux resolution fail.

`required-environments` lists one `linux` / `x86_64` environment per Python
version the target builds for. With it, `uv lock` fails unless every locked package
can install on each of those Pythons — so a pin with no cp314 wheel is caught when
locking, not halfway through an image build:

```toml
[tool.uv]
required-environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.14'",
    "sys_platform == 'linux' and platform_machine == 'x86_64' and python_version == '3.13'",
]
```

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
| `transformers < 5` in speech/full | `coqui-tts` needs it (`isin_mps_friendly` removed in 5.0). | `transformers` override → `4.57.6` on `speech` and `full`; `nlp` stays on `5.18.0`. |
| `sentence-transformers` held on 4-compatible release | Its 5.3+ requires `transformers` 5, which `full` can't ship. | `sentence-transformers` override → `5.2.0` on `full`. |
| `tokenizers <= 0.23.0` in full | `transformers` 4.57.6 (held for coqui-tts) caps it. | `tokenizers` override → `0.22.2` on `full`; `nlp` ships `0.23.2`. |
| `diffusers` 0.39 in full | `diffusers` ≥ 0.40 needs `huggingface-hub` ≥ 1.23; `transformers` 4.57.6 needs `huggingface-hub` < 1.0. | `diffusers` override → `0.39.0` on `full`; `face` ships `0.40.0`. |
| spaCy 3.8.14 in full | spaCy ≥ 3.8.15 needs `click` ≥ 8.2.1; `gtts` 2.5.4 needs `click` < 8.2. | `spacy` override → `3.8.14` on `full`; `nlp` ships `3.8.16`. |
| `bokeh < 3.10` | `panel` 1.9.4 cap. | `bokeh` pinned to `3.9.2`. |
| `pandas < 3` in timeseries/full | `statsforecast`, `mlforecast`, and `skforecast` cap it. | `pandas` override → `2.3.3` on `timeseries` and `full`; every other target ships `3.0.6`. |
| Cloud filesystems held in full | `datasets` 5.0.1 (nlp) caps `fsspec` at 2026.6.0; `s3fs` pins `fsspec` exactly. | `s3fs` → `2026.6.0` and `gcsfs` → `2026.7.0` on `full`; `dataio` ships the newest. |
| `h5py < 3.15` in full | `tensorflow` 2.21 cap; only `full` merges both stacks. | `h5py` override → `3.14.0` on `full`; `dataio` ships `3.16.0`. |
| `opencv-python-headless` 4.13 in face/full | OpenCV 5 dropped the bundled haarcascade files `deepface` needs. | `opencv-python-headless` override → `4.13.0.90` on `face` and `full`. |
| GUI `opencv-python` / `opencv-contrib-python` excluded on vision/face | They double-install `cv2` over the pinned headless build (same paths, corrupted mix); `mediapipe` pulls the contrib build. | `exclude-dependencies` on `vision` (`opencv-python`) and `face` (both). |
| `torchcodec` + FFmpeg only in audio/speech/full | `torchaudio` ≥ 2.10 delegates load/save to `torchcodec`, which needs FFmpeg shared libs present only in those stages. | `torchcodec` `introduced-by = ["audio", "speech"]` (and `full`). |
| `nvidia-nccl-cu13` kept on ml/timeseries | `xgboost` ≥ 3.4 depends on it (~200 MB, distributed GPU only). Excluding it on `ml` would also remove it from `deeplearn` and `full`, where torch needs it. | No exclusion; the comment on `xgboost` explains the cost. |
| Python 3.13 only for deeplearn/face/full | `tensorflow` 2.21 has no cp314 wheels (and `face` also needs `tf-keras` 2.22). | `python = ["3.13"]` on `deeplearn`, `face`, and `full`. |

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
