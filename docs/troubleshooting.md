---
tags:
  - troubleshooting
  - debugging
---

# Troubleshooting

Concrete failure modes for `jupyter-docker`, grouped by where they bite: importing
libraries inside a running image, regenerating and building images, and starting
the container. Each entry lists the symptom, the underlying cause, and the fix.

!!! note "Reproduce import failures fast"
    Every image ships a verification script that imports all of a target's declared
    packages. Run it to reproduce (and isolate) an import error without opening a
    notebook:

    ```bash
    docker run --rm ds-<target> uv run --no-project python /home/jupyter/scripts/verify_<target>.py
    ```

    For the `base` and `full` targets the path is
    `/home/jupyter/scripts/verify_imports.py`.

## Runtime import errors

### Segfault when importing TensorFlow before torch

**Symptom**: the process dies with a segmentation fault (no Python traceback) while
importing TensorFlow, on the `deeplearn`, `face`, or `full` images.

**Cause**: TensorFlow and torch bundle their own native C++ runtimes, and importing
TensorFlow *before* torch in the same process triggers a symbol clash that
segfaults the interpreter.

**Fix**: import torch first. The matrix marks the torch-family packages
[`verify-first = true`](reference/configuration.md), so the generated verify scripts already order them ahead of
TensorFlow, and example 20 (`examples/20_face_analysis.py`) runs face-alignment
(torch) before DeepFace (TensorFlow).

!!! warning "Keep torch ahead of TensorFlow in new code"
    Any new example or notebook that mixes both stacks must import torch (or a
    torch-based library such as face-alignment) before it imports TensorFlow or
    DeepFace. Reversing the order reintroduces the segfault.

---

### `libgomp.so.1: cannot open shared object file`

**Symptom**: importing scikit-learn, XGBoost, or LightGBM fails with this error.

**Cause**: those libraries link the OpenMP runtime, which lives in the `libgomp1`
system package. The base image installs `libgomp1` explicitly because the runtime
images ship no build toolchain that would otherwise pull it in transitively.

**Fix**: keep `libgomp1` in the base stage's `apt-get install` list. If you trim it
to slim the image, these imports break across every target that inherits `ml`.

---

### `libpython3.X.so.1.0: cannot open shared object file`

**Symptom**: importing torchcodec (used by torchaudio ≥ 2.10) fails with this
error on the `audio`, `speech`, or `full` images.

**Cause**: torchcodec's compiled extension links `libpython` directly. That shared
object lives in the `libpython3.X` system package (matching the image's Python), which the base image installs
for exactly this reason.

**Fix**: keep `libpython${PYTHON_VERSION}` in the base stage's `apt-get install` list.

---

### `libavutil.so.*: cannot open shared object file`

**Symptom**: torchaudio load/save or a torchcodec call fails looking for an FFmpeg
shared library such as `libavutil`, `libavcodec`, or `libavformat`.

**Cause**: torchaudio ≥ 2.10 delegates audio load/save to torchcodec, which needs
the FFmpeg shared libraries at runtime. Only the `audio`, `speech`, and `full`
stages install `ffmpeg`.

**Fix**: use one of those targets for torchaudio/torchcodec work. If you add
torchaudio to a different target, add `ffmpeg` to that stage's `apt-get install`
list too.

!!! note "FFmpeg is not in every image"
    FFmpeg is intentionally scoped to `audio`, `speech`, and `full` to keep the
    other images lean — it is not a base dependency.

---

### Corrupted `cv2` after installing GUI OpenCV

**Symptom**: `import cv2` fails or crashes after you `pip install opencv-python`
inside a `vision` or `face` image.

**Cause**: these targets pin `opencv-python-headless`. Installing the GUI
`opencv-python` on top writes to the same `cv2` paths, producing a corrupted mix of
the two builds. The matrix therefore lists `opencv-python` under
`exclude-dependencies` for `vision` and `face`.

**Fix**: do not install `opencv-python` in these images — the headless build
already provides `cv2`. Separately, the `face` and `full` targets pin OpenCV
`4.13.0.90` on purpose: OpenCV 5 dropped the bundled haarcascade files that
DeepFace needs, so staying on the last 4.x release keeps DeepFace working.

## Build and generation errors

### CI `consistency` job fails with `STALE:`

**Symptom**: the CI `consistency` job (or a local `just gen-check`) fails and prints
`STALE:` followed by a generated file path.

**Cause**: a generated `targets/<name>/pyproject.toml` or
`targets/<name>/verify_imports.py` no longer matches `targets/matrix.toml` — usually
because a generated file was hand-edited, or the matrix was changed without
regenerating.

**Fix**: edit the matrix, then regenerate and relock:

```bash
# edit targets/matrix.toml, then:
just gen     # regenerate pyproject.toml + verify_imports.py from the matrix
just lock    # re-resolve every target's uv.lock
```

!!! warning "Never edit generated files by hand"
    `targets/matrix.toml` is the single source of truth. Every
    `targets/<name>/pyproject.toml` and `verify_imports.py` is generated from it by
    [`scripts/gen_targets.py`](reference/cli.md). Edit the matrix and regenerate — changes made directly
    to a generated file are reported as `STALE:` and will not survive the next
    `just gen`.

---

### `uv sync --locked` fails, or `just lock-check` reports an out-of-date lockfile

**Symptom**: an image build fails at `uv sync --locked`, or `just lock-check`
(`uv lock --check`) reports that a `targets/<name>/uv.lock` is out of date.

**Cause**: the target's `pyproject.toml` changed but its committed lockfile was not
re-resolved. The Dockerfile deliberately runs `uv sync --locked`, which refuses to
resolve dependencies at build time, so images stay reproducible.

**Fix**: after editing the matrix and running `just gen`, relock every target:

```bash
just lock
```

---

### Build fails on `--mount=type=cache` (BuildKit not enabled)

**Symptom**: `docker build` errors on the Dockerfile's cache-mount syntax, or the
`uv` cache mounts are silently ignored and every build re-downloads wheels.

**Cause**: the Dockerfile uses BuildKit cache mounts
(`RUN --mount=type=cache,target=/home/jupyter/.cache/uv ...`), which require
BuildKit.

**Fix**: enable BuildKit with `DOCKER_BUILDKIT=1`, or use a `docker buildx`
builder. The `just build` recipe and `build-all.sh` already export
`DOCKER_BUILDKIT=1`, so this only bites raw `docker build` invocations.

## Running the container

### Can't find the JupyterLab URL or login token

**Symptom**: the browser shows a token/login prompt and you don't have the token,
or you don't know which URL to open.

**Cause**: jupyter-server generates a fresh random token on every start — there is
no passwordless mode (the config intentionally never sets `token = ''`).

**Fix**: read the login URL (with token) from the container logs, or pin the token
yourself:

```bash
# random per-start token — copy the URL printed in the logs
docker logs <container>

# or fix the token at start time
docker run --rm -p 8888:8888 -e JUPYTER_TOKEN=my-secret-token \
  ghcr.io/wlame/jupyter-docker:scientific
```

See [Getting started](getting-started.md) for the full run-and-open flow.

---

### Example tests fail or hang downloading model weights

**Symptom**: an example smoke test stalls or fails fetching model weights from the
network.

**Cause**: most weights the examples need are **pre-baked into the image at build
time** by [`scripts/bake_models.sh`](reference/cli.md) (YOLOv8n for `vision`, NLTK corpora and a
sentence-transformers MiniLM for `nlp`, Whisper `tiny` for `speech`, face-alignment
nets for `face`) so the tests run offline. The pytest run sets `HF_HUB_OFFLINE=1`
(in `build-all.sh`, not baked into the image) so a HuggingFace Hub outage can't
flake the MiniLM load.

**Fix**: rely on the baked caches — the marked example tests do not need network.
DeepFace's ~1.5 GB attribute models are deliberately *not* baked; example 20 skips
them when offline, so their absence is expected, not a failure. If you add an
example that downloads a new model, add it to `scripts/bake_models.sh` and keep the
download call as a fallback for runs outside the image.

## Related pages

- [Getting started](getting-started.md) — pull or build an image and open JupyterLab
- [Contributing](contributing.md) — the matrix-driven `just gen` / `just lock` workflow
- [Architecture](concepts/architecture.md) — targets, inheritance, and the builder-split stages
- [Deployment](operations/deployment.md) — running and publishing images
