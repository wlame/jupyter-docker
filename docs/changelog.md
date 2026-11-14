---
tags:
  - changelog
---

# Changelog

`jupyter-docker` is at **v1.0.0**. There are no tagged releases yet — the
repository carries no git tags, and no root `CHANGELOG.md`.

Until tagged releases begin, changes are tracked through **git history**: every
change lands as a commit on the `main` branch. Each push to `main` also publishes
prebuilt images to GitHub Container Registry, including an immutable
`ghcr.io/wlame/jupyter-docker:<target>-<short-sha>` tag you can pin to a specific
commit for reproducibility and rollback. See
[Deployment & publishing](operations/deployment.md) for how images are tagged and
published.

To review what changed, browse the commit history on GitHub:

- [Commits on `main`](https://github.com/wlame/jupyter-docker/commits/main)

## Notable changes

### November 2026 — runnable notebooks, 12 new examples, dependency refresh

- Every example notebook now runs in JupyterLab as shipped. Before, 32 of 37
  stopped at their first cell (`__file__` does not exist in a kernel), plots never
  appeared inline, and three notebooks had a `try` block split across cells. The
  notebooks now open with a title and explanation, use markdown headings for each
  section, and show figures inline.
- The image tests execute the notebooks instead of the scripts, and a generator
  test compiles every notebook cell, so these failures cannot return unnoticed.
- Each image's `examples/` folder has a `README.md` listing its notebooks, its own
  first.
- Twelve new examples (38–49), one per image, on real or generated data that works
  offline: tabular ML end to end, a Bayesian A/B test, questions over your own
  documents, shift scheduling, one network in PyTorch and Keras, coin counting and
  superpixels, a CO₂ forecast, chord synthesis, review analysis, face
  anonymization, a speech round trip with WER, and messy CSV to Parquet. Examples
  that take your own data read it from `/home/jupyter/data`.
- Dependencies refreshed to a 2026-11-07 cutoff, including Polars 2.1, PyArrow 26,
  DuckDB 2.0, Pillow 13, pandas 3.1, torch 2.15, transformers 5.22, JAX 0.12, and
  the `anthropic` client 2.x. DuckDB 2.0 removed `.pl()` and `.arrow()` from
  `execute()` results; use `con.sql(..., params=...)` for those.
- Lifted holds: numba 0.68 everywhere (PyTensor allows it), bokeh 3.10 with panel
  1.10, and pedalboard's own `exclude-newer` exception. New holds: python-dateutil
  2.9 (aiobotocore caps it below 3) and datasets 5.0.1 in `full` (transformers
  4.57.6 there needs huggingface-hub < 1.0).
- `face` and `full` now also exclude `opencv-contrib-python-headless`, which
  deepface and retina-face started requiring and which would install a second
  `cv2` over the pinned OpenCV.

### October 2026 — four new images and per-image examples

- New images: `optimization` (OR-Tools, CVXPY, Pyomo, PuLP), `jax` (JAX, Optax,
  Flax NNX, Equinox, NumPyro), `probabilistic` (PyMC, nutpie, ArviZ, Bambi, PreliZ;
  built on `jax`), and `genai` (TRL, FAISS, bm25s, diffusers, the `openai` and
  `anthropic` clients, and a baked SmolLM2-135M; built on `nlp`), each with an
  example (34–37). `full` includes all of them.
- Each image now ships only the examples its libraries can run: its own and its
  parents' (`full` still ships all). The matrix records which target owns each
  example.
- `full` now holds protobuf 6.33 (OR-Tools caps it) and numba 0.67 (PyTensor caps
  it), and leaves out highspy, which cannot share a process with OR-Tools.

### October 2026 — data engines, Dask, and YOLO26

- `dataio` adds Ibis (with its DuckDB backend) and ADBC drivers for SQLite and
  PostgreSQL. Example 32 answers one question in pandas, Polars, DuckDB, and Ibis
  over the same Parquet file, checks that the answers agree, and times them.
- `scientific` (and so `ml`, `deeplearn`, `geospatial`, `timeseries`) adds Numba
  and Dask with distributed. The Dask dashboard opens through JupyterLab at
  `/proxy/8787/status` (jupyter-server-proxy). Example 33 explores the Lorenz
  attractor with SymPy, Numba, an FFT, and a Dask ensemble.
- `ml` adds ONNX, onnxruntime, and skl2onnx (previously ONNX came only with
  `deeplearn`); `visualization` adds itables; `nlp` adds SentencePiece; every image
  adds rich and tenacity.
- The `vision` image bakes YOLO26n weights instead of YOLOv8n, and example 13 uses them.
- The matrix accepts `extras` on a package, rendered as `name[extra]==version`.

### October 2026 — images per Python version

- **Breaking for moving-tag users:** the plain `:<target>` tag now points to
  Python **3.14** for every target except `deeplearn`, `face`, and `full`, which
  stay on 3.13 until TensorFlow supports 3.14. To keep 3.13, pull
  `:<target>-py3.13`, or pin an immutable `-<short-sha>` tag from before the change.
- Every target is published per supported Python as `:<target>-py<X.Y>` (plus
  `-<short-sha>`). See [Deployment & publishing](operations/deployment.md).
- All dependencies were refreshed to the 2026-10-01 cutoff; scikit-learn is no longer
  held back in `timeseries` and `full`.
- pedalboard is pinned to 0.9.26, ahead of the global cutoff through a per-package
  `exclude-newer` exception: 0.9.25 crashed with an illegal instruction on some
  x86_64 CPUs (images built before this fix may be affected).
- The TensorFlow images (`deeplearn`, `face`, `full`) no longer install triton: it
  segfaults when TensorFlow is already loaded, which made `import umap` (and any
  TensorFlow-first import followed by torchvision) crash. torch.compile was not
  usable there anyway, since the runtime images ship no C/C++ compiler.
- The committed lockfiles now resolve only for Linux (the images) and Apple-silicon
  macOS. Local `uv sync` of a target on Windows or an Intel Mac is no longer
  supported; use the images there.
- About 75 libraries were added across the targets (JupyterLab Git/LSP/Jupytext
  extensions in every image; polars, DuckDB, xarray; plotnine, datashader; Delta Lake,
  zarr, cloud filesystems; CatBoost, SHAP, MAPIE, UMAP; Lightning, ONNX; timm,
  kornia; pedalboard; rasterio, H3, OSMnx; statsforecast, skforecast; datasets,
  PEFT, KeyBERT; jiwer, silero-vad; InsightFace, MediaPipe), with examples 21–31. See
  [Image targets](reference/targets.md). The new forecasting libraries require
  pandas < 3, so `timeseries` and `full` run pandas 2.3.3.
- Left out on purpose: audiomentations (caps librosa < 0.12, below the shipped 1.0),
  webrtcvad (no Python 3.14 wheel before the cutoff), and jupyterlab-execute-time
  (its Python module cannot be imported).

## Related pages

- [Deployment & publishing](operations/deployment.md) — how images are tagged and published on each push to `main`
- [About the project](about.md) — project status and design philosophy
- [Contributing](contributing.md) — how changes get proposed and land on `main`
