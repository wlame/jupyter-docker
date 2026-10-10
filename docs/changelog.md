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
