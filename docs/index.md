---
tags:
  - home
  - overview
hide:
  - navigation
  - toc
---

# Data Science Jupyter Notebook Environment — modular, multi-target Docker images for data science with Python 3.13. Build only what you need.

General-purpose data science notebook images bundle every library under one roof, so
you pull gigabytes of tooling you will never import. `jupyter-docker` splits the stack
into 14 composable [**targets**](reference/targets.md) arranged in an inheritance tree — from a lean `base`
image up to an all-in-one `full` environment — so each image carries only the libraries
its work actually needs.

Every target is defined once in [`targets/matrix.toml`](reference/configuration.md), the single source of truth for
dependencies. Per-target `pyproject.toml`, import-verification scripts, and committed
`uv.lock` files are generated from it, giving reproducible `uv sync --locked` builds.
Images run JupyterLab as a non-root user on Ubuntu 24.04 with Python 3.13, and are
published to GitHub Container Registry on every push to `main`.

<br>

[Get started](getting-started.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/wlame/jupyter-docker){ .md-button }

---

## Features

- **14 composable targets** — an inheritance tree from a ~330 MB `base` to a ~5.3 GB `full` image (compressed); build only what you need.
- **Single source of truth** — all dependencies live in `targets/matrix.toml`; per-target files are generated, never hand-edited.
- **Reproducible builds** — a committed `uv.lock` per target; the Dockerfile runs `uv sync --locked` and never resolves at build time.
- **Python 3.13 on Ubuntu 24.04** — via the deadsnakes PPA, with packages managed by a version-pinned uv 0.11.28.
- **JupyterLab out of the box** — token-authenticated, running as a non-root `jupyter` user (UID 1000).
- **Prebuilt images on GHCR** — published to `ghcr.io/wlame/jupyter-docker:<target>` on every push to `main`, rebuilt weekly for OS security patches.
- **Toolchain-free runtime images** — every target installs from prebuilt wheels, so images ship no compilers or headers.

---

## Quick start

Pull and run a target directly from GitHub Container Registry:

```bash
docker run --rm -p 8888:8888 ghcr.io/wlame/jupyter-docker:scientific
```

Replace `scientific` with any target name. JupyterLab starts on port 8888.

!!! note "Finding your login URL"
    A random access token is generated on every container start. Look in the
    container logs for the `http://localhost:8888/lab?token=...` URL and open it in
    your browser. To set a fixed token instead, pass `-e JUPYTER_TOKEN=your-secret-token`.

---

## What's next?

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started.md)** — Pull, run, and mount volumes in minutes
- :material-lightbulb: **[Use cases](use-cases/index.md)** — Pick a target, run JupyterLab, GPU deep learning, extend the matrix
- :material-book-open: **[Image targets](reference/targets.md)** — Every target, its libraries, sizes, and inheritance
- :material-cogs: **[Architecture](concepts/architecture.md)** — How the matrix, generator, and Docker stages fit together

</div>

---

## Related pages

- [About the project](about.md) — design philosophy and how it compares to a monolithic image
- [Configuration reference](reference/configuration.md) — the matrix, environment variables, and build options
- [Deployment](operations/deployment.md) — running the images in production
