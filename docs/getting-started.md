---
tags:
  - installation
  - quickstart
---

# Getting started

Get from zero to a running JupyterLab in under 10 minutes — either by pulling a
prebuilt image from GitHub Container Registry or by building one target locally.

## Prerequisites

- **Docker** with BuildKit enabled — required for every path. BuildKit is the
  default in current Docker Engine; the [`just build`](reference/cli.md) recipe also sets
  `DOCKER_BUILDKIT=1` explicitly.
- **[`just`](https://github.com/casey/just)** and **[`uv`](https://docs.astral.sh/uv/)** — optional, only for the local-build path
  and the dev workflow. Pulling a prebuilt image needs neither.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — only if you want GPU acceleration (for example with
  the `deeplearn` target and `--gpus all`). Not needed for CPU-only use.

!!! note "Everything runs inside the container"
    The images ship Ubuntu 24.04 with Python 3.13 and all libraries preinstalled,
    so the only thing your host needs is Docker. There is no Python, `pip`, or
    virtualenv to set up on the host for the pull-and-run path.

## Installation

Pick a path. The `scientific` target (NumPy, SciPy, Pandas, Statsmodels) is used
throughout — swap in any target name from the [image targets](reference/targets.md) reference.

=== "Pull from GHCR"

    Prebuilt images are published to GitHub Container Registry on every push to
    `main`. Pull and run one directly:

    ```bash
    docker run --rm -p 8888:8888 ghcr.io/wlame/jupyter-docker:scientific
    ```

    Mount your local `notebooks/` and `data/` folders so your work persists on the
    host after the container exits:

    ```bash
    docker run --rm -p 8888:8888 \
      -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
      -v "$(pwd)/data:/home/jupyter/data" \
      ghcr.io/wlame/jupyter-docker:scientific
    ```

    To pin the login token instead of using the random per-start one, pass
    `JUPYTER_TOKEN`:

    ```bash
    docker run --rm -p 8888:8888 \
      -e JUPYTER_TOKEN=my-secret-token \
      ghcr.io/wlame/jupyter-docker:scientific
    ```

=== "Build locally"

    Build a single target image (tagged `ds-<target>`) with `just`:

    ```bash
    just build scientific
    ```

    That recipe is exactly:

    ```bash
    DOCKER_BUILDKIT=1 docker build --target scientific -t ds-scientific .
    ```

    Then start it with the standard `notebooks/` and `data/` mounts on port 8888:

    ```bash
    just run scientific
    ```

    To use a different host port, pass it as the second argument (here the host's
    9000 maps to the container's 8888):

    ```bash
    just run scientific 9000
    ```

## Expected output

The container starts JupyterLab bound to `0.0.0.0:8888`. Because no token is baked
into the image, jupyter-server prints a fresh random login URL to the container
logs on every start. When you run in the foreground (as above), it appears
directly in your terminal:

```
[I ServerApp] Jupyter Server is running at:
[I ServerApp] http://127.0.0.1:8888/lab?token=<random-token>
```

Open that full `http://127.0.0.1:8888/lab?token=...` URL in your browser to reach
JupyterLab. The container also defines a `HEALTHCHECK` that polls
`http://localhost:8888/api`, so `docker ps` reports the container as `healthy`
once the server is up.

!!! note "Authentication is always on"
    Token authentication is **not** disabled. If you pass `-e JUPYTER_TOKEN=...`,
    log in with that value; otherwise use the random token from the URL. If you ran
    the container detached (`-d`), retrieve the URL with `docker logs <container>`.

## Related pages

- [Run JupyterLab](use-cases/run-jupyterlab.md) — the full walkthrough for daily use, mounts, and ports
- [Image targets](reference/targets.md) — every target, its libraries, and its tag
- [Pick the right target](use-cases/pick-a-target.md) — choose the smallest image that fits your work
