---
tags:
  - use-cases
  - jupyterlab
---

# Run JupyterLab

## Scenario

You want an interactive JupyterLab environment with the scientific Python stack
(NumPy, SciPy, Pandas, Statsmodels) without installing anything on your host.
You have a `notebooks/` folder of work and a `data/` folder of inputs, and you
want both to persist on your machine while the runtime lives in a container.

The `scientific` target is the smallest image that carries the numerical stack,
so it is the natural starting point. Every image already bundles the repository's
example notebooks at `/home/jupyter/examples`, so you can run one immediately.

## Prerequisites

- Docker installed and running.
- Two local folders to mount — create them if they don't exist:

```bash
mkdir -p notebooks data
```

## Complete example

Pull and run the prebuilt `scientific` image, mounting your two folders:

```bash
docker run --rm -p 8888:8888 \
  -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
  -v "$(pwd)/data:/home/jupyter/data" \
  ghcr.io/wlame/jupyter-docker:scientific
```

JupyterLab starts in the foreground and prints its login URL — including the
per-start access token — to your terminal:

```
    To access the server, open this file in a browser:
        file:///home/jupyter/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=<random-token>
```

Open the `http://localhost:8888/lab?token=...` line in your browser. In the file
browser you'll see `notebooks/`, `data/`, and the bundled `examples/`. Open
`examples/02_pandas_data_analysis.ipynb` and run its cells to confirm the stack
works end to end.

## Walkthrough

### Step 1: Map the port

`-p 8888:8888` forwards your host's port 8888 to the container's JupyterLab
server (the image `EXPOSE`s 8888 and launches
`jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`). Change the host side to run
elsewhere, e.g. `-p 9000:8888` to reach it at `http://localhost:9000`.

### Step 2: Mount notebooks and data

The two `-v` flags bind-mount your local folders onto the paths JupyterLab opens
by default. Anything you save under `notebooks/` or `data/` lands on your host
and survives the container. The container user is `jupyter` (UID 1000), which
matches a typical first host user, so mounted files keep sane ownership.

### Step 3: Authenticate with the token

JupyterLab requires token authentication. A fresh random token is generated on
every start, so copy the `?token=...` URL from the terminal. Because this run uses
`--rm` in the foreground, the URL is right there in your terminal output.

### Step 4: Run a notebook

`WORKDIR` is `/home/jupyter`, so the file browser shows the mounted `notebooks/`
and `data/` alongside the baked-in `examples/`. Opening and running
`examples/02_pandas_data_analysis.ipynb` exercises Pandas, and its outputs write
under the examples `output/` directory inside the container.

!!! note "Foreground vs. background"
    `--rm` runs in the foreground and removes the container on exit, so the token
    URL prints straight to your terminal. If you instead run detached with `-d`,
    retrieve the URL from the logs: `docker logs <container>`.

## Expected output

- JupyterLab is reachable at `http://localhost:8888/lab?token=...`.
- The file browser lists `notebooks/`, `data/`, and `examples/`.
- `examples/02_pandas_data_analysis.ipynb` runs without import errors.

## Variations

### Run via `just` against a locally built image

The repository's [`just run`](../reference/cli.md) recipe starts the **locally built** `ds-<target>`
image (not the `ghcr.io` one) with the same standard mounts, so build it first:

```bash
just build scientific
just run scientific
```

`just run scientific` expands to `docker run --rm -p 8888:8888 -v
"$(pwd)/notebooks:/home/jupyter/notebooks" -v "$(pwd)/data:/home/jupyter/data"
ds-scientific`. Pass a different host port as a second argument — `just run
scientific 9000` maps `-p 9000:8888`.

### Pin a fixed token

Instead of hunting for the random token each start, set one explicitly:

```bash
docker run --rm -p 8888:8888 \
  -e JUPYTER_TOKEN=your-secret-token \
  -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
  -v "$(pwd)/data:/home/jupyter/data" \
  ghcr.io/wlame/jupyter-docker:scientific
```

!!! warning "Don't expose the port unprotected"
    Do not publish the port beyond localhost without a token and TLS. Put a
    reverse proxy in front for anything non-local.

### Use a different target

`scientific` is just one of 14 targets. Swap the tag for any other — e.g.
`ghcr.io/wlame/jupyter-docker:ml` or `:full`. See
[Pick the right target](pick-a-target.md) to choose.

## Related pages

- [Pick the right target](pick-a-target.md)
- [Image targets](../reference/targets.md)
- [CLI & dev commands](../reference/cli.md)
- [Troubleshooting](../troubleshooting.md)
