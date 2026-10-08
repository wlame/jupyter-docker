---
tags:
  - operations
  - deployment
  - docker
---

# Deployment and publishing

How to pull, run, and operate the prebuilt images in production, and how the
project publishes them. Every image ships Ubuntu 24.04 with Python 3.13 and its
target's libraries preinstalled, runs JupyterLab as a non-root user, and exposes
a single port and health check — so deployment is a plain `docker run` plus a few
production guardrails.

## Pulling and running

Prebuilt images live in GitHub Container Registry under
`ghcr.io/wlame/jupyter-docker`. Pull any [target](../reference/targets.md) by its
tag:

```bash
docker pull ghcr.io/wlame/jupyter-docker:scientific
```

Run it with the container's port `8888` published and your local `notebooks/` and
`data/` folders bind-mounted so work persists on the host after the container
exits:

```bash
docker run -d -p 8888:8888 --name jupyter \
  -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
  -v "$(pwd)/data:/home/jupyter/data" \
  ghcr.io/wlame/jupyter-docker:scientific
```

The container command is fixed to JupyterLab:

```
uv run --no-project jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

!!! note "Pulling from GHCR"
    Public packages pull without authentication. If the package is private,
    authenticate first with a GitHub token that has `read:packages`:
    `docker login ghcr.io -u wlame`.

### The login token

Token authentication is always on. No token is baked into the image — on every
start jupyter-server generates a random one and prints the full login URL to the
container logs:

```bash
docker logs jupyter
```

```
[I ServerApp] http://127.0.0.1:8888/lab?token=<random-token>
```

To pin the token instead (for a reverse proxy, a shared bookmark, or automation),
set `JUPYTER_TOKEN`:

```bash
docker run -d -p 8888:8888 --name jupyter \
  -e JUPYTER_TOKEN=my-secret-token \
  ghcr.io/wlame/jupyter-docker:scientific
```

### Bind-mount ownership

The image runs as the non-root user `jupyter` (UID 1000), and the working
directory is `/home/jupyter`. Files the container writes into the bind mounts are
therefore owned by UID 1000 on the host.

!!! note "Match the host UID"
    On a single-user Linux host — where your login account is already UID 1000 —
    mounted files stay owned by you with no extra steps. If your host account has a
    different UID, expect the mounted `notebooks/` and `data/` files to appear
    owned by UID 1000; adjust with `chown` on the host, or run behind a user that
    matches.

## Image tags and rollback

Every push to `main` publishes two tags per target:

| Tag form | Example | Mutability | Use it for |
|---|---|---|---|
| `:<target>` | `ghcr.io/wlame/jupyter-docker:scientific` | **Mutable** — moves to the latest build | Development, always-current pulls |
| `:<target>-<short-sha>` | `ghcr.io/wlame/jupyter-docker:scientific-a1b2c3d` | **Immutable** — one specific commit (7-char short SHA) | Production pinning and rollback |

**In production, pin the immutable SHA tag.** It maps to exactly one build, so a
deploy is reproducible and a rollback is just redeploying the previous
`:<target>-<short-sha>` tag:

```bash
docker pull ghcr.io/wlame/jupyter-docker:scientific-a1b2c3d
```

!!! warning "The `:<target>` tag moves under you"
    Because the plain target tag is reused on every rebuild — including the weekly
    OS-patch rebuild — two `docker pull` calls of `:scientific` days apart can
    return different images. Never pin production to it.

Published images are also **rebuilt weekly** (a scheduled CI run every Monday at
05:30 UTC) so the mutable `:<target>` tags pick up Ubuntu security patches without
a code change. Pinned SHA tags are unaffected by the rebuild — refresh a pinned
deployment by choosing a newer SHA tag when you are ready.

## GPU acceleration

The deep-learning-capable targets (for example `deeplearn`) can use an NVIDIA GPU.
Add `--gpus all`, which requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host:

```bash
docker run -d --gpus all -p 8888:8888 --name jupyter \
  -v "$(pwd)/notebooks:/home/jupyter/notebooks" \
  -v "$(pwd)/data:/home/jupyter/data" \
  ghcr.io/wlame/jupyter-docker:deeplearn
```

Without `--gpus all` the same image runs CPU-only. The GPU path is PyTorch: its
wheels bundle the CUDA 13.0 runtime, so the host needs an R580-or-newer NVIDIA
driver and nothing else; TensorFlow in these images runs on the CPU. See
[GPU deep learning](../use-cases/gpu-deep-learning.md) for the full walkthrough.

## Health checks

The image defines a Docker `HEALTHCHECK` that curls JupyterLab's REST API — the
single monitoring surface the container exposes:

```
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8888/api || exit 1
```

The probe waits `60s` after start before failures count, then polls every `30s`
with a `5s` timeout and marks the container `unhealthy` only after `3` consecutive
failures. Read the status with `docker ps` (the `STATUS` column shows `(healthy)`
/ `(unhealthy)`) or directly:

```bash
docker inspect --format '{{.State.Health.Status}}' jupyter
```

That prints `starting`, `healthy`, or `unhealthy`. Orchestrators (Compose,
Kubernetes, Swarm) can gate readiness and restarts on the same signal.

## Security scanning

Scan a locally built image for HIGH/CRITICAL CVEs with the [`just scan`](../reference/cli.md) recipe,
which wraps [Trivy](https://trivy.dev/):

```bash
just scan scientific
```

That recipe runs exactly:

```bash
trivy image --severity HIGH,CRITICAL --ignore-unfixed ds-scientific
```

`--ignore-unfixed` drops advisories that have no upstream fix yet, so the report
focuses on vulnerabilities you can actually remediate.

!!! note "CI scans are report-only"
    The publish pipeline runs the same Trivy scan (`CRITICAL,HIGH`,
    `ignore-unfixed`, with a 30-minute timeout for the multi-gigabyte images) on
    every target, but with `exit-code: 0` — it surfaces findings in the job log
    without blocking the build or the push. Treat the scan as visibility, not a
    gate, until a baseline is triaged.

## How images are published (CI)

Publishing is fully automated by GitHub Actions (`.github/workflows/ci.yml` and
`.github/actions/build-and-test/action.yml`). On every push to `main`, each target
is:

1. **Built** for its `--target` stage with registry layer-caching.
2. **Tested** by running its import verification and example suite
   (`./build-all.sh --test-only <target>`).
3. **Scanned** with Trivy (report-only, as above).
4. **Pushed** to GHCR as both `:<target>` and `:<target>-<short-sha>`, with a short
   retry to ride out occasional registry blob races.

The jobs are **tiered to mirror the image inheritance tree** — a fast no-Docker
gate tier (lint, generation/lockfile/notebook consistency) runs first; then `base`
builds; then the targets that build `FROM base` run in parallel; then those
`FROM scientific`; then the heavyweight images. A child target's job depends on
its parent's, so caches flow down the tree.

Pushing happens **only on `main`** — pull-request runs build, test, and scan but
never push (the push step is gated on
`github.event_name != 'pull_request' && github.ref == 'refs/heads/main'`). A
scheduled weekly run rebuilds and republishes so the mutable tags absorb OS
patches.

## Production checklist

!!! warning "Never expose an unauthenticated JupyterLab to the public internet"
    JupyterLab grants full code execution and filesystem access to whoever reaches
    it. Exposing port `8888` to the internet without a token (and ideally without
    TLS and an auth proxy in front) hands anonymous users a root-capable Python
    shell inside your container. Keep it bound to localhost or a private network,
    and always require authentication.

Before running an image in production:

- [ ] **Pin the immutable SHA tag** (`:<target>-<short-sha>`), never the moving
      `:<target>` tag.
- [ ] **Require authentication** — set a strong `JUPYTER_TOKEN`, or front the
      container with a reverse proxy that enforces its own auth and TLS.
- [ ] **Do not publish port `8888` to the public internet.** Bind it to localhost
      (`-p 127.0.0.1:8888:8888`) or a private network and reach it through the
      proxy.
- [ ] **Mount data read-only where possible** — append `:ro` to mounts the
      notebooks only read, e.g. `-v "$(pwd)/data:/home/jupyter/data:ro"`.
- [ ] **Set resource limits** so a runaway notebook can't starve the host, e.g.
      `--memory=8g --cpus=4` (and `--gpus` only where GPU work is intended).
- [ ] **Watch the health check** — wire `docker inspect` / orchestrator health
      probes to alerting so an `unhealthy` container is noticed.
- [ ] **Track CVEs** — run `just scan <target>` (or read the CI Trivy output) and
      refresh to a newer SHA tag when the weekly rebuild ships patches.

## Related pages

- [Getting started](../getting-started.md) — pull-and-run and local-build quickstart
- [Image targets](../reference/targets.md) — every target, its libraries, and its tag
- [CLI and dev commands](../reference/cli.md) — the `just run`, `just scan`, and build recipes
- [GPU deep learning](../use-cases/gpu-deep-learning.md) — running the `deeplearn` image on a GPU
- [Troubleshooting](../troubleshooting.md) — common runtime and permission issues
