---
tags:
  - use-cases
---

# Use cases

`jupyter-docker` ships one `Dockerfile` with 14 curated [build targets](../reference/targets.md), so the
right image depends on the work you're doing. Each page below walks through a
complete, copy-pasteable scenario — from a first JupyterLab session to extending
the dependency matrix — with the exact commands and what to expect.

<div class="grid cards" markdown>

- :material-rocket-launch: **[Run JupyterLab](run-jupyterlab.md)** — Pull a prebuilt image, mount your notebooks and data, and open a tokened JupyterLab session.
- :material-sitemap: **[Pick the right target](pick-a-target.md)** — Map the work you're doing to the smallest target that carries the libraries you need.
- :material-expansion-card-variant: **[GPU deep learning](gpu-deep-learning.md)** — Run the `deeplearn` or `full` image against an NVIDIA GPU with the Container Toolkit.
- :material-table-plus: **[Extend the matrix](extend-the-matrix.md)** — Add or upgrade a library by editing `targets/matrix.toml` and regenerating.

</div>

## Related pages

- [Getting started](../getting-started.md)
- [Image targets](../reference/targets.md)
- [Architecture](../concepts/architecture.md)
