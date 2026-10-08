---
tags:
  - use-cases
  - targets
---

# Pick the right target

## Scenario

You know what you want to do — analyze data, train a model, process audio — but
not which of the 14 images to pull. Picking well matters: targets inherit from
one another, and bigger targets mean bigger images (from ~330 MB for `base` to
~5.3 GB for `full`, compressed). The goal is the **smallest** target that carries
every library your work needs.

## Prerequisites

- None beyond Docker. This page is a decision guide, not a runtime step.

## Complete example

Map the work you're doing to a target, then pull it:

| Work you're doing | Target | Pull command |
|---|---|---|
| Data analysis with Pandas | `scientific` | `docker pull ghcr.io/wlame/jupyter-docker:scientific` |
| Creating charts / dashboards | `visualization` | `docker pull ghcr.io/wlame/jupyter-docker:visualization` |
| Reading/writing Parquet, HDF5, Excel, SQL | `dataio` | `docker pull ghcr.io/wlame/jupyter-docker:dataio` |
| Classical machine learning models | `ml` | `docker pull ghcr.io/wlame/jupyter-docker:ml` |
| Deep learning / neural networks | `deeplearn` | `docker pull ghcr.io/wlame/jupyter-docker:deeplearn` |
| Image processing | `vision` | `docker pull ghcr.io/wlame/jupyter-docker:vision` |
| Audio processing | `audio` | `docker pull ghcr.io/wlame/jupyter-docker:audio` |
| Geographic data / maps / GIS | `geospatial` | `docker pull ghcr.io/wlame/jupyter-docker:geospatial` |
| Time series forecasting | `timeseries` | `docker pull ghcr.io/wlame/jupyter-docker:timeseries` |
| Text / NLP work | `nlp` | `docker pull ghcr.io/wlame/jupyter-docker:nlp` |
| Speech recognition / synthesis | `speech` | `docker pull ghcr.io/wlame/jupyter-docker:speech` |
| Face detection / recognition | `face` | `docker pull ghcr.io/wlame/jupyter-docker:face` |
| Need everything | `full` | `docker pull ghcr.io/wlame/jupyter-docker:full` |

Then run it as in [Run JupyterLab](run-jupyterlab.md), swapping the tag for your
chosen target.

## Walkthrough

### Step 1: Match the work, not the library

Start from the task ("forecast a time series", "detect faces"), find its row
above, and pull that target. Every image also carries the `base` utilities —
JSON/HTTP/date helpers, IPython, JupyterLab — so a specialized target is a
superset of general tooling.

### Step 2: Understand inheritance

Targets form a tree, and a child **includes every library its parent carries**:

```
base
├── scientific
│   ├── ml
│   │   └── deeplearn
│   ├── geospatial
│   └── timeseries
├── visualization
├── dataio
├── vision
├── audio
├── nlp
├── speech
└── face

full (standalone — includes all)
```

So `ml` already has NumPy/SciPy/Pandas (from `scientific`), and `deeplearn`
already has scikit-learn/XGBoost (from `ml`). If your work spans a chain — say,
classical ML *and* neural nets — pick the deepest target (`deeplearn`), not two
separate images. `full` is the exception: it is `FROM base` and installs the
union of every package, standalone rather than inheriting a chain.

### Step 3: Weigh the size cost

Deeper and broader targets are larger. Approximate **compressed download** sizes
(from GHCR; the on-disk uncompressed size is larger):

| Target | Approx. compressed size | Inherits from |
|---|---|---|
| `base` | ~330 MB | Ubuntu 24.04 |
| `scientific` | ~530 MB | base |
| `visualization` | ~470 MB | base |
| `dataio` | ~465 MB | base |
| `ml` | ~655 MB | scientific |
| `deeplearn` | ~3.9 GB | ml |
| `vision` | ~4.7 GB | base |
| `audio` | ~3.4 GB | base |
| `geospatial` | ~750 MB | scientific |
| `timeseries` | ~810 MB | scientific |
| `nlp` | ~3.4 GB | base |
| `speech` | ~3.7 GB | base |
| `face` | ~4.1 GB | base |
| `full` | ~5.3 GB | standalone |

See [Image targets](../reference/targets.md) for the full catalog.

!!! tip "Reach for `full` only when you truly need everything"
    `full` (~5.3 GB) is a convenience for exploratory work that crosses many
    domains. For a focused task, a specialized target pulls and starts far faster.
    `full` also holds a few libraries back to co-exist (e.g. it pins an older
    pandas and transformers so time-series and TTS stacks resolve together).

## Expected output

- A single target name that carries every library your task needs.
- The `docker pull` command for it — then follow [Run JupyterLab](run-jupyterlab.md).

## Variations

### Two unrelated domains at once

If your domains don't share a parent chain (e.g. `audio` and `geospatial`), you
have two options: pull `full` (everything in one image) or run two containers,
one per target, and move data between them through mounted folders.

### Inspect what a target actually contains

The [Image targets](../reference/targets.md) reference lists every target's
libraries. Exact versions live in [`targets/matrix.toml`](../reference/configuration.md)
and in each target's committed `targets/<name>/uv.lock`.

## Related pages

- [Image targets](../reference/targets.md)
- [Run JupyterLab](run-jupyterlab.md)
- [GPU deep learning](gpu-deep-learning.md)
- [Architecture](../concepts/architecture.md)
