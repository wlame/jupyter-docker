# AGENTS.md

One multi-stage `Dockerfile` builds 14 JupyterLab images ("targets") for data
science on Ubuntu 24.04 with Python 3.14 and 3.13 (deadsnakes) and uv, published as
`ghcr.io/wlame/jupyter-docker:<target>`. `just` is the dev entrypoint; run it
bare to list recipes.

## Dependencies live in the matrix

`targets/matrix.toml` is the single source of truth for Python dependencies.
`scripts/gen_targets.py` generates `targets/<t>/pyproject.toml` and
`targets/<t>/verify_imports.py` from it, and `uv lock` resolves the committed
`targets/<t>/uv.lock` from those. Change dependencies by editing the matrix, then:

```bash
just gen    # regenerate pyprojects + verify scripts
just lock   # re-resolve all 14 lockfiles
just ci     # every fast gate CI enforces
```

The generated files carry a do-not-edit header; CI's `consistency` job fails on
any drift between them, the lockfiles, and the matrix.

- A package lands in each target named in its `introduced-by` and in all their
  descendants; `full` gets every package. Key reference:
  `docs/reference/configuration.md`.
- Held-back pins and per-target `overrides` are deliberate. Each carries a matrix
  comment naming the library that caps it; confirm that cap has lifted upstream
  before raising the pin.
- `[settings] exclude-newer` blocks packages published after its date. A pin
  newer than that date fails to lock until you bump it, and a bump re-resolves
  all 14 lockfiles.
- Keep `exclude-newer` a full UTC timestamp (`…T00:00:00Z`); the generator
  rejects bare dates, which uv reads in the local timezone. A package may carry
  its own later `exclude-newer` for an urgent fix, with a comment saying when to
  drop it (currently pedalboard).
- A package that needs optional dependencies lists them in `extras` (rendered
  as `name[extra]==version`). Point its `module` at the part those extras enable
  (`ibis.backends.duckdb`, `dask.dataframe`), so the verify script catches a
  missing extra.
- Images run `uv sync --locked` and never resolve at build time.
- Every pyproject carries `required-environments` for each of its Python versions
  on linux x86_64, so a pin without a wheel for one of them fails `just lock`
  instead of failing an image build.

## Targets and the Dockerfile

- The tree is `[targets.*] parent` in the matrix, mirrored by each stage's
  `FROM`: `base` → everything; `scientific` → `ml` → `deeplearn`;
  `scientific` → `geospatial` / `timeseries`; the rest sit directly on `base`.
  `full` is `FROM base` and installs the union of every package and every
  system library.
- Python versions are data: `[settings] python` (newest first; the first is the
  default and owns the plain `:<target>` tag), narrowed per target with its own
  `python` key. The interpreter belongs to the whole `FROM` chain, so a child
  lists only versions its parent builds; the generator rejects anything else.
  `deeplearn`, `face`, and `full` are 3.13-only (see their matrix comments).
- The Dockerfile's `ARG PYTHON_VERSION` defaults to 3.14; `just build <t> [python]`,
  `build-all.sh --python=X.Y`, and CI (`just python-matrix`) pick each target's
  versions from the matrix, so prefer those over a raw `docker build`.
- Runtime images ship no compilers. Packages install from wheels; `dlib` (the
  one source build) compiles in `face-builder` / `full-builder`, and only the
  finished `.venv` is copied across (`UV_LINK_MODE=copy` keeps it relocatable).
  A new package that must compile belongs in a builder stage too.
- A shared library a wheel loads at runtime goes in that target's `apt-get`
  list and in `full`'s.
- The container user is `jupyter`, UID 1000, and `uv sync` runs as that user so
  `pip install` keeps working inside notebooks.
- Images expose only Jupyter's port. Web UIs inside the container go through
  jupyter-server-proxy at `/proxy/<port>/`; the base stage sets
  `DASK_DISTRIBUTED__DASHBOARD__LINK` so Dask prints links in that form.
- Jupyter generates a random token per start (`JUPYTER_TOKEN` overrides); keep
  the config free of `token = ''`.
- `.dockerignore` uses Docker semantics: bare names match only at the context
  root, so nested excludes need `**/` (e.g. `**/.venv`).

## Runtime gotchas

- **triton segfaults when TensorFlow is already loaded.** torchvision (through
  `torch._dynamo`) imports triton, so TensorFlow, Keras, DeepFace, or umap-learn
  (whose ParametricUMAP imports TensorFlow) followed by torchvision kills the
  process. The TensorFlow targets (`deeplearn`, `face`, and `full` by union)
  exclude triton, which is unusable there anyway (torch.compile needs a compiler
  the images lack); a new target that ships both stacks needs the same
  exclusion. Verify scripts still import `verify-first` (torch-family) packages
  before everything else as a second guard.
- torchcodec (torchaudio's I/O backend) needs FFmpeg shared libraries and
  `libpython3.X` (the image's Python) at runtime; FFmpeg is installed only in
  `audio`, `speech`, and `full`.

## Examples and tests

- `examples/NN_*.py` are the source; the paired `.ipynb` files are generated by
  `just nb` and checked by `just nb-check`.
- Examples write outputs under `OUTPUT_DIR`, derived from `__file__`.
- Each example belongs to one target: its matrix `examples` list. An image ships
  its own target's and its ancestors' examples (`full` ships all), selected at the
  end of each Dockerfile stage by `scripts/select_examples.sh` from the generated
  `targets/<t>/examples.txt`. A new example needs a matrix entry, or `just gen`
  fails.
- Each example has a test in `tests/test_examples.py` marked with its owning
  target (a generator test checks the match); tests marked `slow` touch the network. A network-dependent section skips
  loudly when offline; a test passes only on real outputs.
- Model weights are pre-baked at build time by `scripts/bake_models.sh` into
  each library's default cache, and the in-image pytest run sets
  `HF_HUB_OFFLINE=1`. An example that needs a new model gets a bake entry, and
  keeps its download call as a fallback for runs outside the image.
- `tests/test_gen_targets.py` (host-run) holds the generator invariants: child ⊇
  parent, `full` ⊇ everything, verify scripts cover every declared package,
  everything pinned or URL-sourced.
- `build-all.sh` captures pytest's exit code with `|| pytest_exit=$?` so it
  stays correct under `set -e`; keep that pattern when editing.

## Verifying a change

- `just ci` — gen-check, lock-check, nb-check, lint, generator tests. No Docker.
- `just build <t> [python]` then `just test <t>` — builds the image (default:
  the target's first Python), then runs its verify script and its marked
  example tests inside it. BuildKit is required.
- `just docs-build` — strict MkDocs build; any warning fails it.
- On an Apple-silicon host, local builds are arm64 and pull different wheels
  (no CUDA stack), so they prove nothing about the published amd64 images. CI's
  per-target build-and-test jobs are the arbiter for image contents and size.

## Docs

`docs/` is the user-facing MkDocs Material site (nav in `mkdocs.yml`, deployed
to GitHub Pages by `.github/workflows/docs.yml`). When a change alters what a
page describes — targets, recipes, the constraint table, sizes, the Python
version — update that page in the same change. `README.md` is the short GitHub
landing page.

## Commits

One imperative sentence ending with a period, with no conventional-commit
prefix and no trailers.

## Where things live

- `Dockerfile` — one stage per target, plus `face-builder` and `full-builder`
- `targets/matrix.toml`, `targets/<t>/` — dependency source and generated files
- `scripts/` — `gen_targets.py` (generator), `bake_models.sh` (model weights),
  `select_examples.sh` (per-image examples)
- `build-all.sh` — build and test driver behind `just build-all` / `just test` and CI
- `.github/workflows/ci.yml`, `.github/actions/build-and-test/` — tiered CI:
  lint + consistency, then per-target builds with registry cache, Trivy
  (report-only), pushes on `main` (`:<t>` and `:<t>-<sha>`), weekly rebuild
- `examples/`, `tests/`, `docs/`
