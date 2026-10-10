---
tags:
  - use-cases
  - dependencies
  - contributing
---

# Extend the matrix

## Scenario

You need a library that isn't in a target yet, or you want to bump one to a newer
version. In this project you never edit a target's `pyproject.toml` or
`verify_imports.py` by hand — they are **generated**. The single source of truth
for every Python dependency is [`targets/matrix.toml`](../reference/configuration.md). Change it, regenerate, and
re-resolve the lockfiles.

## Prerequisites

- The repository cloned locally.
- [`uv`](https://docs.astral.sh/uv/) and `just` installed (the generator and lock
  steps run on the host, no Docker needed).
- Docker, only for the `build` / `test` verification at the end.

## Complete example

Add a small utility — `humanize` — to the `base` target (so every image inherits
it). Add a package entry under the `# ── introduced in: base ──` section of
`targets/matrix.toml`:

```toml
[packages."humanize"]
version = "4.14.0"   # pin the exact version you want
module = "humanize"  # the import name, used by the generated verify script
introduced-by = ["base"]
```

Then run the generate → lock → build → test loop:

```bash
just gen                 # regenerate every pyproject.toml + verify_imports.py
just lock                # re-resolve all committed uv.lock files
just build base          # build the affected image (BuildKit required)
just test base           # verify imports + run marked example tests in the image
```

## Walkthrough

### Step 1: Add the package entry

Each `[packages."<name>"]` block declares three required fields:

- **`version`** — the exact version to pin. Resolution is reproducible, so pin a
  concrete version rather than a range.
- **`module`** — the import name used by the generated verify script (it differs
  from the package name often enough to matter, e.g. `beautifulsoup4` →
  `bs4`, `pyyaml` → `yaml`).
- **`introduced-by`** — the list of targets that add the package. It lands in
  those targets **and all their descendants**. Adding to `["base"]` reaches every
  image; adding to `["scientific"]` reaches `scientific`, `ml`, `deeplearn`,
  `geospatial`, `timeseries`, `optimization`, `jax`, and `probabilistic`.

Two optional fields cover harder cases: `overrides` pins a different version for
specific targets (with a comment explaining the constraint), and `source-url`
installs from a direct wheel URL instead of PyPI.

### Step 2: Regenerate the per-target files

```bash
just gen
```

This rewrites every `targets/<name>/pyproject.toml` and
`targets/<name>/verify_imports.py` from the matrix. `just gen-check` is the
read-only CI gate that fails if these drift from the matrix.

### Step 3: Re-resolve the lockfiles

```bash
just lock
```

This re-runs `uv lock` (with the pinned uv 0.12.23) in every target directory so each committed
`uv.lock` reflects the new dependency. Images build from these lockfiles with
`uv sync --locked` and never resolve at build time.

### Step 4: Build and test the affected image

```bash
just build base
just test base
```

`just test` runs the generated verify script (which imports every declared
package) plus the example tests marked for that target. Repeat for any other
target that changed — if you added to `base`, higher targets pick it up too.

!!! tip "Run the fast host-only gates before you push"
    `just ci` runs every no-Docker gate CI enforces — `gen-check`, `lock-check`,
    notebook sync, lint, and the generator tests — in one shot. It catches drift
    without needing to build an image.

!!! warning "The `exclude-newer` supply-chain guard"
    `[settings].exclude-newer` in the matrix refuses to resolve any package
    published within roughly the last week. If you pin a version newer than that
    date, `just lock` fails to find it — bump the `exclude-newer` date in
    `[settings]` when upgrading to a fresh release.

## Expected output

- `just gen` and `just lock` complete without error, updating the generated
  `pyproject.toml`, `verify_imports.py`, and `uv.lock` files.
- `just test base` reports the new package's import verifying successfully inside
  the built image.

## Variations

### Upgrade an existing library

To bump a version rather than add a package, edit the `version` field of its
existing entry — for example, changing `tqdm`'s `version` string — then run the
same `just gen` → `just lock` → `just build <target>` → `just test <target>` loop.

### Add to a specific target and its descendants

Set `introduced-by` to the narrowest target that needs the library. Adding to
`["scientific"]` keeps it out of unrelated images like `vision` or `audio` while
still reaching `ml`, `deeplearn`, `geospatial`, `timeseries`, `optimization`, `jax`, and
`probabilistic`.

### Limit a target to some Python versions

Targets build for every version in `[settings] python` unless they narrow it. When
a library has no wheels for one Python yet, narrow the target that needs it (its
descendants inherit the narrower list):

```toml
[targets.deeplearn]
parent = "ml"
description = "Deep learning with PyTorch and TensorFlow"
# tensorflow 2.21 ships no cp314 wheels; widen once tensorflow 2.22 is released.
python = ["3.13"]
```

A child may only list versions its parent builds, since it is built `FROM` the
parent's image; `just gen` rejects a matrix that breaks this. `just lock` then
fails loudly if any pin lacks a wheel for a listed Python, because every pyproject
carries a `required-environments` entry per version.

### Pin a different version for one target

When one target needs an older version to satisfy a constraint, add an
`overrides` block (mirroring how the matrix caps `pandas` and `scikit-learn` for
`timeseries` and `full`):

```toml
[packages."humanize"]
version = "4.14.0"
module = "humanize"
introduced-by = ["base"]
# explain the constraint that forces the pin here
[packages."humanize".overrides]
full = "4.12.0"
```

## Related pages

- [Configuration (matrix.toml)](../reference/configuration.md)
- [Contributing](../contributing.md)
- [Image targets](../reference/targets.md)
- [CLI & dev commands](../reference/cli.md)
