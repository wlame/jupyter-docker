---
tags:
  - contributing
  - development
---

# Contributing

Contributions are welcome — dependency updates, new examples, docs, and fixes.
This is a short guide; the [CLI & dev commands](reference/cli.md) reference and the
[Extend the matrix](use-cases/extend-the-matrix.md) walkthrough cover the details.

<!-- TODO: link root CONTRIBUTING.md if added -->

## Dev setup

```bash
git clone https://github.com/wlame/jupyter-docker
cd jupyter-docker
```

Install the two host tools the dev workflow uses:

- **[`just`](https://github.com/casey/just)** — the task runner; run `just` to list every recipe.
- **[`uv`](https://docs.astral.sh/uv/)** — resolves and locks dependencies (`just lock` calls `uv lock`).

Docker with BuildKit is required to build or test images, but not for the fast
host-only gates below.

!!! warning "The one rule that matters"
    **[`targets/matrix.toml`](reference/configuration.md) is the single source of truth for dependencies.** Every
    `targets/<name>/pyproject.toml` and `verify_imports.py` is *generated* from it
    by [`scripts/gen_targets.py`](reference/cli.md). Never hand-edit the generated files — edit the
    matrix and regenerate. CI's consistency check fails if generated files or
    lockfiles drift from the matrix.

## Contribution workflow

1. **Edit `targets/matrix.toml`** — change a version, target membership, or override.
2. **`just gen`** — regenerate the per-target `pyproject.toml` and `verify_imports.py`.
3. **`just lock`** — re-resolve every target's committed `uv.lock`.
4. **`just ci`** — run every fast host gate (see below) before pushing.
5. **Build and test the affected target(s)** — `just build <target>` then `just test <target>`.
6. **Open a pull request against `main`** with a clear description of what changed and why.

!!! note "CI is the arbiter for image builds"
    Image changes cannot be fully verified without Docker. The GitHub Actions
    workflow builds each target, runs its import-verification and example tests,
    and publishes to GHCR on `main`. Open the PR and let CI confirm the image
    builds — see [Deployment & publishing](operations/deployment.md).

## Fast host gates

`just ci` runs the no-Docker gates in the same order CI enforces them:

| Recipe | What it checks |
|--------|----------------|
| `gen-check` | Generated per-target files match `targets/matrix.toml` |
| `lock-check` | Every target's `uv.lock` is up to date |
| `nb-check` | Example notebooks are in sync with their `.py` sources |
| `lint` | Ruff on `scripts/` and `tests/` (plus shellcheck/hadolint when installed) |
| `test-gen` | The generator test suite (`tests/test_gen_targets.py`) |

## Commit and PR norms

- Write commit messages as a **single imperative sentence ending with a period**.
- **No conventional-commit prefixes** (`feat:`, `fix:`, `chore:`, …) and **no trailers**.
- Keep pull requests focused and describe what changed and why.

## Related pages

- [CLI & dev commands](reference/cli.md) — every `just` recipe and its flags
- [Extend the matrix](use-cases/extend-the-matrix.md) — the full walkthrough for adding or upgrading a library
- [Configuration (matrix.toml)](reference/configuration.md) — the matrix schema and every key
- [Changelog](changelog.md) — how changes are tracked
