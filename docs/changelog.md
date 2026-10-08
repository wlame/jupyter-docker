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

!!! note "When a root `CHANGELOG.md` is added"
    Once a `CHANGELOG.md` file exists at the repository root, this page should
    render it directly rather than duplicating its content — for example with a
    [pymdownx snippet](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/)
    include (`--8<-- "CHANGELOG.md"`). It is intentionally **not** included yet:
    `pymdownx.snippets` runs with `check_paths: true`, so referencing a file that
    does not exist would fail the `mkdocs build --strict` step in CI.

<!-- TODO: import CHANGELOG.md when it exists -->

## Related pages

- [Deployment & publishing](operations/deployment.md) — how images are tagged and published on each push to `main`
- [About the project](about.md) — project status and design philosophy
- [Contributing](contributing.md) — how changes get proposed and land on `main`
