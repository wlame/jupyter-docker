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

## Notable changes

### October 2026 — images per Python version

- **Breaking for moving-tag users:** the plain `:<target>` tag now points to
  Python **3.14** for every target except `deeplearn`, `face`, and `full`, which
  stay on 3.13 until TensorFlow supports 3.14. To keep 3.13, pull
  `:<target>-py3.13`, or pin an immutable `-<short-sha>` tag from before the change.
- Every target is published per supported Python as `:<target>-py<X.Y>` (plus
  `-<short-sha>`). See [Deployment & publishing](operations/deployment.md).
- All dependencies were refreshed to the 2026-10-01 cutoff; scikit-learn and pandas
  are no longer held back in `timeseries` and `full`.

## Related pages

- [Deployment & publishing](operations/deployment.md) — how images are tagged and published on each push to `main`
- [About the project](about.md) — project status and design philosophy
- [Contributing](contributing.md) — how changes get proposed and land on `main`
