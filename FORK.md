# Fork notes

- Upstream: `https://github.com/llmsresearch/paperbanana`
- Fork: `https://github.com/petadimensionlab/paperbanana`
- Base commit: `87e01fd` (v0.3.0 + two upstream commits)
- Local branch: `local`
- Version: `0.3.0+petadim.1`

This fork adds fully local/offline, vector-first rendering. It does not publish
to PyPI; install from a GitHub release or directly from git.

## Branch model

| Branch | Purpose |
|---|---|
| `main` | Fast-forwards to upstream/main only |
| `local` | Fork features (default branch on GitHub) |

Upstream changes should be merged/replayed onto `local`; do not rewrite the
fork-only commits unless necessary.

## Sync with upstream

```bash
git fetch upstream
git checkout local
git rebase upstream/main
.venv/bin/python -m pytest -q
git push --force-with-lease origin local
```

## Release

1. Update the version in `pyproject.toml`, `paperbanana/__init__.py`, and
   `server.json` (all three must match).
2. Run `ruff check`, `ruff format --check`, and `pytest`.
3. Commit, then tag with the exact version prefixed by `v`:
   ```bash
   git tag v0.3.0+petadim.1
   git push origin v0.3.0+petadim.1
   ```
4. `.github/workflows/release.yml` builds the wheel/sdist, verifies the CLI,
   and attaches the artifacts to a GitHub release. It does **not** publish to
   PyPI.

## Security

- Never commit `.env`, generated outputs, model weights, or the local
  `exec_generate_figure.py` experiment file.
- The fork only stores endpoints and model names in example configs; no API
  keys are committed.
- Remote Tailnet endpoints should be configured through environment variables
  or an untracked `.env`.
