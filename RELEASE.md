# Release Process

This document describes the process for releasing a new version of yapocis to PyPI.

## Pre-Release Checklist

Before releasing, ensure:

1. All tests pass: `make test`
2. Code quality checks pass: `make check`
3. No uncommitted changes: `git status` should be clean
4. All features are documented
5. CHANGELOG.md is up to date
6. AI.md is updated with latest contributions

## Release Steps

### Automated Release (Recommended)

Use the release helper script:

```bash
make release
```

This will guide you through the process interactively.

### Manual Release

If you prefer to do it manually:

1. **Ensure clean working directory:**
   ```bash
   git status
   ```
   Commit any pending changes.

2. **Update version in pyproject.toml:**
   ```toml
   [project]
   version = "0.3.3"  # Bump version
   ```

3. **Generate changelog:**
   ```bash
   make changelog
   ```
   Review and edit CHANGELOG.md if needed.

4. **Update AI.md:**
   Document AI contributions vs human contributions for this release.

5. **Commit version bump:**
   ```bash
   git add pyproject.toml CHANGELOG.md AI.md
   git commit -m "chore: bump version to 0.3.3"
   ```

6. **Create git tag:**
   ```bash
   git tag -a v0.3.3 -m "Release v0.3.3"
   ```

7. **Build distribution packages:**
   ```bash
   make build
   ```
   This creates wheels and source distribution in `dist/`

8. **Test on TestPyPI (optional but recommended):**
   ```bash
   make publish-test
   ```
   Then test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ yapocis
   ```

9. **Publish to PyPI:**
   ```bash
   make publish
   ```

10. **Push to GitHub:**
    ```bash
    git push origin main
    git push origin v0.3.3
    ```

11. **Verify on PyPI:**
    Visit https://pypi.org/project/yapocis/ to confirm the new version is live.

## Post-Release

1. Announce the release (if applicable)
2. Create GitHub release with changelog
3. Update documentation if needed

## Versioning Scheme

yapocis uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Current development focus (0.3.x):
- Performance infrastructure
- Validation framework
- Backend maturity

## API Token Setup

### PyPI Token

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token (scope: entire account or specific project)
3. Add to `~/.pypirc`:
   ```ini
   [pypi]
   repository = https://upload.pypi.org/legacy/
   username = __token__
   password = pypi-...your-token...
   ```

### TestPyPI Token (Different from PyPI!)

1. Go to https://test.pypi.org/manage/account/token/
2. Create a new API token for TestPyPI
3. Add to `~/.pypirc`:
   ```ini
   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-...your-testpypi-token...
   ```

**Note:** PyPI and TestPyPI require separate tokens. They cannot be the same.

## Troubleshooting

### "File already exists" error

If you get an error about the file already existing on PyPI:
- You cannot overwrite a version that's already published
- Bump the version number and try again
- PyPI does not allow re-uploading the same version

### Authentication failed

Check that:
- Your `~/.pypirc` has the correct tokens
- The tokens haven't expired
- You're using `__token__` as the username (not your PyPI username)

### Build fails

Run code quality checks:
```bash
make check
make test
```

Fix any issues before building.

## Reference

- PyPI: https://pypi.org/project/yapocis/
- TestPyPI: https://test.pypi.org/project/yapocis/
- Packaging guide: https://packaging.python.org/
- Twine documentation: https://twine.readthedocs.io/
