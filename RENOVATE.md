# Package Renovation Changes

This document tracks the improvements adopted from the pymetallic project to modernize yapocis packaging and development infrastructure.

## Completed Changes

### 1. Modern Build Backend (Hatchling)
- **Changed from**: setuptools
- **Changed to**: hatchling>=1.13
- **Benefits**:
  - Faster builds
  - Better PEP 517/518 compliance
  - Simpler configuration
  - Modern best practices

### 2. Ruff Configuration Modernization
- **Changed**: Migrated from deprecated top-level `[tool.ruff]` to modern `[tool.ruff.lint]`
- **Updated**: Comment from "handled by black" to "handled by formatter"
- **Benefits**:
  - Future-proof configuration
  - Aligns with ruff's recommended format
  - Eliminates deprecation warnings

### 3. License Correction
- **Changed from**: PSF (Python Software Foundation)
- **Changed to**: MIT
- **Benefits**:
  - Correctly reflects the project's actual license
  - Proper PyPI classification
  - Updated copyright year to 2025 in LICENSE.txt

### 4. Coverage Configuration
- **Added**: `[tool.coverage.run]` section
  - Source tracking for yapocis package
  - Excludes tests, scripts, external, and claude-debug directories
- **Added**: `[tool.coverage.report]` section
  - Excludes standard boilerplate (repr, debug, abstract methods)
  - Improves coverage report accuracy
- **Benefits**:
  - Better test coverage visibility
  - Cleaner coverage reports

### 5. Development Tools
- **Added**: isort>=5.0.0 for import sorting
- **Added**: licensecheck>=2025.1.0 for license compliance
- **Added**: pytest-benchmark>=3.4.0 for performance testing
- **Added**: `[tool.isort]` configuration
  - Profile: black (compatible formatting)
  - Multi-line output: 3
- **Benefits**:
  - Consistent import ordering
  - License compliance checking
  - Performance regression detection

### 6. Publishing Tools
- **Added**: twine>=4.0.0 for PyPI uploads
- **Added**: wheel>=0.37.0 for wheel building
- **Added**: `[tool.cibuildwheel]` for CI/CD wheel building
- **Added**: `[tool.twine]` configuration
  - Repository: pypi
  - Unsigned packages (sign: false)
  - Skip existing: false
  - Verify SSL: true
- **Benefits**:
  - Streamlined PyPI publishing workflow
  - CI/CD integration ready
  - Reproducible builds

### 7. Makefile Enhancements
- **Added**: isort to format and format-check targets
- **Added**: publish-test target (TestPyPI)
- **Added**: publish target (PyPI)
- **Added**: license-check target
- **Benefits**:
  - Simplified publishing workflow
  - License compliance automation
  - Import sorting automation

### 8. Documentation Updates
- **Updated**: CLAUDE.md with new sections:
  - Code Quality & Formatting (isort, black, ruff, mypy)
  - Building & Distribution (build, publish)
- **Created**: RENOVATE.md (this file)
- **Benefits**:
  - Better developer onboarding
  - Clear documentation of changes
  - Workflow guidance

## Package Data Handling

The migration from setuptools to hatchling required updating how package data is specified:

**Before (setuptools)**:
```toml
[tool.setuptools.package-data]
yapocis = [
    "*.interface",
    "*.mako",
    "*.cl",
    "*.ttf",
    "test.jpg",
    "test.txt",
]
```

**After (hatchling)**:
```toml
[tool.hatch.build.targets.wheel]
packages = ["yapocis"]

[tool.hatch.build.targets.wheel.force-include]
"yapocis/**/*.interface" = "yapocis"
"yapocis/**/*.mako" = "yapocis"
"yapocis/**/*.cl" = "yapocis"
"yapocis/**/*.ttf" = "yapocis"
"yapocis/**/test.jpg" = "yapocis"
"yapocis/**/test.txt" = "yapocis"
```

## Configuration Files Comparison

### Key Differences from pymetallic

**Similarities Adopted**:
- Hatchling build backend
- Modern ruff lint configuration
- Coverage configuration
- isort, twine, cibuildwheel tools
- MIT license

**Yapocis-Specific Retained**:
- Multiple backend support (opencl, metal, cuda, numba)
- Extensive dev dependencies (matplotlib, pygame, duckdb, etc.)
- Backend-specific pytest markers
- Validation framework configuration
- Multi-backend testing infrastructure

## Migration Notes

1. **Build System**: Projects using setuptools should rebuild after switching to hatchling
2. **Import Sorting**: Run `make format` to apply isort to existing codebase
3. **License**: Verify LICENSE.txt matches pyproject.toml declaration
4. **Coverage**: Run `make test-coverage` to generate coverage reports with new config

## Future Improvements

Based on pymetallic structure but not yet implemented:
- [ ] UV package manager integration (`[tool.uv]` section)
- [ ] Automated version management (setuptools_scm)
- [ ] Pre-commit hooks configuration
- [ ] CLI entry points for key scripts
- [ ] Enhanced PyPI metadata (classifiers, keywords)
- [ ] Automated changelog generation

See the `modernize` branch for these additional improvements.
