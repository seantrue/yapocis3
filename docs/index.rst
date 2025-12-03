Yapocis Documentation
=====================

.. image:: https://badge.fury.io/py/yapocis.svg
   :target: https://badge.fury.io/py/yapocis
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/yapocis.svg
   :target: https://pypi.org/project/yapocis/
   :alt: Python Versions

.. image:: https://img.shields.io/github/license/seantrue/yapocis3.svg
   :target: https://github.com/seantrue/yapocis3/blob/main/LICENSE
   :alt: License

Overview
--------

Yapocis is a multi-backend GPU/CPU computing framework for image processing and graphics operations. It provides a
unified Python interface to OpenCL, CUDA, Metal, and Numba backends through an IDL-like interface definition language
and Mako template system.

The framework enables developers to write compute kernels once and execute them across different hardware platforms
without code modification. Backend selection occurs at runtime through environment variables or programmatic
configuration.

**Key Features:**

- **IDL-based interface system** for clean kernel definitions
- **Multi-backend support**: OpenCL, Metal, CUDA, and Numba
- **High-level graphics libraries** for 2D primitives and image processing
- **Performance tracking** with DuckDB-backed metrics
- **Validation framework** for pixel-exact regression testing
- **Modern packaging** with Python 3.10-3.13 support

**Links:**

- PyPI: https://pypi.python.org/pypi/yapocis
- GitHub: https://github.com/seantrue/yapocis3
- Issues: https://github.com/seantrue/yapocis3/issues

Quick Start
-----------

**Installation:**

.. code-block:: bash

   # Basic installation
   pip install yapocis

   # With OpenCL support
   pip install yapocis[opencl]

   # With all backends
   pip install yapocis[all-backends]

**Simple Example:**

.. code-block:: python

   from yapocis import core
   from yapocis.core import get_manager
   import numpy as np

   # Select backend (or use YAPOCIS_BACKEND env var)
   mgr = get_manager("metal")  # "ocl", "cuda", "metal", "numba"

   # Load and execute kernel
   prog = core.load_program("mandelbrot")
   grid = np.mgrid[-2:2:1024j, -2:2:1024j].astype(np.complex64)
   result = prog.mandelbrot(grid, 256)  # positional args for kernel params

Supported Backends
------------------

**OpenCL** (via PyOpenCL)
   Cross-platform GPU/CPU support. Mature and well-tested.

**Metal** (via PyMetallic)
   Apple GPU acceleration. macOS only.

**CUDA** (via PyCUDA)
   NVIDIA GPU support. All kernel interfaces supported.

**Numba**
   CPU fallback with JIT compilation. Automatic selection when GPU unavailable.

Current State (v0.3.x)
----------------------

**Version 0.3.x Development Focus:**

1. **Performance Infrastructure**: Comprehensive performance tracking system with DuckDB storage, cross-backend
   comparison, and longitudinal analysis.

2. **Validation Framework**: Automated regression testing with pixel-exact validation across backends.

3. **Modern Packaging**: Migration to hatchling, modern tooling (ruff, isort, mypy), and Python 3.13 support.

4. **Backend Maturity**: Complete feature parity across OpenCL, Metal, and Numba backends with extensive test coverage.

5. **Type Modernization**: Python 3.9+ native type annotations (PEP 585, PEP 604).

**Modern Infrastructure:**

- Build system: Hatchling (PEP 517/518)
- Development tools: isort, black, ruff, mypy
- Testing: pytest with coverage tracking
- Changelog: Automated with git-cliff
- Python: 3.10-3.13 support

Documentation
-------------

.. toctree::
   :maxdepth: 1

   getting-started/index

.. toctree::
   :maxdepth: 1
   :caption: Application Development

   user-guide/index
   user-guide/summary
   user-guide/examples
   user-guide/testing
   user-guide/intro

.. toctree::
   :maxdepth: 1
   :caption: Kernel Development

   kernel-guide/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   kernels/index
   reference/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Architecture

   architecture/index

.. toctree::
   :caption: Backends
   :maxdepth: 1

   backends/index

.. toctree::
   :maxdepth: 1
   :caption: Project Documentation

   README </../README.md>
   CHANGELOG </../CHANGELOG.md>

.. toctree::
   :maxdepth: 1
   :caption: Project History & Credits

   HISTORY <../HISTORY.md>
   CONTRIBUTORS <../CONTRIBUTORS.md>
   AI <../AI.md>
   AGENTS <../AGENTS.md>

Historical Context
------------------

Yapocis has evolved over more than a decade, adapting to changes in GPU computing, Python packaging, and development
practices. The current version represents a significant modernization while maintaining the core design philosophy of
making GPU computing accessible through clean interfaces.

For the archaeological record of how this came to be, see :doc:`HISTORY <../HISTORY.md>`.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
