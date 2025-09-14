.. Yapocis documentation master file, created by
   sphinx-quickstart on Mon Aug 15 11:50:26 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Yapocis
==================

Yapocis makes writing and calling GPU kernels from Python easier across multiple backends: OpenCL, CUDA, and Metal.
You can find Yapocis at:

* Pypi: http://pypi.python.org/pypi/yapocis

* Github: https://github.com/seantrue/Yapocis

Backends and dependencies
-------------------------

- OpenCL via PyOpenCL: https://documen.tician.de/pyopencl/
- CUDA via PyCUDA: https://documen.tician.de/pycuda/
- Metal via a thin PyMetallic shim for testing and basic compute coverage.

Select backend using the environment variable `YAPOCIS_BACKEND` (values: `ocl`, `cuda`, `metal`).

I've been using Yapocis for about 10 years; it isn't completely raw, and appears to be nutritious.
Now for modern Python 3.

Contents:

.. toctree::
   :maxdepth: 2

   summary
   intro
   architecture
   examples
   kernels
   metal
   testing
   
Testing
-------

See the testing overview for backend markers and examples: :doc:`testing`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
