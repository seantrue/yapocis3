"""Numba backend: pure-Python/NumPy/Numba implementations of interfaces.

This backend plugs into the same KernelManager / BoundKernels flow as OCL/Metal,
but executes kernels as Python callables. It supports a subset of parameter
semantics from the interface language:

- in/out/inout/resident map to NumPy arrays passed by reference
- out_like allocates an output array like a referenced input and returns it
- size_of/width_of/height_of/depth_of derive integers from array shapes

Implementations live under yapocis.numbax.kernels.<interface>.py
"""
