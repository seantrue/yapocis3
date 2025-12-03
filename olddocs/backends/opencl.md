# OpenCL

## Requirements

Install `PyOpencl`

## Configuration

* Backend key: set `YAPOCIS_BACKEND=ocl`
*
* Templates: place Metal Shader Language templates in `yapocis/kernels/ocl/<name>.mako`.
* Device selection: optionally set `YAPOCIS_DEVICE` to a device name or index as reported by `pymetallic.get_devices()`.
* Build cache: `YAPOCIS_CACHE` points to a directory used for compiled program caching.

## Overview



