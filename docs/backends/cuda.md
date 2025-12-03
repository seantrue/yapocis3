# Cuda

## Requirements

Install `PyCuda`

## Configuration

* Backend key: set `YAPOCIS_BACKEND=cuda`
*
* Templates: place Cuda templates in `yapocis/kernels/cuda/<name>.mako`.
* Device selection: optionally set `YAPOCIS_DEVICE` to a device name or index as reported by `cuda.get_devices()`.
* Build cache: `YAPOCIS_CACHE` points to a directory used for compiled program caching.

## Overview



