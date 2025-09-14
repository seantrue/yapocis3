Metal Backend (PyMetallic)

- Backend key: set `YAPOCIS_BACKEND=metal` (also accepts `metallic` or `mtl`).
- Templates: place Metal Shader Language templates in `yapocis/metalkernels/<name>.mako`.
  - The loader falls back to `yapocis/kernels/<name>.mako` if a Metal-specific template is not present.
  - OpenCL templates are unlikely to compile as-is on Metal; use only for early bring-up.
- Device selection: optionally set `YAPOCIS_ENGINE` to a device name or index as reported by `pymetallic.get_devices()`.
- Build cache: `YAPOCIS_CACHE` points to a directory used for compiled program caching if supported by PyMetallic.

Requirements

- Install `PyMetallic` (Apple Metal Python binding providing a PyOpenCL-like API).
  The metallic backend assumes a Program/Context/Queue/Buffer and `enqueue_copy` interface similar to PyOpenCL.

