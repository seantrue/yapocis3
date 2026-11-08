"""The ``rustcpu`` backend: yapocis-rs's CPU backend through its Python bindings.

``YAPOCIS_BACKEND=rustcpu`` runs every kernel as a Rust port of its OpenCL
source, on the host CPU, with the same argument derivation and dispatch sizes
as the Rust runtime. It needs the ``yapocis-rs`` wheel (module ``yapocis_rs``),
built from yapocis-rs's ``bindings/python``.
"""
