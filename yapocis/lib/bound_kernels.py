#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import os
from yapocis.core import load_program
from yapocis.utils.stereo_typing import *
from typing import Any, Callable


class BoundKernels:
    """Lightweight wrapper that binds an interface to a backend KernelManager.

    Notes
    - Honors `YAPOCIS_BACKEND` automatically, or accept explicit `backend`.
    - `device_name` may be a device index or name string.
    - Extra `kwargs` are forwarded to the backend loader (e.g., Mako vars,
      `debug`, `options`, `async_write`).
    """

    INTERFACE = None
    load_program_config = {}

    def __init__(
        self,
        device_name: DeviceID | None = None,
        backend: str | None = None,
        **kwargs,
    ):
        assert device_name is None or isinstance(
            device_name, (int, str, bytes)
        ), device_name
        # Pass backend through so callers can force Metal/CUDA/OCL per instance.
        self.kmgr = load_program(
            self.INTERFACE,
            backend=backend,
            device_name=device_name,
            **self.load_program_config,
            **kwargs,
        )

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a kernel"
        if attr not in self.kmgr.callable:
            print(
                "BoundKernel %s not found, %s available"
                % (attr, list(self.kmgr.callable.keys()))
            )
            raise KeyError(attr)
        return self.kmgr.callable[attr]

    def write(self, a):
        self.kmgr.write(a)

    def read(self, a):
        return self.kmgr.read(a)

    def snapshot(self, **kwargs):
        self.kmgr.purge()
        self.kmgr.snapshot(**kwargs)
        self.kmgr.reset()

    def purge(self):
        self.kmgr.purge()

    def __enter__(self):
        """Context manager entry - return self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup on exit."""
        # Purge buffers when exiting context
        self.kmgr.purge()
        return False  # Don't suppress exceptions

    def __del__(self):
        """
        Conditionally purge buffers when instance is deleted.

        Automatic purge is controlled by YAPOCIS_AUTO_PURGE environment variable:
        - Set to '1', 'true', or 'yes' to enable automatic purge (useful for loops)
        - Disabled by default to prevent issues during pytest teardown

        For explicit cleanup, use context managers or call purge() manually.
        """
        # Only purge if explicitly enabled via environment variable
        if os.environ.get('YAPOCIS_AUTO_PURGE', '').lower() in ('1', 'true', 'yes'):
            try:
                if hasattr(self, 'kmgr'):
                    self.kmgr.purge()
            except Exception:
                # Silently ignore errors during cleanup
                pass
