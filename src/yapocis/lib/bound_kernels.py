#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
import os
from collections.abc import Callable

from yapocis.core import load_program
from yapocis.utils.stereo_typing import *


class BoundKernels:
    """Lightweight wrapper that binds an interface to a backend KernelManager.

    Notes:

    - Honors `YAPOCIS_BACKEND` automatically, or accept explicit `backend`.
    - `device_name` may be a device index or name string.
    - Extra `kwargs` are forwarded to the backend loader (e.g., Mako vars,
      `debug`, `options`, `async_write`).
    """

    INTERFACE = None
    load_program_config = {}
    use_async = None  # Can be overridden by subclasses (True/False) or left as None
    use_async_write = (
        None  # Can be overridden by subclasses (True/False) or left as None
    )

    def __init__(
        self,
        device_name: DeviceID | None = None,
        backend: str | None = None,
        use_async: bool | None = None,
        async_write: bool | None = None,
        **kwargs,
    ):
        assert device_name is None or isinstance(
            device_name, (int, str, bytes)
        ), device_name

        # Three-level resolution for use_async: instance > class > environment
        if use_async is not None:
            # Instance parameter takes highest priority
            self.use_async = use_async
        elif self.__class__.use_async is not None:
            # Class variable takes second priority
            self.use_async = self.__class__.use_async
        else:
            # Environment variable is the fallback
            self.use_async = os.environ.get("YAPOCIS_ASYNC", "").lower() in (
                "1",
                "true",
                "yes",
            )

        # Three-level resolution for async_write: instance > class > environment
        if async_write is not None:
            # Instance parameter takes highest priority
            self.async_write = async_write
        elif self.__class__.use_async_write is not None:
            # Class variable takes second priority
            self.async_write = self.__class__.use_async_write
        else:
            # Environment variable is the fallback
            self.async_write = os.environ.get("YAPOCIS_ASYNC_WRITE", "").lower() in (
                "1",
                "true",
                "yes",
            )

        # Pass backend through so callers can force Metal/CUDA/OCL per instance.
        self.kmgr = load_program(
            self.INTERFACE,
            backend=backend,
            device_name=device_name,
            async_write=self.async_write,
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

    def write(self, a, wait=None):
        """
        Write array to GPU buffer.

        Args:
            a: Array to write
            wait: If True, wait for transfer to complete (synchronous)
                  If False, return immediately (asynchronous)
                  If None, use instance's async_write setting
        """
        if wait is None:
            # Use the instance's async_write setting
            # Note: kmgr.write expects async_write parameter, not wait
            # When async_write=True, we want wait=False
            self.kmgr.write(a, async_write=self.async_write)
        else:
            # User explicitly specified wait behavior
            # Convert wait -> async_write for kmgr
            self.kmgr.write(a, async_write=not wait)

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
        if os.environ.get("YAPOCIS_AUTO_PURGE", "").lower() in ("1", "true", "yes"):
            try:
                if hasattr(self, "kmgr"):
                    self.kmgr.purge()
            except Exception:
                # Silently ignore errors during cleanup
                pass
