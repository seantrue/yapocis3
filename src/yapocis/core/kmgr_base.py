#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

"""
Base class for backend-specific kernel managers.

This module provides KernelManagerBase, which encapsulates all the common
functionality shared across OpenCL, Metal, CUDA, and Numba backends:
  - Template rendering (Mako)
  - Interface parsing
  - Buffer management delegation
  - Statistics tracking
  - Program loading/importing

Backend-specific subclasses need only implement:
  - get_device_manager() - Return backend device manager
  - get_kernel() - Extract kernel from compiled program
  - _backend_name - String identifier for cache directories
  - _kernel_class - Backend-specific Kernel class
  - (Optional) find_kernel_template() - Custom template resolution

Example usage:

    class OCLKernelManager(KernelManagerBase):
        _backend_name = "ocl"
        _kernel_class = OCLKernel

        def get_device_manager(self, device_name):
            return get_ocl_device_manager(device_name=device_name)

        def get_kernel(self, program, kernel_name):
            return getattr(program, kernel_name)

The base class handles:
  - Mako template lookup and rendering
  - Interface specification parsing
  - Compiled program caching
  - Kernel wrapper instantiation
  - Buffer management API (read/write/purge)
  - Statistics tracking (calls, runtime)
  - Module-level factory methods (load_program, import_program)
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from mako.lookup import TemplateLookup
from mako.template import Template

from yapocis.core.interface import Interface, get_interface, load_interface
from yapocis.core.interface_context import get_interface_context
from yapocis.utils.internals import get_yapocis_dir, get_yapocis_dirs
from yapocis.utils.stereo_typing import Context


class KernelManagerBase(ABC):
    """
    Base class for backend-specific kernel managers.

    Provides common infrastructure for:
    - Mako template rendering
    - Interface specification parsing
    - Kernel callable registration
    - Buffer management (via device manager)
    - Performance statistics tracking
    - Program compilation caching

    Subclasses must implement:
    - get_device_manager(device_name) -> DeviceManager
    - get_kernel(program, kernel_name) -> backend-specific kernel object
    - _backend_name: str - Backend identifier ("ocl", "metal", "cuda", "numbax")
    - _kernel_class: Type - Backend-specific Kernel wrapper class
    """

    # Class-level program cache: (source_hash, device_name) -> compiled program
    _programs: dict[tuple[str, str], Any] = {}

    # Backend identifier for template directories (override in subclass)
    _backend_name: str = None

    # Kernel wrapper class (override in subclass)
    _kernel_class: type = None

    @abstractmethod
    def get_device_manager(self, device_name: str | int | None):
        """
        Return backend-specific device manager.

        Must implement device selection, buffer allocation, and compilation.
        """
        raise NotImplementedError("Subclass must implement get_device_manager")

    @abstractmethod
    def get_kernel(self, program: Any, kernel_name: str) -> Any:
        """
        Extract kernel function from compiled program.

        Backend-specific extraction:
        - OpenCL: getattr(program, kernel_name)
        - Metal: pm.Kernel(device, source, func=kernel_name)
        - CUDA: program.get_function(kernel_name)
        - Numba: getattr(program, kernel_name)
        """
        raise NotImplementedError("Subclass must implement get_kernel")

    def find_kernel_template(self, base_filename: str) -> str:
        """
        Locate kernel template file for given interface name.

        Searches in order:
        1. YAPOCIS_PROJECT/kernels/{backend}/*.mako (if YAPOCIS_PROJECT set)
        2. Current working directory/kernels/{backend}/*.mako
        3. yapocis package/kernels/{backend}/*.mako

        Subclasses can override for custom template resolution (e.g., Metal's fallback).

        Args:
            base_filename: Interface name (without .mako extension)

        Returns:
            Full path to template file

        Raises:
            FileNotFoundError: If template not found
        """
        # Search in all yapocis directories (project, cwd, package)
        kernel_subdir = os.path.join("kernels", self._backend_name)
        template_filename = f"{base_filename}.mako"

        searched_paths = []
        for base_dir in get_yapocis_dirs():
            kernel_path = os.path.join(base_dir, kernel_subdir, template_filename)
            searched_paths.append(kernel_path)
            if os.path.exists(kernel_path):
                return kernel_path

        raise FileNotFoundError(
            f"Template not found for '{base_filename}' (backend={self._backend_name}). "
            f"Searched: {', '.join(searched_paths)}"
        )

    def __init__(
        self,
        interface: Interface,
        device_name: str | int | None = None,
        debug: bool = False,
        async_write: bool = False,
        options=None,
        **context: Context,
    ):
        """
        Initialize kernel manager with parsed interface.

        Args:
            interface: Parsed Interface specification
            device_name: Backend-specific device selector (None = default)
            debug: Print generated source code
            async_write: Enable asynchronous buffer writes
            options: Backend-specific compilation options
            **context: Mako template variables
        """
        # Validate subclass configuration
        if self._backend_name is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must define _backend_name class attribute"
            )
        if self._kernel_class is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must define _kernel_class class attribute"
            )

        self.interface = interface
        self.device_name = device_name
        self.callable: dict[str, Any] = {}
        self.calls = 0
        self.runtime = 0.0
        self.async_write = async_write

        # Async execution support
        self.pending_events: list[Any] = []  # Track pending async operations
        self._default_async = False  # Default to synchronous execution

        # Get backend-specific device manager
        self.dmgr = self.get_device_manager(device_name)

        # Render kernel source from Mako template
        try:
            source = self.render_program(interface.name, **context)
        except Exception:
            print(f"Source rendering issue for {interface.name}")
            print(f"Context: {context}")
            raise

        self.source = source

        if debug:
            print(f"=== {self._backend_name.upper()} Source ===")
            print(source)
            print("=" * 40)

        # Save source to cache if YAPOCIS_CACHE is set
        cache_dir = os.environ.get("YAPOCIS_CACHE")
        if cache_dir and os.path.isdir(cache_dir):
            ext = {"ocl": "cl", "metal": "metal", "cuda": "cu", "numbax": "py"}.get(
                self._backend_name, "txt"
            )
            source_file = os.path.join(cache_dir, f"{interface.name}.{ext}")
            with open(source_file, "w") as f:
                f.write(source)

        # Check program cache
        cache_key = (hash(source), str(device_name))
        if cache_key in self._programs:
            self.program = self._programs[cache_key]
        else:
            # Compile and cache
            self.program = self.dmgr.build(source, options=options)
            self._programs[cache_key] = self.program

        # Register all kernels as callables
        for kernel_spec in interface.kernels:
            # For extension kernels, lookup the base kernel they extend
            # For regular kernels, use their own name
            if hasattr(kernel_spec, "extends") and kernel_spec.extends:
                lookup_name = kernel_spec.extends
            elif hasattr(kernel_spec, "alias") and kernel_spec.alias:
                lookup_name = kernel_spec.alias
            else:
                lookup_name = kernel_spec.name

            kernel_func = self.get_kernel(self.program, lookup_name)
            self.callable[kernel_spec.name] = self._kernel_class(
                lookup_name,
                self,
                kernel_func,
                kernel_spec.parameters,
                async_write=async_write,
                traits=kernel_spec.traits,
            )

    def read(self, a):
        return self.dmgr.read(a)

    def write(self, a, async_write=None):
        async_write = self.async_write if async_write is None else async_write
        return self.dmgr.write(a, wait=not async_write)

    def purge(self):
        self.stats()
        return self.dmgr.purge()

    def reset(self):
        self.dmgr.reset()
        self.runtime = self.calls = 0

    def wait_all(self):
        """
        Wait for all pending async operations to complete.

        This method blocks until all kernels enqueued in async mode have finished.
        Backend-specific implementations should override this to handle their
        specific event types (OpenCL events, Metal command buffers, CUDA streams).

        The default implementation calls the backend-specific wait method if available.
        """
        if self.pending_events:
            # Call backend-specific wait implementation
            if hasattr(self.dmgr, "wait_events"):
                self.dmgr.wait_events(self.pending_events)
            self.pending_events.clear()

    @contextmanager
    def async_mode(self):
        """
        Context manager for async kernel execution.

        All kernel calls within this context will be enqueued without waiting,
        allowing the GPU to execute them asynchronously. The context manager
        automatically waits for all operations to complete on exit.

        Usage:
            with prog.async_mode():
                prog.conv2d_3x3(x, w, b, out, channels)  # Async
                prog.relu(out, out, size)                # Async
                prog.max_pool(out, pool, params)         # Async
            # All operations complete here
            result = prog.read(pool)

        Yields:
            self: The kernel manager instance
        """
        old_default = self._default_async
        self._default_async = True
        try:
            yield self
        finally:
            self.wait_all()
            self._default_async = old_default

    def snapshot(self, **kwargs):
        d = dict(
            interfacename=self.interface.name,
            calls=self.calls,
            runtime=self.runtime,
            device=self.dmgr.device_name,
        )
        d.update(kwargs)
        self.dmgr.snapshot(**d)

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a callable"
        if attr not in self.callable:
            print(
                "Kernel %s not found, %s available" % (attr, list(self.callable.keys()))
            )
            raise KeyError(attr)
        return self.callable[attr]

    def stats(self):
        d = dict(
            interfacename=self.interface.name,
            calls=self.calls,
            runtime=self.runtime,
            device=self.device_name,
        )
        d.update(self.dmgr.stats())
        return d

    def __str__(self):
        s = " ".join([f"{key}: {value}" for key, value in self.stats().items()])
        return s

    @classmethod
    def load_program(
        cls,
        interface_spec: str,
        device_name: str | None = None,
        debug: bool = False,
        options=None,
        **context: Context,
    ):
        """
        Factory returning runnable interface based on an interface specification.

        Args:
            interface_spec: Interface name (e.g., "mandelbrot")
            device_name: Backend-specific device selector (None = default)
            debug: Print interface and source
            options: Backend-specific compilation options
            **context: Mako template variables (override YAML context)

        Returns:
            KernelManager instance with compiled kernels

        Notes:
            - Loads template context from <interface>.yml if present
            - Programmatic context (kwargs) overrides YAML context
            - Allows externalizing template variables to YAML files
        """
        assert device_name is None or isinstance(
            device_name, (str, int, bytes)
        ), device_name

        # Load and merge YAML context with programmatic context
        # Programmatic context takes precedence
        merged_context = get_interface_context(interface_spec, **context)

        source = load_interface(interface_spec)
        source = cls.render_interface(source, **merged_context)
        interface = get_interface(source)
        if debug:
            print("Interface", interface)
        return cls(
            interface,
            device_name,
            debug=debug,
            options=options,
            **merged_context,
        )

    @classmethod
    def import_program(
        cls,
        interface_spec: str,
        device_name: str | None = None,
        namespace: bool = False,
        debug: bool = False,
        import_kernels=None,
        options=None,
        kernel_finder: Callable | None = None,
        **context: Context,
    ):
        program = cls.load_program(
            interface_spec,
            device_name=device_name,
            debug=debug,
            options=options,
            **context,
        )
        import inspect

        caller = inspect.currentframe().f_back
        target = inspect.getmodule(caller.f_code)
        for name, callable in program.callable.items():
            if import_kernels == "*" or name in context:
                name_as = context.get(name, name)
                setattr(target, name_as, callable)
        if namespace:
            return program, program.callable
        else:
            return program, list(program.callable.values())

    def render_program(self, base_filename: str, **context: Context) -> str:
        """
        Render kernel source from Mako template.

        Args:
            base_filename: Interface name (e.g., "mandelbrot")
            **context: Template variables for Mako rendering

        Returns:
            Rendered kernel source code
        """
        tfilename = self.find_kernel_template(base_filename)
        kernel_subdir = os.path.join("kernels", self._backend_name)

        # Build template lookup directories:
        # 1. Directory containing the found template (for extension templates)
        # 2. All yapocis kernel directories (for base templates like constants.mako)
        template_directories = [os.path.dirname(tfilename)]
        for base_dir in get_yapocis_dirs():
            kernel_dir = os.path.join(base_dir, kernel_subdir)
            if os.path.isdir(kernel_dir) and kernel_dir not in template_directories:
                template_directories.append(kernel_dir)

        mako_modules = os.path.join(
            get_yapocis_dir(), "mako_modules", self._backend_name
        )
        mylookup = TemplateLookup(
            directories=template_directories, module_directory=mako_modules
        )
        filename = os.path.split(tfilename)[-1]
        t = mylookup.get_template(filename)
        source = str(t.render(**context))
        return source

    @staticmethod
    def render_interface(source: str, **context: Context) -> str:
        """
        Render interface specification from Mako template.

        Used for parameterized interface specs that need template expansion
        before parsing.

        Args:
            source: Interface specification source (may contain Mako directives)
            **context: Template variables for Mako rendering

        Returns:
            Rendered interface specification
        """
        if not context:
            return source
        t = Template(source)
        return str(t.render(**context))
