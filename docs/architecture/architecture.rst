
Architecture
============

Interface specification
-----------------------

Interface specifications allow Python to properly construct arguments and return values for  talking to the backend code. The primary idiosyncracy is in the
direction attributes that are supported. *in*, *inout*, and *out* are common. *resident*, *out_like*, *size_of*, *width_of*, and *height_of* are unique to the yapocis implementation.

* *in* specifies that the array data should be copied into the engine before the call is made
* *out* specifies that the array data should be copied out after the call
* *inout* specifies that the data should be copied in and then back out
* *out_like* specifies that an array should be allocated that matches the shape and datatype of the referenced array, used as an *out* array, and then returned as a return value of the call
* *resident* specifies that the call is using data already resident on the engine, that had been loaded from the argument
* *size_of*, *width_of*, and *height_of* are used to generate parameters from the size or shape of the referenced array, which simplifies Python interface construction

Buffer management
-----------------

Much of the work in implementing the RPC is in properly and efficiently mapping Python data to backend buffers. The general notion is that any data
that moves back and forth between processors will be in a Numpy array, and that yapocis tracks the relationship between arrays and buffers using weak references. When the array goes out of scope, the buffer will be freed as well. Hinting in the interface specification tells the runtime when data needs to be copied to or from the backend engine.

Helper functions *program.write(array)* and *program.read(array)* will move data between Engine and application on demand. Useful for debugging, or for fetching results from a program implemented using engine resident data.

Device access
_____________

The backend architecture allows multiple implementations (CPU,GPU,etc), and multiple instances of a given implementation (graphic cards, drivers). These are accessible via a backend/device abstraction.

BoundKernels - High-Level Wrapper Pattern
------------------------------------------

The ``BoundKernels`` class provides a lightweight, Pythonic wrapper around kernel interfaces, simplifying common usage patterns and providing a clean API for library code.

**Core Concept**

``BoundKernels`` is a base class that automatically:

* Loads a kernel interface on instantiation
* Honors ``YAPOCIS_BACKEND`` environment variable
* Allows explicit backend and device selection
* Provides attribute-based access to kernel functions
* Includes helper methods for buffer management

**Basic Usage**

Subclasses define which interface to load by setting the ``INTERFACE`` class variable:

.. code-block:: python

   from yapocis.lib.bound_kernels import BoundKernels

   class Mandelbrot(BoundKernels):
       INTERFACE = "mandelbrot"

   # Create instance (automatically loads interface)
   mb = Mandelbrot()  # Uses YAPOCIS_BACKEND environment

   # Or specify backend explicitly
   mb = Mandelbrot(backend="metal")

   # Or select specific device
   mb = Mandelbrot(backend="ocl", device_name=0)

   # Call kernels via attribute access
   result = mb.mandelbrot(grid, maxiter=256)

**Attribute-Based Kernel Access**

``BoundKernels`` uses ``__getattr__`` to provide dynamic access to kernels. When you access an attribute that doesn't exist on the instance, it checks if a kernel with that name exists in the loaded interface:

.. code-block:: python

   class Graphics4(BoundKernels):
       INTERFACE = "graphics4"

   g = Graphics4()

   # These become kernel calls
   g.circles(canvas, coords, colors)  # Calls graphics4.circles kernel
   g.lines(canvas, line_data, distance, colors)  # Calls graphics4.lines kernel

If the kernel doesn't exist, a ``KeyError`` is raised with available kernels listed.

**Helper Methods**

``BoundKernels`` provides convenience methods for buffer management:

* **write(array)**: Copy array data to GPU without executing a kernel
* **read(array)**: Read array data from GPU
* **purge()**: Release all GPU buffers associated with this instance
* **snapshot(**kwargs)**: Capture performance/timing snapshot

These wrap the underlying ``KernelManager`` methods for easier access.

**Advanced Configuration**

Subclasses can customize loading behavior with ``load_program_config``:

.. code-block:: python

   class CustomKernel(BoundKernels):
       INTERFACE = "mykernel"
       load_program_config = {
           "debug": True,
           "options": "-cl-fast-relaxed-math",
       }

Additional kwargs passed to ``__init__`` are forwarded to the backend loader, allowing per-instance customization:

.. code-block:: python

   # Pass Mako template variables
   kernel = CustomKernel(TILE_SIZE=16, UNROLL_FACTOR=4)

**Library Integration**

Most high-level yapocis libraries (``Graphics4``, ``HSI``, ``Colormath``, etc.) inherit from ``BoundKernels``, providing:

* Consistent initialization patterns
* Automatic backend selection
* Clean Python APIs wrapping low-level kernels
* Easy device/backend switching for testing

**Examples from the Library**

.. code-block:: python

   # Simple wrapper - just specifies interface
   class HSI(BoundKernels):
       INTERFACE = "hsi"

   # Extended wrapper with helper methods
   class Graphics4(AbstractGraphics):  # AbstractGraphics extends BoundKernels
       INTERFACE = "graphics4"

       def circles(self, canvas, coords, color=None, **kwargs):
           # Validate/coerce inputs
           coords = self.coerce_coords(coords, width=3)
           colors = self.coerce_colors(coords, color=color)

           # Call underlying kernel
           self.kmgr.callable['circles'](canvas, coords, colors)

**Benefits**

* **Reduced boilerplate**: No manual kernel loading or device setup
* **Consistent interface**: All libraries use same initialization pattern
* **Testability**: Easy to switch backends for testing
* **Flexibility**: Environment variables or explicit control
* **Discoverability**: Attribute errors list available kernels

KernelDraw - An Elaborate BoundKernels Example
-----------------------------------------------

The ``KernelDraw`` family (``KernelDraw1``, ``KernelDraw3``, ``KernelDraw4``) demonstrates a sophisticated multi-layer architecture built on top of ``BoundKernels``, providing an immediate-mode and retained-mode graphics API.

**Architecture Layers**

The KernelDraw architecture consists of three abstraction layers:

1. **BoundKernels Layer**: Low-level kernel access (``Graphics4`` extends ``BoundKernels``)
2. **AbstractGraphics Layer**: Direct kernel wrapper with parameter coercion
3. **AbstractKernelDraw Layer**: Display list management and batch rendering

**Layer 1: Graphics Classes (BoundKernels)**

The base ``Graphics`` classes (``Graphics1``, ``Graphics3``, ``Graphics4``) extend ``BoundKernels`` and provide direct access to graphics kernels with automatic parameter validation:

.. code-block:: python

   class Graphics4(AbstractGraphics):
       INTERFACE = "graphics4"
       CHANNELS = 4

       def lines(self, image, coords, **kwargs):
           coords = self.coerce_coords(coords, width=4)  # x1,y1,x2,y2
           colors = self.coerce_colors(coords, **kwargs)
           widths = self.coerce_widths(coords, **kwargs)
           self.kmgr.lines(image, coords, widths, colors)

       def circles(self, image, coords, **kwargs):
           coords = self.coerce_coords(coords, width=3)  # x,y,radius
           colors = self.coerce_colors(coords, **kwargs)
           self.kmgr.circles(image, coords, colors)

**Layer 2: AbstractGraphics (Immediate Mode)**

``AbstractGraphics`` provides:

* **Parameter coercion**: Convert various input formats to kernel-ready arrays
* **Batch processing**: Accept lists of primitives and process them efficiently
* **Helper methods**: ``write()``, ``read()``, ``snapshot()`` for buffer management

.. code-block:: python

   g = Graphics4()
   canvas = np.zeros((512, 512, 4), dtype=np.float32)

   # Immediate mode - draws immediately to GPU
   g.circles(canvas, [(256, 256, 100)], color=[1.0, 0.0, 0.0, 1.0])
   g.lines(canvas, [(0, 0, 512, 512)], color=[0.0, 1.0, 0.0, 1.0])

   result = g.read(canvas)

**Layer 3: KernelDraw (Retained Mode)**

``KernelDraw`` adds display list functionality for deferred rendering:

.. code-block:: python

   class KernelDraw4(AbstractKernelDraw):
       COLOR_WIDTH = 4
       GRAPHICS = Graphics4

   kd = KernelDraw4(canvas)

   # Retained mode - build display list
   kd.circle((256, 256, 100), color=[1.0, 0.0, 0.0, 1.0])
   kd.line((0, 0, 512, 512), color=[0.0, 1.0, 0.0, 1.0])
   kd.rect((100, 100, 200, 200), color=[0.0, 0.0, 1.0, 1.0])

   # Batch render all primitives
   kd.render()

   result = kd.graphics.read(canvas)

**Display List Management**

``AbstractKernelDraw`` maintains separate display lists for each primitive type:

* ``_lines``, ``_rects``, ``_circles``, ``_triangles``, ``_ellipses``
* ``_sprites``, ``_draw_strings``

The ``render()`` method processes all lists in order, enabling:

* **Batching**: Group similar primitives for efficient GPU dispatch
* **Chunking**: Process large lists in manageable batches (configurable chunk_size)
* **Deferred execution**: Build complex scenes before any GPU work

**Advanced Features**

**Text Rendering**:

.. code-block:: python

   kd.draw_string((100, 100), "Hello World",
                  font="helvetica", size=24,
                  color=[1.0, 1.0, 1.0, 1.0],
                  rotation=45)
   kd.render()

**Sprite Compositing**:

.. code-block:: python

   sprites = [load_sprite("player.png"), load_sprite("enemy.png")]
   kd.sprite((100, 100), sprites[0], alpha=0.8)
   kd.sprite((200, 150), sprites[1], alpha=1.0)
   kd.render_sprites()

**Global State Management**:

.. code-block:: python

   kd.color = [1.0, 0.0, 0.0, 1.0]  # Set default color
   kd.alpha = 0.5                    # Set default alpha

   # All subsequent calls use these defaults unless overridden
   kd.circle((100, 100, 50))         # Uses default color/alpha
   kd.circle((200, 200, 75), alpha=1.0)  # Override alpha only

**Why This Architecture?**

The layered design provides multiple usage patterns:

1. **Direct kernel access** (``BoundKernels``): Maximum control, minimum overhead
2. **Immediate mode** (``AbstractGraphics``): Simple API, automatic parameter handling
3. **Retained mode** (``KernelDraw``): Display lists, batching, scene management

Each layer builds on the previous, allowing developers to choose the appropriate abstraction level:

* Performance-critical code can use ``Graphics4`` directly
* Interactive applications benefit from ``KernelDraw``'s display lists
* Both share the same underlying kernel infrastructure via ``BoundKernels``

This demonstrates how ``BoundKernels`` serves as a foundation for building sophisticated, domain-specific APIs while maintaining the flexibility to drop down to lower levels when needed.


