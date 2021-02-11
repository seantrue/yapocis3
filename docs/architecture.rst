
Architecture
============

Interface specification
-----------------------

Interface specifications allow Python to properly construct arguments and return values for  talking to the OpenCL code. The primary idiosyncracy is in the
direction attributes that are supported. *in*, *inout*, and *out* are common. *resident*, *outlike*, *sizeof*, *widthof*, and *heightof* are unique to the yapocis implementation.

* *in* specifies that the array data should be copied into the engine before the call is made
* *out* specifies that the array data should be copied out after the call
* *inout* specifies that the data should be copied in and then back out
* *outlike* specifies that an array should be allocated that matches the shape and datatype of the referenced array, used as an *out* array, and then returned as a return value of the call
* *resident* specifies that the call is using data already resident on the engine, that had been loaded from the argument
* *sizeof*, *widthof*, and *heightof* are used to generate parameters from the size or shape of the referenced array, which simplifies Python interface construction

Buffer management
-----------------

Much of the work in implementing the RPC is in properly and efficiently mapping Python data structures to OpenCL buffers. The general notion is that any data
that moves back and forth between processors will be in a Numpy array, and that yapocis tracks the relationship between arrays and buffers using weak references. When the array goes out of scope, the buffer will be freed as well. Hinting in the interface specification tells the runtime when data needs to be copied to or from the OpenCL engine.

Helper functions *program.write(array)* and *program.read(array)* will move data between Engine and application on demand. Useful for debugging, or for fetching results from a program implemented using engine resident data.

Engine access
_____________

The OpenCL architecture allows multiple implementations (CPU,GPU,etc), and multiple instances of a given implementation (multiple graphic cards). These are accessible via a complicated information structure. rpc.kernels specifies what
may be system specific ways for specifying a CPU or GPU based engine. 

This code is likely fragile, and is certainly early.

