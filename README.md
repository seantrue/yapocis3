# What is Yapocis?

Yapocis is "Yet another Python OpenCL interface specification".

Memorable. Easy to pronounce and spell. Clearly related to the semantics of the project.

Or not, all the good names are used and I am not in marketing.

It includes easy-to-call-from-Python, reasonably performing implementations of the Mandelbrot kernel, 3x3 median filters, image gradients, and HSI <-> RGB image conversions. I use it to develop much less generally useful implementations of bizarre image processing ideas.

## Why this happened?

*Some brief history, "living backwards" order:*

I'm putting this code up on Github to be used or snickered at. Both can be helpful. It has been used on Mac OS/X Lion with a 64-bit Python, and assumes Numpy and Matplotlib. It also assumes a working implementation of OpenCL and Pyopencl. I can't help resolve issues on dependencies, it's hard enough to make new dependencies happen.

*This in response to a note I got back from the pyopencl list:*

Sounds interesting. Please let the list know once you have public code to play with.

Andreas

*Which was in response to the following inquiry I sent to the list:*

I'm wondering if yet-another-python-opencl-interface-layer would be of interest. This layer is built on top of pyopencl, and is intended to use familiar RPC definitions to ease the special joys of talking to opencl.

I've implemented this on OS/X and it is working for me and has been stable, oh, for several days (insert smiley face here). It eliminates the need for routine glue code, and can be extended to minimize movement of data between host and opencl service.

This code is currently private, and I will be happy to either fork pyopencl and submit a pull after integration, or a separate github project.

-- Sean

> Sean True  
> Swapwizard Consulting  
> *Not my day job*

I've been trying to make these algorithms run fast enough to be useful and easy to use enough to make more core ideas testable for, oh, 25 years. Now I can work on camera resolution images in relatively quick turn around.

## Presumed highlights

```python
# Use the rpc extension to define and load the kernel as a callable.
from yapocis.rpc import kernels, interfaces
calc_fractal_opencl = kernels.loadProgram(interfaces.mandelbrot).mandelbrot
# Call it the way we like to call Python callables:
output = calc_fractal(q, maxiter)
```

```text
# RPC definition language loosely based on Apollo NCS/OSF DCE/Microsoft IDL
# outlike is a novel keyword that says: allocate for me, return as part
# of return vals.

interface mandelbrot {
      kernel mandelbrot(in complex64 *q, outlike int16 *q, in int32 maxiter);
      };
```

```c
// mandelbrot.mako is just what it has always been.
__kernel void mandelbrot(__global float2 *q,  __global short *output, int const maxiter)
{
    int gid = get_global_id(0);
    float nreal, real = 0;
    float imag = 0;
    output[gid] = 0;
    for(int curiter = 0; curiter < maxiter; curiter++) {
            nreal = real*real - imag*imag + q[gid].x;
            imag = 2* real*imag + q[gid].y;
            real = nreal;
            if (real*real + imag*imag > 4.0f)
              output[gid] = curiter;
    }
}
```

## Known misfeatures

There are several. Here are some I remember:

- Untested in any other environment, as noted.
- The `requirements.txt` does not include matplotlib or scipy, which are needed for some performance comparisons and demos.

## Changes and new features

- Reducing data movement between engine and Python code. This is implemented using an alias keyword to add alternate signatures to a kernel, and a resident parameter hint which hints the glue code to just use the buffer as-is on the engine. V0.2.2
- Framework for dealing with two-dimensional arrays auto-magically. New buffer hints: sizeof, widthof, heightof for setting arguments invisibly.
- The OpenCL engine to use is configured in `rpc.kernels`. Developer has choice of `CPU_ENGINE` and `GPU_ENGINE`.

## Future features

Other than removing the misfeatures:

- Selecting the OpenCL based on strategy (operation and image size).

