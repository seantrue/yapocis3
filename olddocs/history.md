# History

## Problem

Images are inherently computationally challenging. Lots of data
leads to lots of approximations, shortcuts, and undone experiments. We
need code that runs fast, and is easy to develop.

## Distant past

I\'ve tried a fair number of hardware and software configurations for
doing these experiments. Some, like CADDR-1, were not readily
available. Some, like the dedicated array processors, were
overwhelmingly expensive. Some, like Python, come both included as
batteries (pre-installed) and come with batteries included (Python
packages like PIL and Numpy). Things I actually tried:

-   [CADDR](https://en.wikipedia.org/wiki/Lisp_machine)-1 & convolution hardware (1980)
-   Intel 3/486, lots of memory, and C/Assembly language with DOS extenders like [DOS/4G](https://doom.fandom.com/wiki/DOS_%E2%81%84_4GW) (1985)
-   Pentium computers, Python, and [PIL](https://en.wikipedia.org/wiki/Python_Imaging_Library) (1995)
-   Fast computers, [Python](https://image.slidesharecdn.com/why-i-love-python-1214879408845659-9/95/why-i-love-python-7-728.jpg?cb=1214853919), Numeric/Numarray/Numpy, and PIL (2000-2005)
-   Fast computers, Python, Numpy, PIL, and Pyrex/Cython (2010)

Things I ignored:

-   Cluster based multi-processing

I have/had a large number of computers (meaning, too many), which made cluster-based computing worth considering. But the computers are usually doing something else.
and cluster based computer means cluster management. Too much overhead, both literally and in headaches.

## Alternatives in 2011
All things I actually tried.

-   Multi-core multi-processing
-   GPU based solutions
    -   NVidia & Cuda
    -   Everybody else & OpenCL


GPU based solutions were promising, and I got one for free with my Mac. In fact, I got two, because Apple included a CPU implementation of OpenCL for Intel which is really fast. And I didn't have an Nvidia card, but had a good AMD graphics card with OpenCL support.

CUDA was, and is, less widely available and has massive vendor lock-in 
consequences.

Writing in OpenCL is a lot like writing in C. It helps to have other
peoples examples, and to remember that premature optimization is not a
good idea. Optimized OpenCL can be hard to read, and can run at near FTL speed. Unoptimized OpenCL can be very easy to read, and can run 10 times faster than Numpy. 

PyOpenCL is terrific. The procedure call interface is a little rough: we can do better. Back to the future, with a modified Apollo RPC / Apollo NCS RPC, OSF DCE RPC, MS RPC (COM) interface specification (sweet and nutritious) and a home grown but easy to use Python thunking layer (bitter, but you can mostly ignore it). With a little care, and syntactic sugar, even the bitter parts can go down pretty well.

## Current alternatives
Things I am currently doing.

- Taking Numpy closer to the metal with accelerators like Numba
- OpenCL almost everywhere


There are some newer alternatives to writing almost in Python and having it compiled to machine instructions. Numba is particularly good for accelerating operations over Numpy arrays where there is no baked in support in Numpy and OpenCL is not a good fit,

The world is a lot more complicated than in 2011. GPU computing has largely evolved in response to the needs of deep neural network development. Nvidia has been the default there for a long time. But there are signs of change.

OpenCL development has continued, although Apple is deprecating it in favor of the proprietary Metal architecture (not yet an option for general compute kernels). Intel and AMD have doubled down on OpenCL across a variety of architectures. Intel supports it on CPU, on Intel graphics hardware, on the Myriad Neural Compute chip, and on FPGA. FPGAs are becoming practical in the enthusiast world. AMD continues support for OpenCL for their GPUs.

