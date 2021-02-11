
Introduction
============

Problem
-------

Image processing is inherently computationally challenging. Lots of data leads to lots of approximations,
shortcuts, and undone experiments. We need code that runs fast, and is easy to develop.

Past solutions
--------------

I've tried a fair number of hardware and software configurations for doing
these experiments. Some, like CADDR-1, are were not readily available. Some, like the dedicated array processors, were overwhelmingly expensive. Some, like Python, come both included as batteries (pre-installed) and come with batteries included (Python packages like PIL and Numpy).

* CADDR-1 & convolution hardware
* Skipping lots of dedicated array processors
* Intel 3/486, lots of memory, and C/Assembly language
* Pentium computers, Python, and PIL
* Fast computers, Python, and Numpy

Current alternatives
--------------------

* Cluster based multi-processing
* Multi-core multi-processing
* GPU based solutions

  * NVidia  & Cuda
  * Everybody else & OpenCL

I have the usual complement (meaning, too many) of computers, which would certainly make cluster-based computing worth considering. But images are large, image movement is a bottleneck, and the computers are usually doing something else.

Multi-core multi-processing is more efficient, but image processing really looks
like SIMD (single-instruction, multiple-data stream computing), and that is not a strength for message-passing type architectures.

GPU based solutions are promising, and I got one for free with my Mac. In fact, I got two, because Apple includes a CPU implementation of OpenCL which is really fast. And while I don't have an NVidia card, OpenCL runs directly (and fast) on Intel and AMD CPUs, not to mention ATI graphics chips. And, more importantly, my computer comes *out of the box* with a decent implementation.

Writing in OpenCL is a lot like writing in C. It helps to have other peoples examples, and to remember that premature optimization is
not a good idea. Optimized OpenCL can be hard to read, and can run at near FTL speed. Unoptimized OpenCL can be very easy to read, and can run
10 times faster than Numpy. 
  
PyOpenCL is terrific. The procedure call interface is a little rough: we can do better. Back to the future, with a modified 
Apollo RPC / Apollo NCS RPC, OSF DCE RPC, MS RPC (COM) interface specification (sweet and nutritious) and a tacky but easy to use
Python thunking layer (bitter, but you only have to swallow it once). And with a little care, and syntactic sugar, even the bitter parts can go down pretty well.
