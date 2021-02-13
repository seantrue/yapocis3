
Short examples
==============

First, a kernel that implements a 3x3 median filter on a 2-d array:

.. literalinclude:: /../yapocis/kernels/median3x3.mako
   :language: c
   :linenos:
   
and the interface specification that lets us call it:

.. literalinclude:: /../yapocis/interfaces/median3x3.interface
   :language: c
   :linenos:

The primary kernel entry point has only one visible argument to Python (the input array *a*). The other values are introspected
at run time.

Note the first/step/last aliases to the kernel function which hint buffers to be downloaded (*first*), used locally (*step*), and then
returned (*last*).

Now, the wrapper code for loading that in an even prettier way:
   
.. literalinclude:: /../yapocis/lib/median.py
   :language: python
   :linenos:

and a driver program that compares running the median filter 10 times
on each of an increasing series of image sizes.

.. literalinclude:: /../examples/median3x3_comparison.py
   :language: python
   :linenos:
   
A lot of the code in this example is included to create the following picture, which shows on my machine that
the OpenCL median 3x3 is about 8 times faster than Numpy over a fairly wide range of array sizes, and that the same kernel
using the resident buffer hinting show above runs about 44 times faster than Numpy.

.. image:: /images/median3x3-performance.png
	:width: 800
