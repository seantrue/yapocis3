
Summary
=======

Yapocis is a Python package that makes it easier to call high performance OpenCL kernels
from high level Python code. It depends on, and was inspired by, PyOpenCL which does the
hard work of making OpenCL callable at all. Yapocis is intended to make calling it much 
less painful.

The code is currently developed and tested with  Python 3.7 on OS/X Big Sur. It was originally developed with Python 2.7 on Snow Leopard, and there were modest changes in jumping forward about 10 years. In addition, the code has been cleaned,  refactored, and extended with kernels and interfaces for on-device graphics using OpenCL.



Quick look
----------

And here's a fast comparison between three ways of getting an embarassingly parallel thing 
done from Python:

First the easy way, with numpy:

.. literalinclude:: /../examples/demo_numpy.py
   :language: python
   :linenos:
   
Second, the hard and fast way, with raw pyopencl:

.. literalinclude:: /../examples/demo_opencl.py
   :language: python
   :linenos:     

Finally, the easy and fast way, with yapocis:

.. literalinclude:: /../examples/demo_yapocis.py
   :language: python
   :linenos: 
   
