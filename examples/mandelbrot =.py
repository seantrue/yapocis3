#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# I adapted Holger's adaptation to show use of pyopencl.rpc1
# Thanks to Holger and Ian for the very useful demo.
# Sean True
# August 15, 2011
# Previous readme follows:
# 
# I found this example for PyCuda here:
# http://wiki.tiker.net/PyCuda/Examples/Mandelbrot
#
# I adapted it for PyOpenCL. Hopefully it is useful to someone.
# July 2010, HolgerRapp@gmx.net
#
# Original readme below these lines.

# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com March 2010

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import numpy as np
import time
from functools import reduce

# You can choose a calculation routine below (calc_fractal), uncomment
# one of the three lines to test the three variations
# Speed notes are listed in the same place

# set width and height of window, more pixels take longer to calculate
from yapocis.rpc.kernel_manager import import_program
w = 1024
h = 1024

# Use the rpc extension to define and load the kernel as a callable, importing the kernel "mandelbrot" as "calc_fractal_opencl"
import_program("mandelbrot", mandelbrot="calc_fractal_opencl")



def calc_fractal_serial(q, maxiter):
    # calculate z using numpy
    # this routine unrolls calc_fractal_numpy as an intermediate
    # step to the creation of calc_fractal_opencl
    # it runs slower than calc_fractal_numpy
    z = np.zeros(q.shape, np.complex64)
    output = np.resize(np.array(0,), q.shape)
    for i in range(len(q)):
        for iter in range(maxiter):
            z[i] = z[i]*z[i] + q[i]
            if abs(z[i]) > 2.0:
                q[i] = 0+0j
                z[i] = 0+0j
                output[i] = iter
    return output

def calc_fractal_numpy(q, maxiter):
    # calculate z using numpy, this is the original
    # routine from vegaseat's URL
    output = np.resize(np.array(0,), q.shape)
    z = np.zeros(q.shape, np.complex64)

    for iter in range(maxiter):
        z = z*z + q
        done = np.greater(abs(z), 2.0)
        q = np.where(done,0+0j, q)
        z = np.where(done,0+0j, z)
        output = np.where(done, iter, output)
    return output

# choose your calculation routine here by uncommenting one of the options
calc_fractal = calc_fractal_opencl
#calc_fractal = calc_fractal_serial
#calc_fractal = calc_fractal_numpy

if __name__ == '__main__':
    import tkinter as tk
    from PIL import Image #@UnresolvedImport
    from PIL import ImageTk #@UnresolvedImport


    class Mandelbrot(object):
        def __init__(self):
            # create window
            self.root = tk.Tk()
            self.root.title("Mandelbrot Set")
            self.create_image()
            self.create_label()
            # start event loop
            self.root.mainloop()


        def draw(self, x1, x2, y1, y2, maxiter=256):
            # draw the Mandelbrot set, from numpy example
            xx = np.arange(x1, x2, (x2-x1)/w)
            yy = np.arange(y2, y1, (y1-y2)/h) * 1j
            q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex64)

            start_main = time.time()
            output = calc_fractal(q, maxiter)
            end_main = time.time()

            secs = end_main - start_main
            print(("Main took", secs))

            self.mandel = (output.reshape((h,w)) /
                    float(output.max()) * 255.).astype(np.uint8)

        def create_image(self):
            """"
            create the image from the draw() string
            """
            # you can experiment with these x and y ranges
            self.draw(-2.13, 0.77, -1.3, 1.3)
            self.im = Image.fromarray(self.mandel)
            self.im.putpalette(reduce(
                lambda a,b: a+b, ((i,0,0) for i in range(255))
            ))


        def create_label(self):
            # put the image on a label widget
            self.image = ImageTk.PhotoImage(self.im)
            self.label = tk.Label(self.root, image=self.image)
            self.label.pack()

    # test the class
    test = Mandelbrot()

