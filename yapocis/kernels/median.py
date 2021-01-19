'''
Provides 2d median filter for 2d arrays
Created on Jul 22, 2011
'''

import numpy as np
from yapocis.rpc import kernels, interfaces
from yapocis.yapocis_types import *

# Provision this program with just one median filter, fixed length of 9
first = last = median3x3_cl = None
program = kernels.import_program("median3x3", median3x3="median3x3_cl", width=9, steps=[9])

def median3x3slow(image:Array, iterations:int=1) -> Array:
    while iterations > 0:
        image = median3x3_cl(image)
        iterations -= 1
    return image.copy()

def median3x3fast(image:Array, iterations:int=1) -> Array:
    if iterations == 1:
        # One pass through
        return median3x3_cl(image)
    input = image
    output = np.zeros_like(input)
    if iterations == 2:
        # Send in data, don't retrieve
        first(input, output)
        input,output = output,input
        # Don't send in, retrieve data
        last(input, output)
        return output
    # Send in data, no retrieve
    first(input, output)
    input,output = output,input
    # Allow for first and last calls
    iterations -= 2
    while iterations > 1:
        # Iterate with resident data
        step(input, output)
        input,output = output,input
        iterations -= 1
    # And retrieve the data
    input,output = output,input
    last(input, output)
    return output.copy()

median3x3=median3x3fast
