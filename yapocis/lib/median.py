'''
Provides 2d median filter for 2d arrays
'''
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

from yapocis.utils.typing import *

# Provision this program with just one median filter, fixed length of 9
from yapocis.rpc.kernel_manager import load_program

class Median:
    def __init__(self, device_name:Optional[str]=None):
        self.kmgr = load_program("median3x3", device_name=device_name, width=9, steps=[9])
    def median3x3slow(self, image:Image1, iterations:int=1) -> Image1:
        """
        3x3 median filter reading and writing the data on each pass

        Args:
            image:
            iterations:

        Returns: A copy of the array.

        """
        while iterations > 0:
            image = self.kmgr.median3x3(image)
            iterations -= 1
        return image.copy()

    def median3x3fast(self, image:Image1, iterations:int=1) -> Image1:
        """
        3x3 median filter iterating use ping/pong buffers on the device.

        Args:
            image: 2D Array
            iterations:

        Returns: A copy of the array.

        """
        if iterations == 1:
            # One pass through
            return self.kmgr.median3x3(image)
        input = image
        output = np.zeros_like(input)
        if iterations == 2:
            # Send in data, don't retrieve
            self.kmgr.first(input, output)
            input,output = output,input
            # Don't send in, retrieve data
            self.kmgr.last(input, output)
            return output
        # Send in data, no retrieve
        self.kmgr.first(input, output)
        input,output = output,input
        # Allow for first and last calls
        iterations -= 2
        while iterations > 1:
            # Iterate with resident data
            self.kmgr.step(input, output)
            input,output = output,input
            iterations -= 1
        # And retrieve the data
        input,output = output,input
        self.kmgr.last(input, output)
        return output.copy()

    median3x3=median3x3fast
