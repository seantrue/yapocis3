from numpy.typing import NDArray
import numpy as np

class Mandelbrot:
    def mandelbrot(self, q: NDArray[np.complexfloating], maxiter: int) -> NDArray[np.signedinteger]: ...
