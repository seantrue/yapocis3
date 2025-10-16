# Kernels

## `mandelbrot`

### Signatures

```
# interface mandelbrot
def mandelbrot(q: Array[complex64], maxiter: int32) -> Array[int16]:
    pass
```


### Specification

```
interface mandelbrot {
        kernel mandelbrot(in complex64 q, out_like short q, int maxiter);
};
```


### Explanation

```
Interface: mandelbrot
Kernel: mandelbrot untiled
  Param 0: complex64 * q
  Coerce 'q' to complex64
  Before call, copy buffer from array to GPU
  Pass as parameter of type *complex64

  Param derived: short * out_like_q
  Allocate an array like 'q' (opencl arg position=1, dtype=int16)
  After call, copy buffer from GPU into array
  Return as int16 (tuple position 0)

  Param 1: int maxiter
```


### Stubs

```
// kernels mandelbrot
__kernel void mandelbrot(__global complex64 *q, __global short *out_like_out_like_q, int maxiter) {
}
```

