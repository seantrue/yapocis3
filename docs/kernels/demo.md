# Kernels

## `demo`

### Signatures

```
# interface demo
def sum(a: Array[float32], b: Array[float32]) -> Array[float32]:
    pass
def sum_res(a: Array[float32], b: Array[float32]) -> Array[float32]:
    pass
```


### Specification

```
interface demo {
    kernel sum(in float a, in float b, out_like a);
    extension res of sum(in float a, in float b, out float c);
    };
```


### Explanation

```
Interface: demo
Kernel: sum untiled
  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float * b
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_a
  Allocate an array like 'a' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: sum_res untiled alias for 'sum'
  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float * b
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * c
  Allocate an array for c
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels demo
__kernel void sum(__global float *a, __global float *b, __global float *out_like_out_like_a) {
}
```

