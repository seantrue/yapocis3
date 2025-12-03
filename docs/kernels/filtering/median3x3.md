# Kernels

## `median3x3`

### Signatures

```
# interface median3x3
def median3x3(a: Array[float32]) -> Array[float32]:
    pass
def median3x3_res(a: Array[float32]) -> Array[float32]:
    pass
```


### Specification

```
interface median3x3 {
    kernel median3x3(size_of int a, height_of int a, in float a, out_like a );
    extension res of median3x3(size_of int a, height_of int a, in float a, out float ret );
};
```


### Explanation

```
Interface: median3x3
Kernel: median3x3 untiled
  Param derived: int a
  Pass in as size_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_a
  Allocate an array like 'a' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: median3x3_res untiled alias for 'median3x3'
  Param derived: int a
  Pass in as size_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float * ret
  Allocate an array for ret
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels median3x3
__kernel void median3x3(int size_of_a, int height_of_a, __global float *a, __global float *out_like_out_like_a) {
}
```

