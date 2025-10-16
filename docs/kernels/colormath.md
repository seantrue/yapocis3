# Kernels

## `colormath`

### Signatures

```
# interface colormath
def transform(input: Array[float32], transform_id: int32) -> Array[float32]:
    pass
def xyz2xyY(input: Array[float32]) -> Array[float32]:
    pass
def xyY2xyz(input: Array[float32]) -> Array[float32]:
    pass
def transform_res(input: Array[float32], transform_id: int32, output: Array[float32]) -> None:
    pass
def xyz2xyY_res(input: Array[float32], output: Array[float32]) -> None:
    pass
def xyY2xyz_res(input: Array[float32], output: Array[float32]) -> None:
    pass
```


### Specification

```
interface colormath {
    kernel transform(in float input, int transform_id, out_like float input);
    alias  transform_res as transform(resident float input, int transform_id, resident float output);
    kernel xyz2xyY(in float input, out_like float input);
    alias  xyz2xyY_res as xyz2xyY(resident float input, resident float output);
    kernel xyY2xyz(in float input, out_like float input);
    alias  xyY2xyz_res as xyY2xyz(resident float input, resident float output);
};
```


### Explanation

```
Interface: colormath
Kernel: transform untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: int transform_id

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: xyz2xyY untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: xyY2xyz untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: transform_res untiled alias for 'transform'
  Param 0: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: int transform_id

  Param 2: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: xyz2xyY_res untiled alias for 'xyz2xyY'
  Param 0: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: xyY2xyz_res untiled alias for 'xyY2xyz'
  Param 0: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32
```


### Stubs

```
// kernels colormath
__kernel void transform(__global float *input, int transform_id, __global float *out_like_out_like_input) {
}
__kernel void xyz2xyY(__global float *input, __global float *out_like_out_like_input) {
}
__kernel void xyY2xyz(__global float *input, __global float *out_like_out_like_input) {
}
```

