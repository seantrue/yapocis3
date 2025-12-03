# Kernels

## `gradient`

### Signatures

```
# interface gradient
def gradient(input: Array[float32], reach: int32) -> Array[float32], Array[float32]:
    pass
def gradient_res(input: Array[float32], reach: int32, grad: Array[float32], theta: Array[float32]) -> None:
    pass
```


### Specification

```
interface gradient {
    kernel gradient(width_of input, height_of input, in float input, int reach,  out_like input, out_like input );
    extension res of gradient(width_of input, height_of input, resident float input, int reach, resident float grad, resident float theta );
};
```


### Explanation

```
Interface: gradient
Kernel: gradient untiled
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: int reach

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)


Kernel: gradient_res untiled alias for 'gradient'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: int reach

  Param 2: float * grad
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 3: float * theta
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32
```


### Stubs

```
// kernels gradient
__kernel void gradient(int width_of_input, int height_of_input, __global float *input, int reach, __global float *out_like_out_like_input, __global float *out_like_out_like_input) {
}
```

