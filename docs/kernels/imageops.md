# Kernels

## `imageops`

### Signatures

```
# interface imageops
def clamp_pixels(input: Array[float32], low: float32, high: float32) -> Array[float32]:
    pass
def stretch(input: Array[float32], input_min: float32, input_max: float32, low: float32, high: float32) -> Array[float32]:
    pass
def binarize(input: Array[float32], threshold: float32) -> Array[float32]:
    pass
def histogram(input: Array[float32], low: float32, high: float32) -> Array[int32]:
    pass
def histogram_res(input: Array[float32], low: float32, high: float32, hst: Array[int32]) -> None:
    pass
```


### Specification

```
interface imageops {
    kernel clamp_pixels(in float input, float low, float high, out_like input);
    kernel stretch(in float input, float input_min, float input_max, float low, float high, out_like input);
    kernel binarize(in float input, float threshold, out_like input);
    kernel histogram(in float input, size_of input, float low, float high, out int hst, size_of hst);
    alias histogram_res as histogram(resident float input, size_of input, float low, float high, resident int hst, size_of hst);
};
```


### Explanation

```
Interface: imageops
Kernel: clamp_pixels untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float low

  Param 2: float high

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: stretch untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float input_min

  Param 2: float input_max

  Param 3: float low

  Param 4: float high

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=5, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: binarize untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float threshold

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: histogram untiled
  Param 0: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: int input
  Pass in as size_of `input`
  Pass as parameter of type int32

  Param 1: float low

  Param 2: float high

  Param 3: int * hst
  Allocate an array for hst
  After call, copy buffer from GPU into array
  Return as int32 (tuple position 0)

  Param derived: int hst
  Pass in as size_of `hst`
  Pass as parameter of type int32


Kernel: histogram_res untiled alias for 'histogram'
  Param 0: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int input
  Pass in as size_of `input`
  Pass as parameter of type int32

  Param 1: float low

  Param 2: float high

  Param 3: int * hst
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *int32

  Param derived: int hst
  Pass in as size_of `hst`
  Pass as parameter of type int32
```


### Stubs

```
// kernels imageops
__kernel void clamp_pixels(__global float *input, float low, float high, __global float *out_like_out_like_input) {
}
__kernel void stretch(__global float *input, float input_min, float input_max, float low, float high, __global float *out_like_out_like_input) {
}
__kernel void binarize(__global float *input, float threshold, __global float *out_like_out_like_input) {
}
__kernel void histogram(__global float *input, int size_of_input, float low, float high, __global int *hst, int size_of_hst) {
}
```

