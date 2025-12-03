# Kernels

## `anisodiff`

### Signatures

```
# interface anisodiff
def filterImage(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage2(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage3(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage4(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
def filterImage2_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
def filterImage3_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
def filterImage4_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
```


### Specification

```
interface anisodiff {
    // HxW image
    kernel filterImage(width_of input, height_of input, in float input, float scale, float size_step, out_like input);
    alias filterImage_res as filterImage(width_of input, height_of input, resident float input, float scale, float size_step, resident float output);
    // HxWx2 image
    kernel filterImage2(width_of input, height_of input, in float input, float scale, float size_step, out_like input);
    alias filterImage2_res as filterImage2(width_of input, height_of input, resident float input, float scale, float size_step, resident float output);
    // HxWx3 image
    kernel filterImage3(width_of input, height_of input, in float input, float scale, float size_step, out_like input);
    alias filterImage3_res as filterImage3(width_of input, height_of input, resident float input, float scale, float size_step, resident float output);
    // HxWx4 image
    kernel filterImage4(width_of input, height_of input, in float input, float scale, float size_step, out_like input);
    alias filterImage4_res as filterImage4(width_of input, height_of input, resident float input, float scale, float size_step, resident float output);
    };
```


### Explanation

```
Interface: anisodiff
Kernel: filterImage untiled
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

  Param 1: float scale

  Param 2: float size_step

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage2 untiled
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

  Param 1: float scale

  Param 2: float size_step

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage3 untiled
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

  Param 1: float scale

  Param 2: float size_step

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage4 untiled
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

  Param 1: float scale

  Param 2: float size_step

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage_res untiled alias for 'filterImage'
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

  Param 1: float scale

  Param 2: float size_step

  Param 3: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: filterImage2_res untiled alias for 'filterImage2'
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

  Param 1: float scale

  Param 2: float size_step

  Param 3: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: filterImage3_res untiled alias for 'filterImage3'
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

  Param 1: float scale

  Param 2: float size_step

  Param 3: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: filterImage4_res untiled alias for 'filterImage4'
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

  Param 1: float scale

  Param 2: float size_step

  Param 3: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32
```


### Stubs

```
// kernels anisodiff
__kernel void filterImage(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_out_like_input) {
}
__kernel void filterImage2(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_out_like_input) {
}
__kernel void filterImage3(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_out_like_input) {
}
__kernel void filterImage4(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_out_like_input) {
}
```

