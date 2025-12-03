# Kernels

## `zcs`

### Signatures

```
# interface zcs
def zcs1(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs2(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs3(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs4(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def sub(input1: Array[float32], input2: Array[float32]) -> Array[float32]:
    pass
def zcs1_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs2_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs3_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs4_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def sub_res(input1: Array[float32], input2: Array[float32], output: Array[float32]) -> None:
    pass
```


### Specification

```
interface zcs {
    kernel zcs1(width_of input, height_of input, float threshold, in float input, out_like input);
    alias zcs1_res as zcs1(width_of input, height_of input, float threshold, resident float input, resident float zcs);
    kernel zcs2(width_of input, height_of input, float threshold, in float input, out_like input);
    alias zcs2_res as zcs2(width_of input, height_of input, float threshold, resident float input, resident float zcs);
    kernel zcs3(width_of input, height_of input, float threshold, in float input, out_like input);
    alias zcs3_res as zcs3(width_of input, height_of input, float threshold, resident float input, resident float zcs);
    kernel zcs4(width_of input, height_of input, float threshold, in float input, out_like input);
    alias zcs4_res as zcs4(width_of input, height_of input, float threshold, resident float input, resident float zcs);
    kernel sub(in float input1, in float input2, out_like input1);
    alias sub_res as sub(resident float input1, resident float input2, resident float output);

};
```


### Explanation

```
Interface: zcs
Kernel: zcs1 untiled
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs2 untiled
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs3 untiled
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs4 untiled
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: sub untiled
  Param 0: float * input1
  Coerce 'input1' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float * input2
  Coerce 'input2' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_input1
  Allocate an array like 'input1' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs1_res untiled alias for 'zcs1'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float * zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: zcs2_res untiled alias for 'zcs2'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float * zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: zcs3_res untiled alias for 'zcs3'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float * zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: zcs4_res untiled alias for 'zcs4'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float * input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float * zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: sub_res untiled alias for 'sub'
  Param 0: float * input1
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float * input2
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float * output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32
```


### Stubs

```
// kernels zcs
__kernel void zcs1(int width_of_input, int height_of_input, float threshold, __global float *input, __global float *out_like_out_like_input) {
}
__kernel void zcs2(int width_of_input, int height_of_input, float threshold, __global float *input, __global float *out_like_out_like_input) {
}
__kernel void zcs3(int width_of_input, int height_of_input, float threshold, __global float *input, __global float *out_like_out_like_input) {
}
__kernel void zcs4(int width_of_input, int height_of_input, float threshold, __global float *input, __global float *out_like_out_like_input) {
}
__kernel void sub(__global float *input1, __global float *input2, __global float *out_like_out_like_input1) {
}
```

