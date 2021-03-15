# Kernels

## ZCS

### Specification

```
interface zcs {
    kernel zcs1(width_of input, height_of input, in float input, out_like input);
    alias zcs1_res as zcs1(width_of input, height_of input, resident float input, resident float zcs);
    kernel zcs2(width_of input, height_of input, in float input, out_like input);
    alias zcs2_res as zcs2(width_of input, height_of input, resident float input, resident float zcs);
    kernel zcs3(width_of input, height_of input, in float input, out_like input);
    alias zcs3_res as zcs3(width_of input, height_of input, resident float input, resident float zcs);
    kernel zcs4(width_of input, height_of input, in float input, out_like input);
    alias zcs4_res as zcs4(width_of input, height_of input, resident float input, resident float zcs);
    kernel sub(in float input1, in float input2, out_like input1);
    alias sub_res as sub(resident float input1, resident float input2, resident float output);

};
```


### Signatures

```
# interface zcs
def zcs1(input: Array[float32]) -> Array[float32]:
    pass
def zcs1_res(input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs2(input: Array[float32]) -> Array[float32]:
    pass
def zcs2_res(input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs3(input: Array[float32]) -> Array[float32]:
    pass
def zcs3_res(input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs4(input: Array[float32]) -> Array[float32]:
    pass
def zcs4_res(input: Array[float32], zcs: Array[float32]) -> None:
    pass
def sub(input1: Array[float32], input2: Array[float32]) -> Array[float32]:
    pass
def sub_res(input1: Array[float32], input2: Array[float32], output: Array[float32]) -> None:
    pass
```


### Explanation

```
Interface: zcs
Kernel: zcs1 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs1_res alias for 'zcs1'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: zcs2 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs2_res alias for 'zcs2'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: zcs3 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs3_res alias for 'zcs3'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: zcs4 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs4_res alias for 'zcs4'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: sub alias for 'self'
Param 0: KernelParam(name='input1', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input1' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='input2', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input2' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='input1', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input1' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: sub_res alias for 'sub'
Param 0: KernelParam(name='input1', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='input2', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='output', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32
```


### Stubs

```
// kernels zcs
__kernel void zcs1(int width_of_input, int height_of_input, __global float *input, __global float *out_like_input) {
}
__kernel void zcs2(int width_of_input, int height_of_input, __global float *input, __global float *out_like_input) {
}
__kernel void zcs3(int width_of_input, int height_of_input, __global float *input, __global float *out_like_input) {
}
__kernel void zcs4(int width_of_input, int height_of_input, __global float *input, __global float *out_like_input) {
}
__kernel void sub(__global float *input1, __global float *input2, __global float *out_like_input1) {
}
```


## GRAPHICS4

### Specification

```
interface graphics4 {
    # Kernels have extensiones for running on resident buffers
    kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, float distance, in float rcolor);
    extension res as lines(width_of a, height_of a, resident float a, width_of lines, in float lines, float distance, in float rcolor);
    kernel rects(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res as rects(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel circles(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res as circles(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel triangles(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res as triangles(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel ellipses(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res as ellipses(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel composite(width_of a, height_of a, inout float a, width_of lines, in int lines, in float pixels, in int ipixels,  width_of pixels);
    extension res as composite(width_of a, height_of a, resident float a, width_of lines, in int lines, in float pixels, in int ipixels,  width_of pixels);
};
```


### Signatures

```
# interface graphics4
def lines(a: Array[float32], lines: Array[float32], distance: float32, rcolor: Array[float32]) -> None:
    pass
def lines_res(a: Array[float32], lines: Array[float32], distance: float32, rcolor: Array[float32]) -> None:
    pass
def rects(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def rects_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def circles(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def circles_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def triangles(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def triangles_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def ellipses(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def ellipses_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def composite(a: Array[float32], lines: Array[int32], pixels: Array[float32], ipixels: Array[int32]) -> None:
    pass
def composite_res(a: Array[float32], lines: Array[int32], pixels: Array[float32], ipixels: Array[int32]) -> None:
    pass
```


### Explanation

```
Interface: graphics4
Kernel: lines alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='distance', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: lines_res alias for 'lines'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='distance', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: rects alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: rects_res alias for 'rects'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: circles alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: circles_res alias for 'circles'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: triangles alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: triangles_res alias for 'triangles'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: ellipses alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: ellipses_res alias for 'ellipses'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: composite alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='pixels', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='ipixels', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='pixels', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `pixels`
  Use as parameter of type int32


Kernel: composite_res alias for 'composite'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='pixels', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='ipixels', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='pixels', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `pixels`
  Use as parameter of type int32
```


### Stubs

```
// kernels graphics4
__kernel void lines(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, float distance, __global float *rcolor) {
}
__kernel void rects(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void circles(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void triangles(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void ellipses(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void composite(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global int *lines, __global float *pixels, __global int *ipixels, int width_of_pixels) {
}
```


## GRAPHICS

### Specification

```
interface graphics {
    # Kernels have aliases for running on resident buffers
    kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, float distance, in float rcolor);
    extension res of lines(width_of a, height_of a, resident float a, width_of lines, in float lines, float distance, in float rcolor);
    kernel rects(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res of rects(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel circles(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res of circles(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel triangles(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res of triangles(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel ellipses(width_of a, height_of a, inout float a, width_of lines, in float lines, in float rcolor);
    extension res of ellipses(width_of a, height_of a, resident float a, width_of lines, in float lines, in float rcolor);
    kernel composite(width_of a, height_of a, inout float a, width_of lines, in int lines, in float pixels, in int ipixels,  width_of pixels);
    extension res of composite(width_of a, height_of a, resident float a, width_of lines, in int lines, in float pixels, in int ipixels,  width_of pixels);
};
```


### Signatures

```
# interface graphics
def lines(a: Array[float32], lines: Array[float32], distance: float32, rcolor: Array[float32]) -> None:
    pass
def lines_res(a: Array[float32], lines: Array[float32], distance: float32, rcolor: Array[float32]) -> None:
    pass
def rects(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def rects_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def circles(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def circles_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def triangles(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def triangles_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def ellipses(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def ellipses_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def composite(a: Array[float32], lines: Array[int32], pixels: Array[float32], ipixels: Array[int32]) -> None:
    pass
def composite_res(a: Array[float32], lines: Array[int32], pixels: Array[float32], ipixels: Array[int32]) -> None:
    pass
```


### Explanation

```
Interface: graphics
Kernel: lines alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='distance', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: lines_res alias for 'lines'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='distance', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: rects alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: rects_res alias for 'rects'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: circles alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: circles_res alias for 'circles'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: triangles alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: triangles_res alias for 'triangles'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: ellipses alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: ellipses_res alias for 'ellipses'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='rcolor', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32


Kernel: composite alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='pixels', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='ipixels', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='pixels', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `pixels`
  Use as parameter of type int32


Kernel: composite_res alias for 'composite'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='lines', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `lines`
  Use as parameter of type int32

Param 1: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='pixels', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='ipixels', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='pixels', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `pixels`
  Use as parameter of type int32
```


### Stubs

```
// kernels graphics
__kernel void lines(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, float distance, __global float *rcolor) {
}
__kernel void rects(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void circles(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void triangles(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void ellipses(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *rcolor) {
}
__kernel void composite(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global int *lines, __global float *pixels, __global int *ipixels, int width_of_pixels) {
}
```


## OPERATORS
* operators.interface has unprocessed template markup.

```
interface operators {
%for name, operator in operators:
    kernel ${name}(in float input1, in float input2, out_like input1);
    extension res of ${name}(resident float input1, resident float input2, resident float output);
%endfor
};
```


## MANDELBROT

### Specification

```
interface mandelbrot {
        kernel mandelbrot(in complex64 q, out_like short q, int maxiter);
};
```


### Signatures

```
# interface mandelbrot
def mandelbrot(q: Array[complex64], maxiter: int32) -> Array[int16]:
    pass
```


### Explanation

```
Interface: mandelbrot
Kernel: mandelbrot alias for 'self'
Param 0: KernelParam(name='q', hint='in', dim='', pointer='', data_type='complex64', np_dtype='complex64', derived=False)
  Coerce 'q' to complex64
  Before call, copy buffer from array to GPU
  Use as parameter of type *complex64

Param (derived): KernelParam(name='q', hint='out_like', dim='', pointer='', data_type='short', np_dtype='int16', derived=True)
  Allocate an array like 'q' (opencl arg position=1, dtype=int16)
  After call, copy buffer from GPU into array
  Return as int16 (tuple position 0)

Param 1: KernelParam(name='maxiter', hint='', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
```


### Stubs

```
// kernels mandelbrot
__kernel void mandelbrot(__global complex64 *q, __global short *out_like_q, int maxiter) {
}
```


## CONVOLVESEP
* convolvesep.interface has unprocessed template markup.

```
interface convolvesep {
%for name,_ in convs:
        kernel ${name}(int vertical, width_of input, height_of input, in float input, out_like input);
        extension res of ${name}(int vertical, width_of input, height_of input, resident float input, resident float output);
        kernel ${name}_h(width_of input, height_of input, resident float input, resident float output);
        kernel ${name}_v(width_of input, height_of input, resident float input, resident float output);
%endfor
    };
```


## DEMO

### Specification

```
interface demo {
    kernel sum(in float a, in float b, out_like a);
    extension res of sum(in float a, in float b, out float c);
    };
```


### Signatures

```
# interface demo
def sum(a: Array[float32], b: Array[float32]) -> Array[float32]:
    pass
def sum_res(a: Array[float32], b: Array[float32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: demo
Kernel: sum alias for 'self'
Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='b', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='a', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'a' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: sum_res alias for 'sum'
Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='b', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='c', hint='out', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Allocate an array for c
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels demo
__kernel void sum(__global float *a, __global float *b, __global float *out_like_a) {
}
```


## MEDIAN3X3

### Specification

```
interface median3x3 {
    kernel median3x3(size_of int a, height_of int a, in float a, out_like a );
    extension res of median3x3(size_of int a, height_of int a, in float a, out float ret );
};
```


### Signatures

```
# interface median3x3
def median3x3(a: Array[float32]) -> Array[float32]:
    pass
def median3x3_res(a: Array[float32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: median3x3
Kernel: median3x3 alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='size_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as size_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='a', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: median3x3_res alias for 'median3x3'
Param (derived): KernelParam(name='a', hint='', dim='size_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as size_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='ret', hint='out', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Allocate an array for ret
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels median3x3
__kernel void median3x3(int size_of_a, int height_of_a, __global float *a, __global float *out_like_a) {
}
```


## GRADIENT

### Specification

```
interface gradient {
    kernel gradient(width_of input, height_of input, in float input, int reach,  out_like input, out_like input );
    extension res of gradient(width_of input, height_of input, resident float input, int reach, resident float grad, resident float theta );
};
```


### Signatures

```
# interface gradient
def gradient(input: Array[float32], reach: int32) -> Array[float32], Array[float32]:
    pass
def gradient_res(input: Array[float32], reach: int32, grad: Array[float32], theta: Array[float32]) -> None:
    pass
```


### Explanation

```
Interface: gradient
Kernel: gradient alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='reach', hint='', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)


Kernel: gradient_res alias for 'gradient'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='reach', hint='', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)

Param 2: KernelParam(name='grad', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='theta', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32
```


### Stubs

```
// kernels gradient
__kernel void gradient(int width_of_input, int height_of_input, __global float *input, int reach, __global float *out_like_input, __global float *out_like_input) {
}
```


## CONVOLVES
* convolves.interface has unprocessed template markup.

```
interface convolves {
%for name,conv in convs:
        kernel ${name}(sizeof a, in float a, outlike a);
%endfor 
};
```


## CONVOLVE
* convolve.interface has unprocessed template markup.

```
interface convolve {
    kernel ${name}(size_of a, in float a, out_like a);
};
```


## ANISODIFF

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


### Signatures

```
# interface anisodiff
def filterImage(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
def filterImage2(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage2_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
def filterImage3(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage3_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
def filterImage4(input: Array[float32], scale: float32, size_step: float32) -> Array[float32]:
    pass
def filterImage4_res(input: Array[float32], scale: float32, size_step: float32, output: Array[float32]) -> None:
    pass
```


### Explanation

```
Interface: anisodiff
Kernel: filterImage alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage_res alias for 'filterImage'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='output', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: filterImage2 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage2_res alias for 'filterImage2'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='output', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: filterImage3 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage3_res alias for 'filterImage3'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='output', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32


Kernel: filterImage4 alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage4_res alias for 'filterImage4'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='scale', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 2: KernelParam(name='size_step', hint='', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)

Param 3: KernelParam(name='output', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32
```


### Stubs

```
// kernels anisodiff
__kernel void filterImage(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_input) {
}
__kernel void filterImage2(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_input) {
}
__kernel void filterImage3(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_input) {
}
__kernel void filterImage4(int width_of_input, int height_of_input, __global float *input, float scale, float size_step, __global float *out_like_input) {
}
```


## HSI

### Specification

```
interface hsi {
    kernel rgb2hsi(in float r, in float g, in float b, out_like float r, out_like float r, out_like float r);
    kernel hsi2rgb(in float h, in float s, in float i, out_like h, out_like h, out_like h);
    kernel rgb2hsi3(depth_of rgb, in float rgb, out_like rgb);
    kernel hsi2rgb3(depth_of hsi, in float hsi, out_like hsi);
};
```


### Signatures

```
# interface hsi
def rgb2hsi(r: Array[float32], g: Array[float32], b: Array[float32]) -> Array[float32], Array[float32], Array[float32]:
    pass
def hsi2rgb(h: Array[float32], s: Array[float32], i: Array[float32]) -> Array[float32], Array[float32], Array[float32]:
    pass
def rgb2hsi3(rgb: Array[float32]) -> Array[float32]:
    pass
def hsi2rgb3(hsi: Array[float32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: hsi
Kernel: rgb2hsi alias for 'self'
Param 0: KernelParam(name='r', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'r' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='g', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'g' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='b', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='r', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

Param (derived): KernelParam(name='r', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)

Param (derived): KernelParam(name='r', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 2)


Kernel: hsi2rgb alias for 'self'
Param 0: KernelParam(name='h', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'h' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='s', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 's' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='i', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'i' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='h', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

Param (derived): KernelParam(name='h', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)

Param (derived): KernelParam(name='h', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 2)


Kernel: rgb2hsi3 alias for 'self'
Param (derived): KernelParam(name='rgb', hint='', dim='depth_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as depth_of `rgb`
  Use as parameter of type int32

Param 0: KernelParam(name='rgb', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'rgb' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='rgb', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'rgb' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: hsi2rgb3 alias for 'self'
Param (derived): KernelParam(name='hsi', hint='', dim='depth_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as depth_of `hsi`
  Use as parameter of type int32

Param 0: KernelParam(name='hsi', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'hsi' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='hsi', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'hsi' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels hsi
__kernel void rgb2hsi(__global float *r, __global float *g, __global float *b, __global float *out_like_r, __global float *out_like_r, __global float *out_like_r) {
}
__kernel void hsi2rgb(__global float *h, __global float *s, __global float *i, __global float *out_like_h, __global float *out_like_h, __global float *out_like_h) {
}
__kernel void rgb2hsi3(int depth_of_rgb, __global float *rgb, __global float *out_like_rgb) {
}
__kernel void hsi2rgb3(int depth_of_hsi, __global float *hsi, __global float *out_like_hsi) {
}
```


## BOUNDEDAVERAGE

### Specification

```
interface boundedaverage {
     kernel boundedaverage(width_of input, height_of input, in float input, in int zcs, out_like input);
     kernel boundedaverageangle(width_of input, height_of input, in float input, in int zcs, out_like input);
     kernel boundedaverageanglesat(width_of input, height_of input, in float input, in float sat, in int zcs, out_like input);
  };
```


### Signatures

```
# interface boundedaverage
def boundedaverage(input: Array[float32], zcs: Array[int32]) -> Array[float32]:
    pass
def boundedaverageangle(input: Array[float32], zcs: Array[int32]) -> Array[float32]:
    pass
def boundedaverageanglesat(input: Array[float32], sat: Array[float32], zcs: Array[int32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: boundedaverage
Kernel: boundedaverage alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='zcs', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'zcs' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: boundedaverageangle alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='zcs', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'zcs' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: boundedaverageanglesat alias for 'self'
Param (derived): KernelParam(name='input', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `input`
  Use as parameter of type int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 0: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='sat', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'sat' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='zcs', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'zcs' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels boundedaverage
__kernel void boundedaverage(int width_of_input, int height_of_input, __global float *input, __global int *zcs, __global float *out_like_input) {
}
__kernel void boundedaverageangle(int width_of_input, int height_of_input, __global float *input, __global int *zcs, __global float *out_like_input) {
}
__kernel void boundedaverageanglesat(int width_of_input, int height_of_input, __global float *input, __global float *sat, __global int *zcs, __global float *out_like_input) {
}
```


## DURER

### Specification

```
interface durer {
    kernel durer(in int seeds, width_of grad, height_of grad, in float grad, in float theta, inout float a, in float color, in int strokelength);
 };
```


### Signatures

```
# interface durer
def durer(seeds: Array[int32], grad: Array[float32], theta: Array[float32], a: Array[float32], color: Array[float32], strokelength: Array[int32]) -> None:
    pass
```


### Explanation

```
Interface: durer
Kernel: durer alias for 'self'
Param 0: KernelParam(name='seeds', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'seeds' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='grad', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `grad`
  Use as parameter of type int32

Param (derived): KernelParam(name='grad', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `grad`
  Use as parameter of type int32

Param 1: KernelParam(name='grad', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'grad' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='theta', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'theta' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param 4: KernelParam(name='color', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'color' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 5: KernelParam(name='strokelength', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'strokelength' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32
```


### Stubs

```
// kernels durer
__kernel void durer(__global int *seeds, int width_of_grad, int height_of_grad, __global float *grad, __global float *theta, __global float *a, __global float *color, __global int *strokelength) {
}
```


## BOUNDEDMEDIAN

### Specification

```
interface boundedmedian {
     kernel boundedmedian(in int vertical, height_of input, in float input, in int zcs, out_like input);
     alias boundedmedian_res as boundedmedian(in int vertical, height_of input, resident float input, resident int zcs, resident int output);
  };
```


### Signatures

```
# interface boundedmedian
def boundedmedian(vertical: Array[int32], input: Array[float32], zcs: Array[int32]) -> Array[float32]:
    pass
def boundedmedian_res(vertical: Array[int32], input: Array[float32], zcs: Array[int32], output: Array[int32]) -> None:
    pass
```


### Explanation

```
Interface: boundedmedian
Kernel: boundedmedian alias for 'self'
Param 0: KernelParam(name='vertical', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'vertical' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 1: KernelParam(name='input', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='zcs', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'zcs' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='input', hint='out_like', dim='', pointer='', data_type='float', np_dtype='float32', derived=True)
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: boundedmedian_res alias for 'boundedmedian'
Param 0: KernelParam(name='vertical', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'vertical' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='input', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `input`
  Use as parameter of type int32

Param 1: KernelParam(name='input', hint='resident', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32

Param 3: KernelParam(name='output', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32
```


### Stubs

```
// kernels boundedmedian
__kernel void boundedmedian(__global int *vertical, int height_of_input, __global float *input, __global int *zcs, __global float *out_like_input) {
}
```


## DRAW

### Specification

```
interface draw {
   kernel draw(in int lines, width_of int a, height_of int a, inout float a, in float color);
   };
```


### Signatures

```
# interface draw
def draw(lines: Array[int32], a: Array[float32], color: Array[float32]) -> None:
    pass
```


### Explanation

```
Interface: draw
Kernel: draw alias for 'self'
Param 0: KernelParam(name='lines', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 1: KernelParam(name='a', hint='inout', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

Param 2: KernelParam(name='color', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'color' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32
```


### Stubs

```
// kernels draw
__kernel void draw(__global int *lines, int width_of_a, int height_of_a, __global float *a, __global float *color) {
}
```


## AVERAGESEGMENTS

### Specification

```
interface averagesegments {
    kernel averagesegments(in int segments,  width_of img,  height_of img, in float maxval, in float img, out float output);
    };
```


### Signatures

```
# interface averagesegments
def averagesegments(segments: Array[int32], maxval: Array[float32], img: Array[float32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: averagesegments
Kernel: averagesegments alias for 'self'
Param 0: KernelParam(name='segments', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'segments' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='img', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `img`
  Use as parameter of type int32

Param (derived): KernelParam(name='img', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `img`
  Use as parameter of type int32

Param 1: KernelParam(name='maxval', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'maxval' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='img', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'img' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='output', hint='out', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Allocate an array for output
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels averagesegments
__kernel void averagesegments(__global int *segments, int width_of_img, int height_of_img, __global float *maxval, __global float *img, __global float *output) {
}
```


## PALLETTE

### Specification

```
interface pallette {
    kernel applypallette(in float a, outlike a);
    kernel applypallette2(in float a, outlike a);
};
```


### Signatures

```
# interface pallette
def applypallette(a: Array[float32]) -> Array[]:
    pass
def applypallette2(a: Array[float32]) -> Array[]:
    pass
```


### Explanation

```
Interface: pallette
Kernel: applypallette alias for 'self'
Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='a', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 0)


Kernel: applypallette2 alias for 'self'
Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='a', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 0)
```


### Stubs

```
// kernels pallette
__kernel void applypallette(__global float *a, __global  *a) {
}
__kernel void applypallette2(__global float *a, __global  *a) {
}
```


## IHSL

### Specification

```
interface ihsl {
    kernel RGBToIHSL(in float r, in float g, in float b, outlike r, outlike g, outlike b);
    kernel IHSLToRGB(in float h, in float s, in float hl, outlike h, outlike s, outlike l);
};
```


### Signatures

```
# interface ihsl
def RGBToIHSL(r: Array[float32], g: Array[float32], b: Array[float32]) -> Array[], Array[], Array[]:
    pass
def IHSLToRGB(h: Array[float32], s: Array[float32], hl: Array[float32]) -> Array[], Array[], Array[]:
    pass
```


### Explanation

```
Interface: ihsl
Kernel: RGBToIHSL alias for 'self'
Param 0: KernelParam(name='r', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'r' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='g', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'g' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='b', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='r', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'r' (opencl arg position=3, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 0)

Param (derived): KernelParam(name='g', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'g' (opencl arg position=3, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 1)

Param (derived): KernelParam(name='b', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'b' (opencl arg position=3, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 2)


Kernel: IHSLToRGB alias for 'self'
Param 0: KernelParam(name='h', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'h' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 1: KernelParam(name='s', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 's' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 2: KernelParam(name='hl', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'hl' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='h', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'h' (opencl arg position=3, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 0)

Param (derived): KernelParam(name='s', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 's' (opencl arg position=3, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 1)

Param (derived): KernelParam(name='l', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'l' (opencl arg position=3, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 2)
```


### Stubs

```
// kernels ihsl
__kernel void RGBToIHSL(__global float *r, __global float *g, __global float *b, __global  *r, __global  *g, __global  *b) {
}
__kernel void IHSLToRGB(__global float *h, __global float *s, __global float *hl, __global  *h, __global  *s, __global  *l) {
}
```


## XY

### Specification

```
interface xy {
    kernel addr(width_of a, height_of a, in int a, outlike a, outlike a, outlike a, outlike a);
};
```


### Signatures

```
# interface xy
def addr(a: Array[int32]) -> Array[], Array[], Array[], Array[]:
    pass
```


### Explanation

```
Interface: xy
Kernel: addr alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'a' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='a', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 0)

Param (derived): KernelParam(name='a', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 1)

Param (derived): KernelParam(name='a', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 2)

Param (derived): KernelParam(name='a', hint='outlike', dim='', pointer='', data_type='', np_dtype='', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=)
  After call, copy buffer from GPU into array
  Return as  (tuple position 3)
```


### Stubs

```
// kernels xy
__kernel void addr(int width_of_a, int height_of_a, __global int *a, __global  *a, __global  *a, __global  *a, __global  *a) {
}
```


## LABEL

### Specification

```
interface label {
    kernel label(width_of int a, height_of int a, in int a, in int zcs, out int ret, inout int flag);
    alias first as label(width_of int a, height_of int a, in int a, in int zcs, in int ret, inout int flag);
    alias step as label(width_of int a, height_of int a, resident int a, resident int zcs, resident int ret, inout int flag);
    alias last as label(width_of int a, height_of int a, resident int a, resident int zcs, out int ret, inout int flag);
    kernel hollow(width_of int a, height_of int a, in int a, out_like a);
};
```


### Signatures

```
# interface label
def label(a: Array[int32], zcs: Array[int32], flag: Array[int32]) -> Array[int32]:
    pass
def first(a: Array[int32], zcs: Array[int32], ret: Array[int32], flag: Array[int32]) -> None:
    pass
def step(a: Array[int32], zcs: Array[int32], ret: Array[int32], flag: Array[int32]) -> None:
    pass
def last(a: Array[int32], zcs: Array[int32], flag: Array[int32]) -> Array[int32]:
    pass
def hollow(a: Array[int32]) -> Array[int32]:
    pass
```


### Explanation

```
Interface: label
Kernel: label alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'a' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 1: KernelParam(name='zcs', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'zcs' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='ret', hint='out', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Allocate an array for ret
  After call, copy buffer from GPU into array
  Return as int32 (tuple position 0)

Param 3: KernelParam(name='flag', hint='inout', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'flag' to int32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array


Kernel: first alias for 'label'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'a' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 1: KernelParam(name='zcs', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'zcs' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='ret', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'ret' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 3: KernelParam(name='flag', hint='inout', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'flag' to int32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array


Kernel: step alias for 'label'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32

Param 1: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='ret', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32

Param 3: KernelParam(name='flag', hint='inout', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'flag' to int32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array


Kernel: last alias for 'label'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32

Param 1: KernelParam(name='zcs', hint='resident', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Buffer is resident on GPU
  Buffer already on GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='ret', hint='out', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Allocate an array for ret
  After call, copy buffer from GPU into array
  Return as int32 (tuple position 0)

Param 3: KernelParam(name='flag', hint='inout', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'flag' to int32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array


Kernel: hollow alias for 'self'
Param (derived): KernelParam(name='a', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `a`
  Use as parameter of type int32

Param (derived): KernelParam(name='a', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `a`
  Use as parameter of type int32

Param 0: KernelParam(name='a', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'a' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param (derived): KernelParam(name='a', hint='out_like', dim='', pointer='', data_type='int', np_dtype='int32', derived=True)
  Allocate an array like 'a' (opencl arg position=1, dtype=int32)
  After call, copy buffer from GPU into array
  Return as int32 (tuple position 0)
```


### Stubs

```
// kernels label
__kernel void label(int width_of_a, int height_of_a, __global int *a, __global int *zcs, __global int *ret, __global int *flag) {
}
__kernel void hollow(int width_of_a, int height_of_a, __global int *a, __global int *out_like_a) {
}
```


## APPLYSEGMENTS

### Specification

```
interface applysegments {
    kernel applysegments(in int segments,  in int width,  in int height, in float maxval, in float img, out float output);
    };
```


### Signatures

```
# interface applysegments
def applysegments(segments: Array[int32], width: Array[int32], height: Array[int32], maxval: Array[float32], img: Array[float32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: applysegments
Kernel: applysegments alias for 'self'
Param 0: KernelParam(name='segments', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'segments' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 1: KernelParam(name='width', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'width' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 2: KernelParam(name='height', hint='in', dim='', pointer='', data_type='int', np_dtype='int32', derived=False)
  Coerce 'height' to int32
  Before call, copy buffer from array to GPU
  Use as parameter of type *int32

Param 3: KernelParam(name='maxval', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'maxval' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 4: KernelParam(name='img', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'img' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 5: KernelParam(name='output', hint='out', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Allocate an array for output
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels applysegments
__kernel void applysegments(__global int *segments, __global int *width, __global int *height, __global float *maxval, __global float *img, __global float *output) {
}
```


## ROTATE

### Specification

```
interface rotate {
    kernel rotateImage(width_of src, height_of src, in float src, width_of dst, height_of dst, out float dst, in float sin, in float cos);
};
```


### Signatures

```
# interface rotate
def rotateImage(src: Array[float32], sin: Array[float32], cos: Array[float32]) -> Array[float32]:
    pass
```


### Explanation

```
Interface: rotate
Kernel: rotateImage alias for 'self'
Param (derived): KernelParam(name='src', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `src`
  Use as parameter of type int32

Param (derived): KernelParam(name='src', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `src`
  Use as parameter of type int32

Param 0: KernelParam(name='src', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'src' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param (derived): KernelParam(name='dst', hint='', dim='width_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as width_of `dst`
  Use as parameter of type int32

Param (derived): KernelParam(name='dst', hint='', dim='height_of', pointer='', data_type='int', np_dtype='int32', derived=True)
  Pass in as height_of `dst`
  Use as parameter of type int32

Param 1: KernelParam(name='dst', hint='out', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Allocate an array for dst
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

Param 2: KernelParam(name='sin', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'sin' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32

Param 3: KernelParam(name='cos', hint='in', dim='', pointer='', data_type='float', np_dtype='float32', derived=False)
  Coerce 'cos' to float32
  Before call, copy buffer from array to GPU
  Use as parameter of type *float32
```


### Stubs

```
// kernels rotate
__kernel void rotateImage(int width_of_src, int height_of_src, __global float *src, int width_of_dst, int height_of_dst, __global float *dst, __global float *sin, __global float *cos) {
}
```

