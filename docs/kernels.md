# Kernels

## `zcs`

### Signatures

```
# interface zcs
def zcs1(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs1_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs2(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs2_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs3(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs3_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def zcs4(threshold: float32, input: Array[float32]) -> Array[float32]:
    pass
def zcs4_res(threshold: float32, input: Array[float32], zcs: Array[float32]) -> None:
    pass
def sub(input1: Array[float32], input2: Array[float32]) -> Array[float32]:
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
Kernel: zcs1
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs1_res alias for 'zcs1'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: zcs2
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs2_res alias for 'zcs2'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: zcs3
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs3_res alias for 'zcs3'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: zcs4
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: zcs4_res alias for 'zcs4'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float threshold

  Param 1: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float zcs
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: sub
  Param 0: float input1
  Coerce 'input1' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float input2
  Coerce 'input2' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input1
  Allocate an array like 'input1' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: sub_res alias for 'sub'
  Param 0: float input1
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float input2
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 2: float output
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


## `graphics4`

### Signatures

```
# interface graphics4
def lines(a: Array[float32], lines: Array[float32], distance: Array[float32], rcolor: Array[float32]) -> None:
    pass
def lines_res(a: Array[float32], lines: Array[float32], distance: Array[float32], rcolor: Array[float32]) -> None:
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


### Specification

```
interface graphics4 {
    # Kernels have extensiones for running on resident buffers
    kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, in float distance, in float rcolor);
    extension res as lines(width_of a, height_of a, resident float a, width_of lines, in float lines, in float distance, in float rcolor);
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


### Explanation

```
Interface: graphics4
Kernel: lines
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float distance
  Coerce 'distance' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: lines_res alias for 'lines'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float distance
  Coerce 'distance' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: rects
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: rects_res alias for 'rects'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: circles
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: circles_res alias for 'circles'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: triangles
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: triangles_res alias for 'triangles'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: ellipses
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: ellipses_res alias for 'ellipses'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: composite
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: int lines
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 2: float pixels
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: int ipixels
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param derived: int pixels
  Pass in as width_of `pixels`
  Pass as parameter of type int32


Kernel: composite_res alias for 'composite'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: int lines
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 2: float pixels
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: int ipixels
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param derived: int pixels
  Pass in as width_of `pixels`
  Pass as parameter of type int32
```


### Stubs

```
// kernels graphics4
__kernel void lines(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *distance, __global float *rcolor) {
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


## `colormath`

### Signatures

```
# interface colormath
def transform(input: Array[float32], transform_id: int32) -> Array[float32]:
    pass
def transform_res(input: Array[float32], transform_id: int32, output: Array[float32]) -> None:
    pass
def xyz2xyY(input: Array[float32]) -> Array[float32]:
    pass
def xyz2xyY_res(input: Array[float32], output: Array[float32]) -> None:
    pass
def xyY2xyz(input: Array[float32]) -> Array[float32]:
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
Kernel: transform
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: int transform_id

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: transform_res alias for 'transform'
  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: int transform_id

  Param 2: float output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: xyz2xyY
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: xyz2xyY_res alias for 'xyz2xyY'
  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: xyY2xyz
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: xyY2xyz_res alias for 'xyY2xyz'
  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float output
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


## `graphics`

### Signatures

```
# interface graphics
def lines(a: Array[float32], lines: Array[float32], distance: Array[float32], rcolor: Array[float32]) -> None:
    pass
def lines_res(a: Array[float32], lines: Array[float32], distance: Array[float32], rcolor: Array[float32]) -> None:
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


### Specification

```
interface graphics {
    # Kernels have aliases for running on resident buffers
    kernel lines(width_of a, height_of a, inout float a, width_of lines, in float lines, in float distance, in float rcolor);
    extension res of lines(width_of a, height_of a, resident float a, width_of lines, in float lines, in float distance, in float rcolor);
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


### Explanation

```
Interface: graphics
Kernel: lines
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float distance
  Coerce 'distance' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: lines_res alias for 'lines'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float distance
  Coerce 'distance' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: rects
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: rects_res alias for 'rects'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: circles
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: circles_res alias for 'circles'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: triangles
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: triangles_res alias for 'triangles'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: ellipses
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: ellipses_res alias for 'ellipses'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: composite
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: int lines
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 2: float pixels
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: int ipixels
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param derived: int pixels
  Pass in as width_of `pixels`
  Pass as parameter of type int32


Kernel: composite_res alias for 'composite'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: int lines
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 2: float pixels
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: int ipixels
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param derived: int pixels
  Pass in as width_of `pixels`
  Pass as parameter of type int32
```


### Stubs

```
// kernels graphics
__kernel void lines(int width_of_a, int height_of_a, __global float *a, int width_of_lines, __global float *lines, __global float *distance, __global float *rcolor) {
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


## `operators`
* operators.interface has unprocessed template markup.

```
interface operators {
%for name, operator in operators:
    kernel ${name}(in float input1, in float input2, out_like input1);
    extension res of ${name}(resident float input1, resident float input2, resident float output);
%endfor
};
```


## `imageops`

### Signatures

```
# interface imageops
def clip(input: Array[float32], low: float32, high: float32) -> Array[float32]:
    pass
def stretch(input: Array[float32], input_min: float32, input_max: float32, low: float32, high: float32) -> Array[float32]:
    pass
def binarize(input: Array[float32], threshold: float32) -> Array[float32]:
    pass
def histogram(input: Array[float32], n_bins: int32, low: float32, high: float32, n_workers: int32, partials: Array[int32]) -> Array[int32]:
    pass
```


### Specification

```
interface imageops {
    kernel clip(in float input, float low, float high, out_like input);
    kernel stretch(in float input, float input_min, float input_max, float low, float high, out_like input);
    kernel binarize(in float input, float threshold, out_like input);
    kernel histogram(in float input, size_of input, int n_bins, float low, float high, int n_workers, in int partials, out int histogram);
};
```


### Explanation

```
Interface: imageops
Kernel: clip
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float low

  Param 2: float high

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: stretch
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float input_min

  Param 2: float input_max

  Param 3: float low

  Param 4: float high

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=5, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: binarize
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float threshold

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: histogram
  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: int input
  Pass in as size_of `input`
  Pass as parameter of type int32

  Param 1: int n_bins

  Param 2: float low

  Param 3: float high

  Param 4: int n_workers

  Param 5: int partials
  Coerce 'partials' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 6: int histogram
  Allocate an array for histogram
  After call, copy buffer from GPU into array
  Return as int32 (tuple position 0)
```


### Stubs

```
// kernels imageops
__kernel void clip(__global float *input, float low, float high, __global float *out_like_out_like_input) {
}
__kernel void stretch(__global float *input, float input_min, float input_max, float low, float high, __global float *out_like_out_like_input) {
}
__kernel void binarize(__global float *input, float threshold, __global float *out_like_out_like_input) {
}
__kernel void histogram(__global float *input, int size_of_input, int n_bins, float low, float high, int n_workers, __global int *partials, __global int *histogram) {
}
```


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
Kernel: mandelbrot
  Param 0: complex64 q
  Coerce 'q' to complex64
  Before call, copy buffer from array to GPU
  Pass as parameter of type *complex64

  Param derived: short out_like_q
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


## `convolvesep`
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
Kernel: sum
  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float b
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_a
  Allocate an array like 'a' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: sum_res alias for 'sum'
  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float b
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float c
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
Kernel: median3x3
  Param derived: int a
  Pass in as size_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_a
  Allocate an array like 'a' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: median3x3_res alias for 'median3x3'
  Param derived: int a
  Pass in as size_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float ret
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
Kernel: gradient
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: int reach

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=2, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)


Kernel: gradient_res alias for 'gradient'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: int reach

  Param 2: float grad
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 3: float theta
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


## `convolves`
* convolves.interface has unprocessed template markup.

```
interface convolves {
%for name,conv in convs:
        kernel ${name}(sizeof a, in float a, outlike a);
%endfor 
};
```


## `convolve`
* convolve.interface has unprocessed template markup.

```
interface convolve {
    kernel ${name}(size_of a, in float a, out_like a);
};
```


## `anisodiff`

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
Kernel: filterImage
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage_res alias for 'filterImage'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param 3: float output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: filterImage2
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage2_res alias for 'filterImage2'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param 3: float output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: filterImage3
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage3_res alias for 'filterImage3'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param 3: float output
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32


Kernel: filterImage4
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Coerce 'input' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param derived: float out_like_input
  Allocate an array like 'input' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: filterImage4_res alias for 'filterImage4'
  Param derived: int input
  Pass in as width_of `input`
  Pass as parameter of type int32

  Param derived: int input
  Pass in as height_of `input`
  Pass as parameter of type int32

  Param 0: float input
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param 1: float scale

  Param 2: float size_step

  Param 3: float output
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


## `hsi`

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


### Specification

```
interface hsi {
    kernel rgb2hsi(in float r, in float g, in float b, out_like float r, out_like float r, out_like float r);
    kernel hsi2rgb(in float h, in float s, in float i, out_like h, out_like h, out_like h);
    kernel rgb2hsi3(depth_of rgb, in float rgb, out_like rgb);
    kernel hsi2rgb3(depth_of hsi, in float hsi, out_like hsi);
};
```


### Explanation

```
Interface: hsi
Kernel: rgb2hsi
  Param 0: float r
  Coerce 'r' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float g
  Coerce 'g' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float b
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_r
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

  Param derived: float out_like_r
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)

  Param derived: float out_like_r
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 2)


Kernel: hsi2rgb
  Param 0: float h
  Coerce 'h' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float s
  Coerce 's' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float i
  Coerce 'i' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_h
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

  Param derived: float out_like_h
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)

  Param derived: float out_like_h
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 2)


Kernel: rgb2hsi3
  Param derived: int rgb
  Pass in as depth_of `rgb`
  Pass as parameter of type int32

  Param 0: float rgb
  Coerce 'rgb' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_rgb
  Allocate an array like 'rgb' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: hsi2rgb3
  Param derived: int hsi
  Pass in as depth_of `hsi`
  Pass as parameter of type int32

  Param 0: float hsi
  Coerce 'hsi' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float out_like_hsi
  Allocate an array like 'hsi' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)
```


### Stubs

```
// kernels hsi
__kernel void rgb2hsi(__global float *r, __global float *g, __global float *b, __global float *out_like_out_like_r, __global float *out_like_out_like_r, __global float *out_like_out_like_r) {
}
__kernel void hsi2rgb(__global float *h, __global float *s, __global float *i, __global float *out_like_out_like_h, __global float *out_like_out_like_h, __global float *out_like_out_like_h) {
}
__kernel void rgb2hsi3(int depth_of_rgb, __global float *rgb, __global float *out_like_out_like_rgb) {
}
__kernel void hsi2rgb3(int depth_of_hsi, __global float *hsi, __global float *out_like_out_like_hsi) {
}
```

