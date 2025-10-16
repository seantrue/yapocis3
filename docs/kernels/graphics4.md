# Kernels

## `graphics4`

### Signatures

```
# interface graphics4
def lines(a: Array[float32], lines: Array[float32], distance: Array[float32], rcolor: Array[float32]) -> None:
    pass
def rects(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def circles(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def triangles(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def ellipses(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def composite(a: Array[float32], lines: Array[int32], pixels: Array[float32], ipixels: Array[int32]) -> None:
    pass
def lines_res(a: Array[float32], lines: Array[float32], distance: Array[float32], rcolor: Array[float32]) -> None:
    pass
def rects_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def circles_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def triangles_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
    pass
def ellipses_res(a: Array[float32], lines: Array[float32], rcolor: Array[float32]) -> None:
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
Kernel: lines untiled
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * distance
  Coerce 'distance' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: rects untiled
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: circles untiled
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: triangles untiled
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: ellipses untiled
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: composite untiled
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Coerce 'a' to float32
  Before call, copy buffer from array to GPU
  After call, copy buffer from GPU into array

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: int * lines
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 2: float * pixels
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: int * ipixels
  Coerce 'ipixels' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param derived: int pixels
  Pass in as width_of `pixels`
  Pass as parameter of type int32


Kernel: lines_res untiled alias for 'lines'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * distance
  Coerce 'distance' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: rects_res untiled alias for 'rects'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: circles_res untiled alias for 'circles'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: triangles_res untiled alias for 'triangles'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: ellipses_res untiled alias for 'ellipses'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: float * lines
  Coerce 'lines' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * rcolor
  Coerce 'rcolor' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32


Kernel: composite_res untiled alias for 'composite'
  Param derived: int a
  Pass in as width_of `a`
  Pass as parameter of type int32

  Param derived: int a
  Pass in as height_of `a`
  Pass as parameter of type int32

  Param 0: float * a
  Buffer is resident on GPU
  Buffer already on GPU
  Pass as parameter of type *float32

  Param derived: int lines
  Pass in as width_of `lines`
  Pass as parameter of type int32

  Param 1: int * lines
  Coerce 'lines' to int32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *int32

  Param 2: float * pixels
  Coerce 'pixels' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 3: int * ipixels
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

