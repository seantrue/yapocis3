# Kernels

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
Kernel: rgb2hsi untiled
  Param 0: float * r
  Coerce 'r' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float * g
  Coerce 'g' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * b
  Coerce 'b' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_r
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

  Param derived: float * out_like_r
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)

  Param derived: float * out_like_r
  Allocate an array like 'r' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 2)


Kernel: hsi2rgb untiled
  Param 0: float * h
  Coerce 'h' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 1: float * s
  Coerce 's' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param 2: float * i
  Coerce 'i' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_h
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)

  Param derived: float * out_like_h
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 1)

  Param derived: float * out_like_h
  Allocate an array like 'h' (opencl arg position=3, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 2)


Kernel: rgb2hsi3 untiled
  Param derived: int rgb
  Pass in as depth_of `rgb`
  Pass as parameter of type int32

  Param 0: float * rgb
  Coerce 'rgb' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_rgb
  Allocate an array like 'rgb' (opencl arg position=1, dtype=float32)
  After call, copy buffer from GPU into array
  Return as float32 (tuple position 0)


Kernel: hsi2rgb3 untiled
  Param derived: int hsi
  Pass in as depth_of `hsi`
  Pass as parameter of type int32

  Param 0: float * hsi
  Coerce 'hsi' to float32
  Before call, copy buffer from array to GPU
  Pass as parameter of type *float32

  Param derived: float * out_like_hsi
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

