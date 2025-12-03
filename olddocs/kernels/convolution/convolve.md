# Kernels

## `convolve`
* convolve.interface has unprocessed template markup.

```
interface convolve {
    kernel ${name}(size_of a, in float a, out_like a);
    kernel ${name}_fast(size_of a, in float a, out_like a, int chunk);
};
```

