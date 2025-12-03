# Your First Kernel

In this tutorial, you'll create a complete kernel that doubles the brightness of an image. This covers the full workflow:

1. Write an interface file
2. Write a kernel template
3. Use the kernel from Python

## Step 1: The Interface File

Create `double.interface` in `yapocis/interfaces/`:

```
interface double {
    kernel double(in float input, out_like float input);
};
```

This defines:
- An interface named `double`
- A kernel named `double` that takes:
  - `in float input` - an input array of floats
  - `out_like float input` - an output array with the same shape as input

## Step 2: The Kernel Template

Create `double.mako` in `yapocis/kernels/ocl/`:

```c
/* Kernel that doubles pixel values */

__kernel void double(
    __global const float *input,
    __global float *output)
{
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f;
}
```

This OpenCL kernel:
- Takes input and output buffers
- Gets the global work item ID
- Doubles the input value and writes to output

## Step 3: Using Your Kernel

```python {.no-exec}
import numpy as np
from yapocis.core import manager

# Create test data
input_data = np.random.rand(100).astype(np.float32)

# Load and run the kernel
prog = manager.load_program("double")
result = prog.double(input_data)

# Verify
print(f"Input max: {input_data.max():.2f}")
print(f"Output max: {result.max():.2f}")  # Should be ~2x input
```

## Understanding the Flow

1. **Load**: `load_program("double")` finds `double.interface`
2. **Parse**: The interface is parsed to understand parameters
3. **Compile**: The Mako template is rendered and compiled
4. **Execute**: When you call `prog.double(input_data)`:
   - Input is transferred to GPU
   - Kernel executes
   - Output is created and transferred back
5. **Return**: Result is returned as a NumPy array

## Making It Work for Images

Let's extend this for 2D images:

```
interface brightness {
    kernel double(width_of img, height_of img, in float img, out_like float img);
};
```

```c
/* Brightness adjustment kernel */

__kernel void double(
    int width,
    int height,
    __global const float *img,
    __global float *output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = fmin(img[idx] * 2.0f, 1.0f);  // Clamp to 1.0
    }
}
```

Key changes:
- `width_of img` and `height_of img` automatically extract dimensions
- 2D work items with `get_global_id(0)` and `get_global_id(1)`
- Bounds checking for edge cases

## Next Steps

- {doc}`interface-basics` - Learn all interface parameter types
- {doc}`template-basics` - Master Mako template features
- {doc}`testing-basics` - Test across all backends
