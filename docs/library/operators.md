# Operators

Element-wise binary operations on GPU.

## Overview

The `Operators` class provides GPU-accelerated element-wise operations on arrays: add, subtract, multiply, etc.

## Class Reference

```python
from yapocis.lib.operators import Operators

ops = Operators()
```

## Available Operations

All operations follow the pattern `c = op(a, b)`:

| Operation | Method | Kernel |
|-----------|--------|--------|
| Addition | `add(a, b)` | `c = a + b` |
| Subtraction | `sub(a, b)` | `c = a - b` |
| Multiplication | `mul(a, b)` | `c = a * b` |
| Division | `div(a, b)` | `c = a / b` |

## Usage

```python
import numpy as np
from yapocis.lib.operators import Operators

a = np.random.rand(1024, 1024).astype(np.float32)
b = np.random.rand(1024, 1024).astype(np.float32)

ops = Operators()

# Add two arrays
c = ops.kmgr.add(a, b)

# Subtract
d = ops.kmgr.sub(a, b)
```

## Resident Mode

For repeated operations, use resident variants to avoid buffer transfers:

```python
ops = Operators()

# Upload once
ops.kmgr.add_res.write(a)
ops.kmgr.add_res.write(b)
ops.kmgr.add_res.write(c)

# Multiple operations
for i in range(10):
    ops.kmgr.sub_res(a, b, c, resident=(a, b, c))
    a, c = c, a  # Swap buffers

# Download once
ops.kmgr.add_res.read(c)
```

## Multi-Channel Support

Operations support 1, 2, 3, and 4 channel images:

```python {.no-exec}
# 1-channel (grayscale)
ops.kmgr.add(gray_a, gray_b)

# 4-channel (RGBA)
ops.kmgr.add4(rgba_a, rgba_b)
```

## In-Place Operations

Use `inout` variants for in-place modification:

```python {.no-exec}
# a = a + b (in-place)
ops.kmgr.add_inout(a, b)
```

## Example: Image Blending

```python
import numpy as np
from yapocis.lib.operators import Operators

def blend_images(img1, img2, alpha):
    """Blend two images: result = img1 * alpha + img2 * (1 - alpha)"""
    ops = Operators()

    # Scale images
    scaled1 = ops.kmgr.mul(img1, np.full_like(img1, alpha))
    scaled2 = ops.kmgr.mul(img2, np.full_like(img2, 1.0 - alpha))

    # Add together
    return ops.kmgr.add(scaled1, scaled2)
```

## Performance

For large arrays, GPU operators provide significant speedup:

| Size | NumPy | GPU | Speedup |
|------|-------|-----|---------|
| 1024x1024 | 2ms | 0.1ms | 20x |
| 4096x4096 | 30ms | 0.5ms | 60x |

## Underlying Kernel: operators

The `Operators` class uses the `operators` interface.

### Interface Definition

```
interface operators {
    # Generated from operators.yml template
    kernel add(in float input1, in float input2, out_like input1);
    extension res of add(resident float input1, resident float input2,
                         resident float output);

    kernel sub(in float input1, in float input2, out_like input1);
    extension res of sub(...);

    kernel mul(in float input1, in float input2, out_like input1);
    extension res of mul(...);

    kernel divop(in float input1, in float input2, out_like input1);
    extension res of divop(...);

    # Backward compatibility alias
    alias div as divop(...);
};
```

### Generated Operators

Operators are generated from `operators.yml`:

```yaml
operators:
  - [add, "+"]
  - [sub, "-"]
  - [mul, "*"]
  - [divop, "/"]
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input1` | `float*` | First operand |
| `input2` | `float*` | Second operand |
| `output` | `float*` | Result array |

### Direct Kernel Access

```python
from yapocis.core import load_program

# Load the operators interface
prog = load_program("operators")

# Call kernels directly (positional args)
c = prog.add(a, b)
c = prog.sub(a, b)
c = prog.mul(a, b)
c = prog.divop(a, b)  # Note: divop, not div

# Resident variants
prog.add_res(a, b, c)
prog.sub_res(a, b, c)
```

## See Also

- {doc}`gaussian` - Uses operators internally
- {doc}`gradient` - Edge computations
