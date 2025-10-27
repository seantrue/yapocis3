# Proposal: global_size_from Directive

**Problem**: Yapocis infers `global_size` from the first array argument, but many GPU kernels need it based on the output array shape (e.g., pooling, fully connected layers).

**Current Workaround**: Manually pass `global_size` parameter:
```python
# Not ideal - requires manual calculation
prog.max_pool_2x2(input, output, global_size=(output.shape[1], output.shape[2], output.shape[0]))
```

---

## Proposed Solutions

### Option 1: Kernel-Level Trait (RECOMMENDED)

Add `dispatch_from` trait to specify which parameter drives global_size inference:

```idl
kernel max_pool_2x2(in float input, out float output,
                    height_of input, depth_of input, width_of input)
    with { dispatch_from output; };
```

**Pros**:
- Clean separation of concerns
- Trait system already exists in parser
- Clear and readable
- Can reference any parameter name

**Implementation**: Extend `trait_key` in parser:
```python
trait_key = oneOf("""class axis radius boundary tile halo fastmem taps identity op
                     deterministic accumulator_dtype texture_input vector_width blend cull
                     dispatch_from""")  # ADD THIS
```

Then in `kernel.py`:
```python
# In __call__ method
dispatch_param = self.traits.get('dispatch_from', None)
if dispatch_param and self.global_size is None:
    # Find array with this parameter name
    for i, arg in enumerate(args):
        if self.params[i].name == dispatch_param and isinstance(arg, np.ndarray):
            self.global_size = arg.shape
            break
```

---

### Option 2: Parameter Role Directive

Add `global_size_from` as a new parameter role (like `depth_of`, `sizeof`):

```idl
kernel max_pool_2x2(in float input, out float output, global_size_from output,
                    height_of input, depth_of input, width_of input);
```

**Pros**:
- Follows existing pattern for parameter directives
- Very explicit at parameter level

**Cons**:
- Somewhat redundant (references `output` twice)
- Adds another "derived" parameter that doesn't map to kernel arg
- Less intuitive than kernel trait

---

### Option 3: Stride/Transform Syntax

Allow expressing size transformations:

```idl
kernel max_pool_2x2(in float input, out float output, ...)
    with {
        dispatch_size (height_of input / 2, depth_of input / 2, width_of input);
    };
```

**Pros**:
- Very explicit about transformation
- Could support arbitrary expressions

**Cons**:
- Requires expression parser/evaluator
- Much more complex to implement
- Overkill for most use cases
- Error-prone (what if expression is wrong?)

---

## Recommendation: Option 1

**Use kernel-level trait `dispatch_from`**:

```idl
interface vgg16 {
    # Pooling - dispatch from output (half size of input)
    kernel max_pool_2x2(in float input, out float output,
                        height_of input, depth_of input, width_of input)
        with { dispatch_from output; };

    # Fully connected - dispatch from output (different size than input)
    kernel fully_connected(in float input, in float weights, in float bias, out float output,
                          sizeof input, sizeof output)
        with { dispatch_from output; };

    # Convolution - dispatch from output (same size as input, but explicit)
    kernel conv2d_3x3(in float input, in float weights, in float bias, out float output,
                      height_of input, depth_of input, width_of input, int output_channels)
        with { dispatch_from output; };

    # Activation functions - default dispatch from input (first array) is fine
    kernel relu_activation(in float input, out_like input, sizeof input);
}
```

---

## Implementation Checklist

### Parser Changes (`src/yapocis/core/interface_parser.py`)

1. Add `dispatch_from` to trait keywords:
```python
trait_key = oneOf("""class axis radius boundary tile halo fastmem taps identity op
                     deterministic accumulator_dtype texture_input vector_width blend cull
                     dispatch_from""")
```

2. Parse trait value as parameter name (already supported - traits take identifiers)

### Kernel Changes (`src/yapocis/metallic/kernel.py`, `src/yapocis/ocl/kernel.py`, etc.)

1. Store traits in Kernel object:
```python
def __init__(self, name, kernel_manager, kernel, params, traits=None, ...):
    self.traits = traits or {}
```

2. Modify `__call__` to use `dispatch_from`:
```python
def __call__(self, *args, **kwargs):
    # ... existing code ...

    if self.global_size is None:
        # Check for dispatch_from trait
        dispatch_param = self.traits.get('dispatch_from', None)

        if dispatch_param:
            # Find the specified parameter
            for i, param in enumerate(self.params):
                if param.name == dispatch_param and i < len(args):
                    arg = args[i]
                    if isinstance(arg, np.ndarray):
                        shape = arg.shape
                        if len(shape) > 1:
                            self.global_size = (shape[0] * shape[1], 1)
                        else:
                            self.global_size = shape
                        break
        else:
            # Default behavior: use first array
            shape = infer_wh_from_args(args)
            if shape:
                if len(shape) > 1:
                    self.global_size = (shape[0] * shape[1], 1)
                else:
                    self.global_size = shape
```

3. Pass traits from KernelManager to Kernel constructor

---

## Benefits

✅ **Cleaner API** - No manual global_size calculations
✅ **Interface-driven** - Behavior specified in interface, not Python code
✅ **Backward compatible** - Kernels without trait use default behavior
✅ **Clear intent** - Documents which parameter drives dispatch
✅ **Flexible** - Can reference any parameter name

---

## Example Before/After

### Before (Manual global_size)

```python
# Pooling
output = np.zeros((64, 112, 112), dtype=np.float32)
prog.max_pool_2x2(input, output, global_size=(112, 112, 64))

# Fully connected
output = np.zeros(1000, dtype=np.float32)
prog.fully_connected(input, weights, bias, output, global_size=(1000,))
```

### After (Automatic via trait)

```python
# Pooling - global_size auto-derived from output via dispatch_from
output = np.zeros((64, 112, 112), dtype=np.float32)
prog.max_pool_2x2(input, output)

# Fully connected - global_size auto-derived from output
output = np.zeros(1000, dtype=np.float32)
prog.fully_connected(input, weights, bias, output)
```

---

## Alternative: Default to Output for `out` Parameters?

Could also make yapocis **default** to using the first `out` parameter for global_size inference:

```python
def infer_wh_from_args(args, params):
    # Try to find first 'out' parameter
    for i, param in enumerate(params):
        if param.hint == 'out' and i < len(args):
            if isinstance(args[i], np.ndarray):
                return args[i].shape

    # Fall back to first array
    for arg in args:
        if isinstance(arg, np.ndarray):
            return arg.shape
    return None
```

This would work for most cases but isn't as flexible as the trait approach.

---

## Conclusion

**Recommend implementing Option 1** (`dispatch_from` trait) as it:
- Fits yapocis's existing design patterns
- Is explicit and clear
- Requires minimal parser changes
- Solves the global_size inference problem elegantly
