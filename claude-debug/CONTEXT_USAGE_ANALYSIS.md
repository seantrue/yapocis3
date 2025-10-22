# Kernel Interface Context Usage Analysis

## Overview

This document analyzes all kernel interfaces in yapocis to identify which ones use Mako template context variables and their externalization status.

## Methodology

1. Searched all `.interface` files for Mako templates (`%` directives)
2. Examined corresponding library classes for context being passed
3. Checked kernel `.mako` templates for context variable usage
4. Identified which contexts should be externalized to YAML

## Summary of Findings

### Interfaces Using Mako Context

| Interface | Context Variables | Library Class | Status | Notes |
|-----------|------------------|---------------|--------|-------|
| **operators** | `operators` | `Operators` | ✅ **Externalized** | Static list of operators |
| **convolvesep** | `convs` | `Gaussian` | ✅ **Externalized** | Gaussian generation params |
| **convolve** | `name`, `conv` | `Convolve1D` | ⚠️ **Not needed** | Utility class, not used internally |
| **convolves** | `convs` | None | ⚠️ **Unused** | No library class uses this |

### Interfaces NOT Using Context

These interfaces have no Mako template variables and don't need externalization:

- anisodiff
- colormath
- demo
- gradient
- graphics
- graphics4
- hsi
- imageops
- mandelbrot
- median3x3
- truetrace
- zcs

## Detailed Analysis

### 1. operators ✅ EXTERNALIZED

**Interface:** `yapocis/interfaces/operators.interface`
```
interface operators {
%for name, operator in operators:
    kernel ${name}(in float input1, in float input2, out_like input1);
    extension res of ${name}(resident float input1, resident float input2, resident float output);
%endfor
    alias div as divop(...);
    alias div_res as divop(...);
};
```

**Context:**
```python
operators = [
    ["add", "+"],
    ["sub", "-"],
    ["mul", "*"],
    ["divop", "/"]
]
```

**Library:** `yapocis/lib/operators.py`
```python
class Operators(BoundKernels):
    INTERFACE = "operators"
    # Context loaded from operators.yml
```

**YAML:** `yapocis/interfaces/operators.yml`
```yaml
operators:
  - [add, "+"]
  - [sub, "-"]
  - [mul, "*"]
  - [divop, "/"]
```

**Status:** ✅ **Complete** - Fully externalized, legacy fallback removed

---

### 2. convolvesep ✅ EXTERNALIZED

**Interface:** `yapocis/interfaces/convolvesep.interface`
```
interface convolvesep {
%for name,_ in convs:
    kernel ${name}(int vertical, width_of input, height_of input, in float input, out_like input);
    extension res of ${name}(...);
    kernel ${name}_h(...);
    kernel ${name}_v(...);
%endfor
};
```

**Context:**
```python
convs = [
    ("gauss9", np.array([...])),    # 9-element Gaussian kernel
    ("gauss13", np.array([...])),   # 13-element Gaussian kernel
    ...
]
```

**Library:** `yapocis/lib/gaussian.py`
```python
class Gaussian(BoundKernels):
    INTERFACE = "convolvesep"
    # Generates convs from GaussianWeights using YAML params
```

**YAML:** `yapocis/interfaces/convolvesep.yml`
```yaml
gaussian_params:
  max_width: 150
  scale: 1.6
  limit: 0.001
  sigmas: null
```

**Status:** ✅ **Complete** - Parameters externalized, legacy fallback removed

**Note:** Unlike operators, the `convs` list is generated at runtime from Gaussian parameters. The YAML externalizes the *generation parameters*, not the convs list itself.

---

### 3. convolve ⚠️ NOT NEEDED

**Interface:** `yapocis/interfaces/convolve.interface`
```
interface convolve {
    kernel ${name}(size_of a, in float a, out_like a);
    kernel ${name}_fast(size_of a, in float a, out_like a, int chunk);
};
```

**Context:**
```python
name = "some_kernel_name"  # e.g., "gauss9"
conv = np.array([...])     # Convolution weights
```

**Library:** `yapocis/lib/convolve.py`
```python
class Convolve1D(BoundKernels):
    INTERFACE = "convolve"

    def __init__(self, *args, name=None, conv=None, **kwargs):
        assert name and conv
        super(Convolve1D, self).__init__(*args, name=name, conv=conv, **kwargs)
```

**Usage:** No internal usage found. This is a utility class for users to create custom 1D convolution kernels.

**Status:** ⚠️ **Not needed** - This is a utility class meant to be used with programmatically-provided kernels, not predefined configurations.

**Recommendation:** Keep as-is. This class is designed for runtime-generated kernels, not configuration.

---

### 4. convolves ⚠️ UNUSED

**Interface:** `yapocis/interfaces/convolves.interface`
```
interface convolves {
%for name,conv in convs:
    kernel ${name}(sizeof a, in float a, outlike a);
%endfor
};
```

**Context:**
```python
convs = [
    ("kernel1", np.array([...])),
    ("kernel2", np.array([...])),
    ...
]
```

**Library:** None - No library class uses this interface

**Usage:** Not found in codebase

**Status:** ⚠️ **Unused** - Consider deprecating or documenting as user-facing utility

**Note:** Similar to `convolve` but supports multiple kernels. Appears to be unused legacy code.

---

## Interfaces Without Context

The following interfaces use no Mako template variables:

### gradient

**Interface:**
```
interface gradient {
    kernel gradient(width_of input, height_of input, in float input, int reach, out_like input, out_like input);
    extension res of gradient(...);
};
```

**Library:** `yapocis/lib/gradient.py`
```python
class Gradient(BoundKernels):
    INTERFACE = "gradient"
    # No context passed
```

**Parameters:** `reach` is a runtime parameter, not template variable

---

### graphics / graphics4

**Interface:** No Mako variables
**Library:** `yapocis/lib/graphics.py`, `yapocis/lib/graphics4.py`
**Context:** None needed - pure runtime parameters

---

### hsi

**Interface:** No Mako variables
**Library:** `yapocis/lib/hsi.py`
**Context:** None needed

---

### anisodiff

**Interface:** No Mako variables
**Library:** `yapocis/lib/anisodiff.py`
**Context:** None needed

---

### Other Interfaces

All remaining interfaces (colormath, demo, imageops, mandelbrot, median3x3, truetrace, zcs) use no Mako template context.

## Recommendations

### ✅ Already Complete

1. **operators** - Externalized to `operators.yml`
2. **convolvesep** - Externalized to `convolvesep.yml`

### ⚠️ No Action Needed

3. **convolve** - Keep as-is (utility class for runtime kernels)
4. **convolves** - Keep as-is or consider deprecating (unused)
5. **All other interfaces** - No context to externalize

## YAML Externalization Pattern

For interfaces that use Mako context, we follow this pattern:

### Pattern 1: Static Lists (operators)

**YAML Structure:**
```yaml
context_var_name:
  - [value1, value2]
  - [value3, value4]
```

**Usage:** Direct substitution in templates

### Pattern 2: Generation Parameters (convolvesep)

**YAML Structure:**
```yaml
param_group_name:
  param1: value1
  param2: value2
```

**Usage:** Parameters used to generate context at runtime

### Pattern 3: Utility Classes (convolve, convolves)

**No YAML:** Context provided programmatically by users

## Summary Statistics

- **Total interfaces:** 16
- **Using Mako context:** 4 (25%)
- **Externalized to YAML:** 2 (operators, convolvesep)
- **Utility classes (no externalization needed):** 2 (convolve, convolves)
- **No context used:** 12 (75%)

## Conclusion

The YAML externalization effort successfully identified and externalized all configuration-driven context:

✅ **operators** - Static operator definitions
✅ **convolvesep** - Gaussian generation parameters

The remaining context-using interfaces (convolve, convolves) are utility classes designed for programmatic use, not configuration, so externalization is not appropriate.

**All configuration-driven context has been successfully externalized to YAML.**
