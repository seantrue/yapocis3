# YAML Context Feature for Interface Templates

## Overview

This feature allows externalizing Mako template context from Python source code to YAML configuration files. This makes kernel and interface definitions more maintainable and easier to customize without modifying source code.

## Motivation

Previously, template context (like operator lists, kernel parameters, etc.) was hardcoded in Python library classes:

```python
class Operators(BoundKernels):
    OPERATORS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("divop", "/")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, operators=self.OPERATORS, **kwargs)
```

This approach has drawbacks:
- Context is scattered across multiple Python files
- Requires code changes to modify operator lists
- Harder to maintain consistency across backends
- Context is not self-documenting

## Solution: YAML Context Files

Template context can now be externalized to `<interface>.yml` files in the `interfaces/` directory:

```yaml
# yapocis/interfaces/operators.yml
operators:
  - [add, "+"]
  - [sub, "-"]
  - [mul, "*"]
  - [divop, "/"]
```

## How It Works

### 1. Context Loading

The `interface_context` module provides utilities to load YAML context:

```python
from yapocis.core.interface_context import load_interface_context

# Load YAML context for an interface
context = load_interface_context("operators")
# Returns: {'operators': [['add', '+'], ['sub', '-'], ['mul', '*'], ['divop', '/']]}
```

### 2. Automatic Merging

The `KernelManagerBase.load_program()` method automatically:
1. Loads YAML context from `<interface>.yml` if it exists
2. Merges with programmatically-passed context
3. Programmatic context takes precedence (allows overrides)

```python
# YAML context is loaded automatically
ops = Operators()  # Uses operators.yml

# Can override YAML context programmatically
ops = Operators(operators=[["add", "+"], ["sub", "-"]])  # Overrides YAML
```

### 3. Backward Compatibility

The system maintains full backward compatibility:
- If no YAML file exists, uses Python-defined context
- Existing code works without changes
- YAML files are optional

## Files Added

1. **yapocis/core/interface_context.py** - YAML context loading utilities
   - `load_interface_context(name)` - Load YAML for interface
   - `merge_context(yaml, **kwargs)` - Merge YAML with programmatic context
   - `get_interface_context(name, **kwargs)` - Complete context loading

2. **yapocis/interfaces/operators.yml** - Operators template context
   - Defines operator list: [name, symbol] pairs
   - Used by both interface and kernel templates

## Files Modified

1. **yapocis/core/kmgr_base.py**
   - Updated `load_program()` to load and merge YAML context
   - Added import for `interface_context` module

2. **yapocis/lib/operators.py**
   - Updated to use YAML context by default
   - Keeps `OPERATORS` as fallback for backward compatibility
   - Only passes context if not already provided

## Usage Patterns

### Pattern 1: Use YAML Context (Recommended)

```python
# Create operators.yml in interfaces directory
# Python code automatically uses it
from yapocis.lib.operators import Operators

ops = Operators()  # Automatically loads from operators.yml
result = ops.add(a, b)
```

### Pattern 2: Override YAML Context

```python
# Override YAML with custom operators
custom_ops = [["add", "+"], ["custom", "-"]]
ops = Operators(operators=custom_ops)
```

### Pattern 3: No YAML (Backward Compatible)

```python
# If operators.yml doesn't exist, uses Python fallback
ops = Operators()  # Uses Operators.OPERATORS class attribute
```

## Testing

### Test Results

All tests passing:
- ✅ YAML context loading: `load_interface_context("operators")`
- ✅ Operators with YAML: Addition/subtraction work correctly
- ✅ Context merging: Programmatic overrides YAML
- ✅ Backward compatibility: 39 existing operator tests pass

### Test Files

- `claude-debug/test_yaml_context.py` - Comprehensive YAML feature tests
- All existing operator tests continue to pass

## YAML File Format

### Structure

```yaml
# Template context variables
context_var_name:
  - [value1, value2]
  - [value3, value4]

another_var: "simple value"
```

### Operators Example

```yaml
# yapocis/interfaces/operators.yml
operators:
  - [add, "+"]      # Addition operator
  - [sub, "-"]      # Subtraction operator
  - [mul, "*"]      # Multiplication operator
  - [divop, "/"]    # Division operator
```

Each operator is a `[name, symbol]` pair where:
- `name` - Kernel function name (e.g., "add")
- `symbol` - C operator symbol (e.g., "+")

## Future Work

Other interfaces that could benefit from YAML externalization:

1. **convolvesep** - Kernel sizes and configurations
2. **gradient** - Filter types and parameters
3. **graphics/graphics4** - Drawing primitive configurations
4. **hsi/colormath** - Color space conversion parameters

## Benefits

1. **Maintainability**
   - Context is centralized and self-documenting
   - Easy to modify without touching Python code
   - Clear separation of configuration and logic

2. **Flexibility**
   - Can customize operators per deployment
   - Easy to add/remove operators
   - No need to rebuild Python packages

3. **Consistency**
   - Single source of truth for template context
   - Same context used by interface and kernel templates
   - Reduces duplication and errors

4. **Backward Compatibility**
   - Existing code works without changes
   - YAML files are optional
   - Gradual migration path

## Implementation Notes

- YAML parsing uses `yaml.safe_load()` for security
- Files must be valid YAML dictionaries
- Missing YAML files return empty context (no error)
- Programmatic context always takes precedence over YAML
- Context is loaded once and cached in compiled programs

## Conclusion

The YAML context feature successfully externalizes template configuration from Python source code while maintaining full backward compatibility. This makes the codebase more maintainable and flexible.

**Status**: ✅ Complete and tested
- Operators interface fully converted to YAML
- All tests passing (39 operator tests)
- Backward compatibility verified
- Ready for production use
