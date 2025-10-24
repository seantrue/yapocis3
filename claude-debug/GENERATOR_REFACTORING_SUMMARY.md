# Removing Hardcoded Weight Generation Special Cases

## Executive Summary

**Problem**: Interface template rendering has hardcoded special cases for `convolvesep` and `convolves` that violate the Open/Closed Principle.

**Solution**: Add `_generator` metadata to YAML files specifying a Python function to call for context transformation.

**Impact**:
- ✅ No code changes needed for new template interfaces
- ✅ Cleaner, more maintainable codebase
- ✅ Better separation of concerns
- ✅ Backward compatible with existing interfaces

## Current Implementation (Hardcoded)

### Code Location
Files with hardcoded special cases:
- `src/yapocis/scripts/tools.py` lines 191-204
- `src/yapocis/scripts/package.py` lines 77-105

### Current Code
```python
# tools.py - render_interface_template()

# Handle special case for convolvesep
if interface_name == 'convolvesep' and 'gaussian_params' in context:
    from yapocis.lib.gaussian import GaussianWeights
    gaussian_params = context['gaussian_params']
    gw = GaussianWeights(**gaussian_params)
    gws = gw.gaussian_weights()
    convs = [("gauss%s" % len(a), a) for a in gws]
    context['convs'] = convs

# Handle special case for convolves
if interface_name == 'convolves':
    from yapocis.lib.convolve_kernels import generate_convs_from_spec
    convs = generate_convs_from_spec(context)
    context['convs'] = convs
```

### Problems
1. **Violates Open/Closed Principle**: Must modify core code to add new interfaces
2. **Poor Separation of Concerns**: Business logic in rendering code
3. **Not Discoverable**: No indication in YAML that generators exist
4. **Hard to Test**: Can't test generators independently
5. **Code Duplication**: Same logic in `tools.py` and `package.py`

## Proposed Solution: YAML Generator Metadata

### New Implementation

#### YAML Files
```yaml
# convolvesep.yml
_generator:
  module: yapocis.lib.gaussian
  function: generate_convolvesep_context

gaussian_params:
  max_width: 150
  scale: 1.6
  limit: 0.001
```

```yaml
# convolves.yml
_generator:
  module: yapocis.lib.convolve_kernels
  function: generate_convolves_context

windows:
  - type: hamming
    sizes: [5, 9, 13]
```

#### Generator Functions
```python
# yapocis/lib/gaussian.py
def generate_convolvesep_context(context: dict) -> dict:
    """Generate Gaussian convolution kernels for convolvesep."""
    if 'gaussian_params' in context:
        gw = GaussianWeights(**context['gaussian_params'])
        gws = gw.gaussian_weights()
        context['convs'] = [("gauss%s" % len(a), a) for a in gws]
    return context
```

```python
# yapocis/lib/convolve_kernels.py
def generate_convolves_context(context: dict) -> dict:
    """Generate signal processing kernels for convolves."""
    # Rename existing function
    convs = generate_convs_from_spec(context)
    context['convs'] = convs
    return context
```

#### Generic Rendering Code
```python
# tools.py - NEW VERSION
def render_interface_template(interface_name: str) -> str:
    template_text = load_interface(f"{interface_name}.interface")

    if '%' not in template_text and '${' not in template_text:
        return template_text

    context = load_interface_context(interface_name)
    if not context:
        return template_text

    # Generic generator invocation (NEW!)
    context = apply_context_generator(context)

    template = Template(template_text)
    return template.render(**context)

def apply_context_generator(context: dict) -> dict:
    """Apply generator if specified in metadata."""
    if '_generator' not in context:
        return context

    gen_spec = context.pop('_generator')
    module = importlib.import_module(gen_spec['module'])
    generator = getattr(module, gen_spec['function'])

    return generator(context)
```

### Benefits

1. **Open/Closed Principle**: Add new interfaces without code changes
2. **Declarative**: Configuration in YAML, not Python
3. **Discoverable**: Generator location explicit in YAML
4. **Testable**: Generators are standalone functions
5. **Reusable**: One implementation for `tools.py` and `package.py`
6. **Extensible**: Can add features like:
   - Multiple generators in pipeline
   - Generator arguments
   - Conditional execution

## Implementation Plan

### Phase 1: Add Generic Infrastructure
1. Add `apply_context_generator()` to `core/interface_context.py`
2. Add unit tests for generator invocation
3. Add error handling (module not found, function not found)

### Phase 2: Add Generator Metadata to YAML
1. Add `_generator` to `convolvesep.yml`
2. Add `_generator` to `convolves.yml`
3. Add `_generator` to `operators.yml` (optional)

### Phase 3: Create Named Generator Functions
1. Create `generate_convolvesep_context()` in `lib/gaussian.py`
2. Rename `generate_convs_from_spec()` to `generate_convolves_context()` in `lib/convolve_kernels.py`
3. Update any internal callers

### Phase 4: Update Rendering Code
1. Replace special cases in `scripts/tools.py` with `apply_context_generator()`
2. Replace special cases in `scripts/package.py` with `apply_context_generator()`
3. Add tests verifying both paths work

### Phase 5: Cleanup
1. Remove old special-case code
2. Update documentation
3. Add migration guide for future interfaces

## Migration Example

### Before (Hardcoded)
```python
# To add a new template interface, must modify tools.py:

if interface_name == 'my_new_interface':
    from yapocis.lib.my_module import generate_stuff
    context['stuff'] = generate_stuff(context['params'])
```

### After (Generic)
```yaml
# my_new_interface.yml
_generator:
  module: yapocis.lib.my_module
  function: generate_my_new_interface_context

params:
  foo: bar
```

```python
# yapocis/lib/my_module.py
def generate_my_new_interface_context(context: dict) -> dict:
    context['stuff'] = generate_stuff(context['params'])
    return context
```

**No changes to core rendering code needed!**

## Testing Strategy

### Unit Tests
```python
def test_apply_context_generator():
    """Test generic generator invocation."""
    context = {
        '_generator': {
            'module': 'yapocis.lib.gaussian',
            'function': 'generate_convolvesep_context'
        },
        'gaussian_params': {'max_width': 50}
    }

    result = apply_context_generator(context)

    assert 'convs' in result
    assert '_generator' not in result  # Metadata removed

def test_missing_generator_module():
    """Test error handling for missing module."""
    context = {
        '_generator': {
            'module': 'nonexistent.module',
            'function': 'foo'
        }
    }

    with pytest.raises(ImportError):
        apply_context_generator(context)
```

### Integration Tests
```python
def test_convolvesep_rendering_with_generator():
    """Test that convolvesep interface renders with new system."""
    interface_text = render_interface_template('convolvesep')

    assert 'kernel gauss' in interface_text
    assert '${' not in interface_text  # Fully rendered

def test_backward_compatibility():
    """Test that interfaces without generators still work."""
    interface_text = render_interface_template('mandelbrot')

    assert 'interface mandelbrot' in interface_text
```

## Alternative Approaches Considered

### 1. Convention-Based Discovery
**Idea**: Auto-discover generator by naming convention

**Pros**: No YAML changes needed

**Cons**: Magic naming, unclear where generators live, multiple import attempts

### 2. Mako Preprocessing
**Idea**: Put generation logic in template

**Pros**: Self-contained

**Cons**: Mixes concerns, hard to test, can't reuse

### 3. Inline Python in YAML
**Idea**: Allow Python code strings in YAML

**Pros**: Everything in one file

**Cons**: Security risk, hard to test, syntax errors not caught early

**Decision**: Option 1 (YAML Metadata) is the best balance of explicit, type-safe, and flexible.

## Future Extensions

Once this is in place, we can add:

### Multiple Generators (Pipeline)
```yaml
_generators:
  - {module: yapocis.lib.foo, function: step1}
  - {module: yapocis.lib.bar, function: step2}
```

### Generator Arguments
```yaml
_generator:
  module: yapocis.lib.gaussian
  function: generate_convs
  args:
    normalize: true
    dtype: float32
```

### Conditional Generators
```yaml
_generator:
  module: yapocis.lib.gaussian
  function: generate_convs
  condition: gaussian_params  # Only run if this key exists
```

## Files to Review

- **Design Document**: `claude-debug/CONTEXT_GENERATOR_DESIGN.md`
- **Working Prototype**: `claude-debug/context_generator_implementation.py`
- **Example YAML Files**:
  - `claude-debug/convolvesep_new.yml`
  - `claude-debug/convolves_new.yml`

## Recommendation

**Implement Phase 1-3 immediately** to add the infrastructure and migrate `convolvesep` and `convolves`. This will:

1. Prove the concept works
2. Provide a template for future interfaces
3. Eliminate technical debt
4. Make the codebase more maintainable

**Estimated Effort**: 4-6 hours of development + testing

**Risk**: Low (backward compatible, well-tested prototype)

**Reward**: High (cleaner architecture, easier to extend)
