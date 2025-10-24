# Design: Generic Context Generator System

## Problem Statement

Currently, interface template rendering has hardcoded special cases for weight generation:

```python
# In tools.py and package.py
if interface_name == 'convolvesep' and 'gaussian_params' in context:
    from yapocis.lib.gaussian import GaussianWeights
    # ... hardcoded logic

if interface_name == 'convolves':
    from yapocis.lib.convolve_kernels import generate_convs_from_spec
    # ... hardcoded logic
```

This violates the **Open/Closed Principle** - adding new template interfaces requires modifying core code.

## Requirements

1. **Generic**: Work for any interface without code changes
2. **Declarative**: Specify generators in YAML, not Python code
3. **Type-safe**: Use Python functions, not arbitrary code strings
4. **Backward compatible**: Existing interfaces continue to work
5. **Discoverable**: Easy to understand where generators come from
6. **Testable**: Generators can be tested independently

## Proposed Solutions

### Option 1: YAML Hook Metadata (Recommended)

Specify a Python callable in the YAML file using metadata:

```yaml
# convolvesep.yml
_generator:
  module: yapocis.lib.gaussian
  function: generate_convolvesep_convs

gaussian_params:
  max_width: 150
  scale: 1.6
  limit: 0.001
```

```python
# yapocis/lib/gaussian.py
def generate_convolvesep_convs(context: dict) -> dict:
    """Generate convs for convolvesep interface.

    Args:
        context: YAML context dict

    Returns:
        Modified context dict with 'convs' added
    """
    if 'gaussian_params' in context:
        gw = GaussianWeights(**context['gaussian_params'])
        gws = gw.gaussian_weights()
        context['convs'] = [("gauss%s" % len(a), a) for a in gws]
    return context
```

**Generic rendering code:**

```python
def render_interface_template(interface_name: str) -> str:
    template_text = load_interface(f"{interface_name}.interface")

    if '%' not in template_text and '${' not in template_text:
        return template_text

    context = load_interface_context(interface_name)
    if not context:
        return template_text

    # Generic generator invocation
    if '_generator' in context:
        context = invoke_generator(context)

    template = Template(template_text)
    return template.render(**context)

def invoke_generator(context: dict) -> dict:
    """Invoke generator function specified in context metadata."""
    gen_spec = context.pop('_generator')  # Remove metadata

    module_name = gen_spec['module']
    function_name = gen_spec['function']

    # Import and call
    module = importlib.import_module(module_name)
    generator_func = getattr(module, function_name)

    return generator_func(context)
```

**Pros:**
- ✅ Explicit and discoverable
- ✅ Type-safe (Python functions)
- ✅ Easy to test generators independently
- ✅ No code changes for new interfaces
- ✅ Clean separation of concerns

**Cons:**
- Requires metadata syntax in YAML
- Generator functions must be written

---

### Option 2: Convention-Based Discovery

Use naming convention `generate_<interface>_context`:

```python
def render_interface_template(interface_name: str) -> str:
    # ... load template and context ...

    # Try to find generator by convention
    generator = discover_generator(interface_name)
    if generator:
        context = generator(context)

    return Template(template_text).render(**context)

def discover_generator(interface_name: str):
    """Try to import generator by convention."""
    # Try yapocis.lib.<interface>:generate_<interface>_context
    try:
        module = importlib.import_module(f'yapocis.lib.{interface_name}')
        return getattr(module, f'generate_{interface_name}_context', None)
    except (ImportError, AttributeError):
        pass

    # Try yapocis.lib.<interface>_context:generate_context
    try:
        module = importlib.import_module(f'yapocis.lib.{interface_name}_context')
        return getattr(module, 'generate_context', None)
    except (ImportError, AttributeError):
        pass

    return None
```

**Pros:**
- ✅ No YAML changes needed
- ✅ Simple and implicit
- ✅ Backward compatible

**Cons:**
- ❌ Less discoverable (magic naming)
- ❌ Multiple import attempts (performance)
- ❌ Unclear where to put generators

---

### Option 3: Mako Preprocessing

Do generation inside the Mako template:

```mako
## convolvesep.interface
<%
# Import and generate convs
from yapocis.lib.gaussian import GaussianWeights

if 'gaussian_params' in context.kwargs:
    gw = GaussianWeights(**gaussian_params)
    convs = [("gauss%s" % len(a), a) for a in gw.gaussian_weights()]
else:
    convs = []
%>

interface convolvesep {
%for name,conv in convs:
    kernel ${name}(vertical, width_of input, height_of input, ...);
%endfor
};
```

**Pros:**
- ✅ Self-contained (logic with template)
- ✅ No separate generator files
- ✅ Immediate visibility

**Cons:**
- ❌ Mixes concerns (template + generation logic)
- ❌ Hard to test generation independently
- ❌ Can't reuse generators
- ❌ Makes templates harder to read

---

### Option 4: Hybrid Approach (YAML + Convention)

Combine explicit YAML hooks with convention-based fallback:

```python
def render_interface_template(interface_name: str) -> str:
    template_text = load_interface(f"{interface_name}.interface")

    if '%' not in template_text and '${' not in template_text:
        return template_text

    context = load_interface_context(interface_name)
    if not context:
        return template_text

    # 1. Try explicit generator from YAML
    if '_generator' in context:
        context = invoke_generator_from_yaml(context)

    # 2. Try convention-based generator
    elif (generator := discover_generator(interface_name)):
        context = generator(context)

    template = Template(template_text)
    return template.render(**context)
```

**Pros:**
- ✅ Flexible (explicit or implicit)
- ✅ Backward compatible
- ✅ Graceful degradation

**Cons:**
- More complex implementation
- Two ways to do the same thing

---

### Option 5: YAML Inline Generator (Simple)

For simple cases, allow inline Python in YAML:

```yaml
# convolves.yml
_generator_inline: |
  from yapocis.lib.convolve_kernels import generate_convs_from_spec
  convs = generate_convs_from_spec(context)
  context['convs'] = convs

windows:
  - type: hamming
    sizes: [5, 9]
```

**Pros:**
- ✅ Simple for small generators
- ✅ Everything in one file

**Cons:**
- ❌ Security risk (arbitrary code execution)
- ❌ Hard to test
- ❌ Syntax errors not caught early
- ❌ Not recommended for production

---

## Recommendation: Option 1 (YAML Hook Metadata)

**Implementation Plan:**

1. **Define generator interface:**
   ```python
   # Type definition
   ContextGenerator = Callable[[dict], dict]
   ```

2. **Add metadata to YAML:**
   ```yaml
   # convolvesep.yml
   _generator:
     module: yapocis.lib.gaussian
     function: generate_convolvesep_convs

   # convolves.yml
   _generator:
     module: yapocis.lib.convolve_kernels
     function: generate_convolves_convs
   ```

3. **Create generator functions:**
   ```python
   # yapocis/lib/gaussian.py
   def generate_convolvesep_convs(context: dict) -> dict:
       """Generate Gaussian convolution kernels."""
       # Move existing logic here

   # yapocis/lib/convolve_kernels.py
   def generate_convolves_convs(context: dict) -> dict:
       """Generate signal processing kernels."""
       # Rename generate_convs_from_spec to this
   ```

4. **Update rendering code:**
   ```python
   def render_interface_template(interface_name: str) -> str:
       # ... existing code ...

       # Replace special cases with generic invocation
       context = apply_context_generator(context)

       # ... rest of rendering ...
   ```

5. **Add tests:**
   - Test generator discovery
   - Test generator invocation
   - Test error handling (missing module, missing function)
   - Test backward compatibility

## Migration Path

1. Add `apply_context_generator()` function
2. Add `_generator` metadata to existing YAML files
3. Create named generator functions (refactor existing code)
4. Test that both old and new code paths work
5. Remove old special-case code
6. Update documentation

## Future Extensions

Once this is in place, we can easily add:

1. **Multiple generators**: Process context through a pipeline
   ```yaml
   _generators:
     - {module: foo, function: gen1}
     - {module: bar, function: gen2}
   ```

2. **Generator arguments**: Pass extra config to generators
   ```yaml
   _generator:
     module: yapocis.lib.gaussian
     function: generate_convs
     args:
       normalize: true
       dtype: float32
   ```

3. **Conditional generators**: Only run under certain conditions
   ```yaml
   _generator:
     module: yapocis.lib.gaussian
     function: generate_convs
     condition: gaussian_params  # Only if this key exists
   ```

## Example: Complete Implementation

See `claude-debug/context_generator_implementation.py` for a working prototype.
