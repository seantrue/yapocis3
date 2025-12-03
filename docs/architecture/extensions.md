# Extension Projects

Using YAPOCIS_PROJECT for extension projects.

## Overview

The `YAPOCIS_PROJECT` environment variable allows you to add custom kernels, interfaces, and assets without modifying the yapocis package. This is useful for:

- Proprietary kernel development
- Domain-specific extensions
- Testing new kernels before upstreaming
- Keeping custom assets separate

## Directory Structure

```
my_extension/
├── interfaces/
│   └── myprocessor.interface
├── kernels/
│   ├── ocl/
│   │   └── myprocessor.mako
│   ├── cuda/
│   │   └── myprocessor.mako
│   ├── metal/
│   │   └── myprocessor.mako
│   └── numbax/
│       └── myprocessor.py
├── images/
│   └── test_data.png
└── icons/
    └── app_icon.png
```

## Configuration

Set the environment variable:

```bash
export YAPOCIS_PROJECT=/path/to/my_extension
```

Or in Python:
```python {.no-exec}
import os
os.environ['YAPOCIS_PROJECT'] = '/path/to/my_extension'
```

## Asset Resolution Order

When loading resources, yapocis searches in this order:

1. **YAPOCIS_PROJECT** (if set)
2. **Current working directory**
3. **yapocis package directory**

```python {.no-exec}
from yapocis.core import load_program

# Searches:
# 1. $YAPOCIS_PROJECT/interfaces/myprocessor.interface
# 2. ./interfaces/myprocessor.interface
# 3. yapocis/interfaces/myprocessor.interface
prog = load_program("myprocessor")
```

## Creating an Extension

### 1. Define the Interface

```idl
# my_extension/interfaces/myprocessor.interface

interface myprocessor {
    meta {
        layout planar;
        class elementwise;
    }

    kernel process(
        width_of a, height_of a,
        inout float a,
        float parameter
    );

    extension res of process(
        width_of a, height_of a,
        resident float a,
        float parameter
    );
};
```

### 2. Create Kernel Templates

```mako
## my_extension/kernels/ocl/myprocessor.mako

<%inherit file="constants.mako"/>

__kernel void process(
    __global float *a,
    int width, int height,
    float parameter
) {
    int xx = get_global_id(0);
    int yy = get_global_id(1);

    if (xx >= width || yy >= height) return;

    int idx = ADDRESS(xx, yy);
    a[idx] = a[idx] * parameter;
}
```

### 3. Create Python Wrapper

```python {.no-exec}
# my_extension/lib/myprocessor.py

from yapocis.lib.bound_kernels import BoundKernels
import numpy as np

class MyProcessor(BoundKernels):
    INTERFACE = "myprocessor"

    def process(self, image, parameter=1.0):
        """Process image with custom kernel."""
        image = np.ascontiguousarray(image, dtype=np.float32)
        return self.kmgr.process(image, parameter)
```

### 4. Use the Extension

```python {.no-exec}
import os
os.environ['YAPOCIS_PROJECT'] = '/path/to/my_extension'

from my_extension.lib.myprocessor import MyProcessor

proc = MyProcessor()
result = proc.process(image, parameter=2.5)
```

## Testing Extensions

Create test files in your extension:

```python {.no-exec}
# my_extension/tests/test_myprocessor.py

import os
import numpy as np
import pytest

os.environ['YAPOCIS_PROJECT'] = os.path.dirname(os.path.dirname(__file__))

from my_extension.lib.myprocessor import MyProcessor

def test_process():
    image = np.ones((100, 100), dtype=np.float32)
    proc = MyProcessor()
    result = proc.process(image, parameter=2.0)
    assert np.allclose(result, 2.0)
```

## Sharing Extensions

To share your extension as a package:

```toml
# pyproject.toml

[project]
name = "yapocis-myextension"
dependencies = ["yapocis"]

[project.entry-points."yapocis.extensions"]
myextension = "yapocis_myextension:register"
```

```python {.no-exec}
# yapocis_myextension/__init__.py

import os

def register():
    """Register this extension's asset path."""
    return os.path.dirname(__file__)
```

## Multiple Extensions

Multiple extensions can be chained:

```bash
export YAPOCIS_PROJECT=/path/to/ext1:/path/to/ext2
```

Or use the API:

```python {.no-exec}
from yapocis.assets import add_asset_path

add_asset_path('/path/to/ext1')
add_asset_path('/path/to/ext2')
```

## Best Practices

1. **Mirror yapocis structure**: Keep the same directory layout
2. **Use unique names**: Avoid conflicts with built-in interfaces
3. **Include all backends**: Create kernels for each backend you support
4. **Test independently**: Ensure your extension works standalone
5. **Document requirements**: Note any additional dependencies
