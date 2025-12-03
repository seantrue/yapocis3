# Assets API

API reference for yapocis asset management modules.

## Overview

Yapocis provides several asset registries for managing resources like fonts, icons, images, interfaces, and kernel templates. Each registry uses the common `AssetManager` class.

## Asset Registries

### Fonts Module

```{eval-rst}
.. automodule:: yapocis.fonts
   :members:
   :undoc-members:
   :show-inheritance:
```

### Icons Module

```{eval-rst}
.. automodule:: yapocis.icons
   :members:
   :undoc-members:
   :show-inheritance:
```

### Images Module

```{eval-rst}
.. automodule:: yapocis.images
   :members:
   :undoc-members:
   :show-inheritance:
```

### Interfaces Module

```{eval-rst}
.. automodule:: yapocis.interfaces
   :members:
   :undoc-members:
   :show-inheritance:
```

### Kernels Module

```{eval-rst}
.. automodule:: yapocis.kernels
   :members:
   :undoc-members:
   :show-inheritance:
```

## Asset Manager

The `AssetManager` class provides the core functionality for all asset registries.

```{eval-rst}
.. autoclass:: yapocis.utils.internals.AssetManager
   :members:
   :undoc-members:
   :show-inheritance:
```

## Path Discovery

Utility functions for discovering yapocis package resources.

```{eval-rst}
.. autofunction:: yapocis.utils.internals.get_yapocis_dir

.. autofunction:: yapocis.utils.internals.get_yapocis_dirs

.. autofunction:: yapocis.utils.internals.get_project_dir

.. autofunction:: yapocis.utils.internals.find_yapocis_file
```

## Quick Reference

### Finding Assets

```python
from yapocis.fonts import font, fonts, image_font
from yapocis.icons import icon, find_icon, icons
from yapocis.images import find_image, images
from yapocis.interfaces import find_interface, interfaces
from yapocis.kernels import find_kernel, kernels

# List available assets
print(fonts())        # ['Ubuntu-Regular.ttf', ...]
print(icons())        # ['default.png', ...]
print(images())       # ['test.jpg', 'lena.png', ...]
print(interfaces())   # ['mandelbrot.interface', ...]
print(kernels())      # ['anisodiff', 'colormath', ...]

# Find specific assets
font_path = font("Ubuntu")              # Partial match
icon_path = find_icon("settings")       # Partial match
image_path = find_image("test.jpg")     # Exact name match
interface_path = find_interface("mandelbrot")  # Name match
kernel_path = find_kernel("gradient")   # Name match
```

### Loading Fonts

```python
from yapocis.fonts import font, image_font

# Get font file path
path = font("Ubuntu")  # Returns path to Ubuntu-Regular.ttf

# Get PIL ImageFont object (cached)
fnt = image_font("Ubuntu", size=24)
```

### Loading Icons

```python
from yapocis.icons import icon, find_icon

# Get icon as numpy array (resized to size)
img = icon("settings", size=64)

# Get icon file path
path = find_icon("settings")
```

### Match Modes

Asset registries support different matching modes:

| Mode | Description | Example |
|------|-------------|---------|
| `exact` | Exact filename match | `"test.jpg"` matches `"test.jpg"` only |
| `name` | Filename without extension | `"test"` matches `"test.jpg"` |
| `lower` | Case-insensitive partial | `"ubuntu"` matches `"Ubuntu-Regular.ttf"` |
| `partial` | Substring match | `"settings"` matches `"icon-settings-64.png"` |

### Search Paths

Assets are searched in multiple directories:

1. Yapocis package directory (e.g., `site-packages/yapocis/fonts/`)
2. Current working directory (e.g., `./fonts/`)
3. Project directory via `YAPOCIS_PROJECT` environment variable

```python
import os
os.environ["YAPOCIS_PROJECT"] = "/path/to/my/assets"

# Now assets in /path/to/my/assets/fonts/ are also discovered
```

## See Also

- {doc}`../assets/index` - Asset management guide
- {doc}`../cli/index` - CLI tools for asset discovery
