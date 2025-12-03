# Images

Test and sample images for image processing operations.

## Quick Start

For most documentation examples and testing, use `get_test_image()`:

```python
from yapocis.utils.basic import get_test_image

# Returns float32 RGB array normalized to [0, 1]
img = get_test_image()
```

For loading specific images by name:

```python
from yapocis.images import images, find_image
from yapocis.utils.basic import imread

# List available images
print(images())  # ['test.jpg', 'lena.png', ...]

# Get path to image
path = find_image("test")

# Load image as float32 array
img = imread(path)
```

## API Reference

### images()

List all available image names.

```python
def images() -> list[str]:
    """Return list of available image filenames."""
```

### find_image()

Find the path to an image file.

```python
def find_image(image_name: str) -> str:
    """
    Find image file path by name.

    Args:
        image_name: Image name (without extension)

    Returns:
        Absolute path to image file

    Raises:
        ValueError: If image not found
    """
```

## Image Format

- **File types**: JPG, PNG, TIF, BMP
- **Matching**: By name (extension auto-detected)
- **No default**: Raises error if image not found

## Asset Manager Configuration

```python
AssetManager(
    dirname="yapocis/images",
    ext=None,  # No primary extension
    extensions=[".jpg", ".png", ".tif", ".bmp"],
    match="name"
)
```

## Loading Images

Use `imread()` for flexible image loading:

```python
from yapocis.utils.basic import imread
from yapocis.images import find_image

# Load as float32 array
img = imread(find_image("test"))

# Load and resize
img = imread(find_image("test"), longest=512)

# Keep alpha channel
img = imread(find_image("test"), keep_alpha=True)
```

## Adding Custom Images

```bash
export YAPOCIS_PROJECT=/path/to/project

# Structure:
# /path/to/project/images/
#   ├── custom.jpg
#   └── sample.png
```

## Example Usage

```python
from yapocis.utils.basic import get_test_image
from yapocis.lib.gaussian import Gaussian
import numpy as np

# Load test image (preferred for examples)
img = get_test_image()

# Apply Gaussian blur
g = Gaussian()
gray = np.mean(img, axis=2).astype(np.float32)
blurred = g.gauss_image(gray, scale=3)
print(f"Blurred shape: {blurred.shape}")
```

For loading a specific image:

```python
from yapocis.images import find_image
from yapocis.utils.basic import imread
from yapocis.lib.gradient import Gradient
import numpy as np

# Load a specific named image
img = imread(find_image("test"))

# Compute gradient
g = Gradient()
gray = np.mean(img, axis=2).astype(np.float32)
grad, theta = g.gradient(gray)
print(f"Gradient shape: {grad.shape}")
```
