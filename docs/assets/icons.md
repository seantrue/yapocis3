# Icons

Icon assets for graphics applications.

## Quick Start

```python
from yapocis.icons import icon, icons, find_icon

# List available icons
print(icons())

# Get icon as numpy array (resized)
img = icon("folder", size=64)

# Get path to icon file
path = find_icon("folder")
```

## API Reference

### icon()

Load and resize an icon, with caching.

```python
def icon(icon_name="default", size=64) -> np.ndarray:
    """
    Return an icon as a numpy array, caching as possible.

    Args:
        icon_name: Icon name (partial match supported)
        size: Longest edge in pixels

    Returns:
        RGBA numpy array (height, width, 4), dtype float32
    """
```

Example:

```python
from yapocis.icons import icon

# Load 64x64 folder icon
folder_icon = icon("folder", size=64)

# Load 128x128 default icon
default_icon = icon(size=128)
```

### icons()

List all available icon names.

```python
def icons() -> list[str]:
    """Return list of available icon filenames."""
```

### find_icon()

Find the path to an icon file.

```python
def find_icon(icon_name: str) -> str:
    """
    Find icon file path with flexible matching.

    Args:
        icon_name: Icon name (partial match supported)

    Returns:
        Absolute path to icon file

    Raises:
        ValueError: If icon not found
    """
```

## Icon Format

- **File types**: PNG (primary), GIF
- **Color format**: RGBA
- **Naming**: Uses partial matching, so "folder" matches "folder-open.png"
- **Default**: Falls back to "default.png" if icon not found

## Asset Manager Configuration

The icon asset manager is configured for recursive, partial matching:

```python
AssetManager(
    dirname="yapocis/icons",
    ext=".png",
    extensions=[".gif"],
    match="partial",
    recursive=True,
    default="default.png"
)
```

## Adding Custom Icons

Place custom icons in a project directory:

```bash
export YAPOCIS_PROJECT=/path/to/project

# Structure:
# /path/to/project/icons/
#   ├── my-icon.png
#   └── subfolder/
#       └── another-icon.png
```

Custom icons override built-in icons with the same name.
