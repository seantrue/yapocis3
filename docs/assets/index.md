# Asset Management

Yapocis provides a unified asset management system for accessing bundled resources like icons, images, fonts, and pre-trained neural network models.

## Overview

Assets are managed through the `AssetManager` class which handles:

- **Discovery**: Scans multiple directories for assets
- **Resolution**: Finds assets by exact name, partial match, or extension
- **Caching**: Caches asset paths and loaded resources for performance
- **Project Override**: Allows project-specific assets via `YAPOCIS_PROJECT`

## Asset Search Path

The asset system searches directories in order:

1. Built-in package directory (`yapocis/<asset_type>/`)
2. Current working directory (`<cwd>/<asset_type>/`)
3. Project directory (`$YAPOCIS_PROJECT/<asset_type>/`)

Project assets override built-in assets with the same name.

## Asset Packages

```{toctree}
:maxdepth: 1

icons
images
fonts
models
```

## CLI Tools

List available assets:

```bash
# List all asset types
yapocis-assets --list --all

# List specific asset types
yapocis-assets --list --icons
yapocis-assets --list --images
yapocis-assets --list --fonts

# Find specific asset
yapocis-assets test.jpg --images
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `YAPOCIS_PROJECT` | Additional asset search directory |
| `YAPOCIS_MODELS` | Additional model search directory |

## AssetManager Class

The base class for all asset managers:

```python
from yapocis.utils.internals import AssetManager

# Example: Create a custom asset manager
am = AssetManager(
    dirname="/path/to/assets",   # Base directory
    ext=".png",                   # Primary extension
    extensions=[".gif", ".jpg"],  # Additional extensions
    match="partial",              # Matching mode
    recursive=True,               # Search subdirectories
    default="default.png"         # Fallback asset
)

# List available assets
asset_names = am.assets()  # Returns list of found asset names

# Get asset path (exact match)
# path = am.asset_path("icon.png")

# Find asset (flexible matching)
# path = am.asset("icon")  # Tries various matches
```

### Matching Modes

| Mode | Behavior |
|------|----------|
| `"exact"` | Exact filename match only |
| `"name"` | Match without extension |
| `"lower"` | Case-insensitive match |
| `"partial"` | Substring match |
