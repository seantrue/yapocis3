# Fonts

TrueType font assets for text rendering.

## Quick Start

```python
from yapocis.fonts import fonts, font, image_font

# List available fonts
print(fonts())

# Get path to font file
path = font("ubuntu")

# Get PIL ImageFont for text rendering
pil_font = image_font("ubuntu", size=24)
```

## API Reference

### fonts()

List all available font names.

```python
def fonts() -> list[str]:
    """Return list of available font filenames."""
```

### font()

Find the path to a font file.

```python
def font(font_suggestion: str) -> str:
    """
    Find font file path with case-insensitive matching.

    Args:
        font_suggestion: Font name (case-insensitive)

    Returns:
        Absolute path to font file
        Falls back to Ubuntu-Regular.ttf if not found
    """
```

### image_font()

Get a PIL FreeTypeFont for text rendering.

```python
def image_font(
    font_name: str | None = "default",
    size: int = 14,
    index: int = 0,
    encoding: str = "",
    layout_engine=None
) -> FreeTypeFont:
    """
    Return a font wrapper, caching as possible.

    Args:
        font_name: Font name (uses font() for resolution)
        size: Font size in pixels
        index: Font index for TTC files
        encoding: Character encoding
        layout_engine: PIL layout engine

    Returns:
        PIL FreeTypeFont object

    Raises:
        RuntimeError: If PIL TrueType support unavailable
    """
```

## Font Format

- **File type**: TrueType fonts (.ttf)
- **Matching**: Case-insensitive partial match
- **Default**: Falls back to Ubuntu-Regular.ttf

## Asset Manager Configuration

```python
AssetManager(
    dirname="yapocis/fonts",
    ext=".ttf",
    match="lower"
)
```

## Text Rendering

Use with yapocis text rendering:

```python
from yapocis.lib.typeset import render_text

# Render text to array (uses font_name, not PIL font)
bbox, text_img = render_text(
    "Hello World",
    font_name="ubuntu",  # Font name, not PIL font object
    size=32
)
```

Or with Graphics4:

```python {.no-exec}
from yapocis.lib.graphics4 import Graphics4
import numpy as np

canvas = np.zeros((512, 512, 4), dtype=np.float32)
g = Graphics4()

# Add text to canvas using draw_strings
g.draw_strings(canvas, [(100, 100)], ["Hello"], font="ubuntu", size=24, color=[1, 1, 1, 1])
```

## Adding Custom Fonts

```bash
export YAPOCIS_PROJECT=/path/to/project

# Structure:
# /path/to/project/fonts/
#   ├── CustomFont.ttf
#   └── AnotherFont.ttf
```

## Bundled Fonts

The package includes Ubuntu font family by default:

- Ubuntu-Regular.ttf
- Ubuntu-Bold.ttf
- Ubuntu-Italic.ttf
- Ubuntu-BoldItalic.ttf
- UbuntuMono-Regular.ttf
