# Models

Pre-trained neural network model assets.

## Quick Start

```python
from yapocis.models import models, find_model, get_weights_path, get_classes_path

# List available models
print(models())

# Get path to model directory
vgg16_dir = find_model("vgg16")

# Get specific model files
weights = get_weights_path("vgg16")
classes = get_classes_path("vgg16")
```

## API Reference

### models()

List all available model names.

```python
def models() -> list[str]:
    """Return sorted list of available model names."""
```

### find_model()

Find the directory for a model.

```python
def find_model(name: str) -> Path | None:
    """
    Find a model directory by name.

    Args:
        name: Model name

    Returns:
        Path to model directory, or None if not found
    """
```

### get_model_path()

Get a specific file within a model directory.

```python
def get_model_path(model_name: str, filename: str) -> Path | None:
    """
    Get path to a specific file within a model directory.

    Args:
        model_name: Model name
        filename: Filename (e.g., 'weights.npz', 'classes.txt')

    Returns:
        Path to file, or None if not found
    """
```

### get_model_files()

Get all files in a model directory.

```python
def get_model_files(model_name: str) -> dict[str, Path] | None:
    """
    Get all files in a model directory.

    Args:
        model_name: Model name

    Returns:
        Dictionary mapping filename to path, or None if not found
    """
```

### get_weights_path()

Convenience method for model weights.

```python
def get_weights_path(model_name: str) -> Path | None:
    """Get path to weights.npz file."""
```

### get_classes_path()

Convenience method for class labels.

```python
def get_classes_path(model_name: str) -> Path | None:
    """Get path to classes.txt file."""
```

### model_exists()

Check if a model exists.

```python
def model_exists(model_name: str) -> bool:
    """Return True if model exists."""
```

## Model Directory Structure

Each model is stored in its own directory:

```
models/
├── vgg16/
│   ├── weights.npz      # Model weights
│   ├── classes.txt      # Class labels
│   ├── config.json      # Model configuration (optional)
│   └── README.md        # Model documentation (optional)
└── resnet50/
    ├── weights.npz
    └── classes.txt
```

## Using Models

### VGG16 Example

```python {.no-exec}
from yapocis.models import get_weights_path, get_classes_path
from yapocis.lib.vgg16 import VGG16

# Load model with weights
vgg = VGG16()
weights_path = get_weights_path("vgg16")
vgg.load_weights(weights_path)

# Load class labels
classes_path = get_classes_path("vgg16")
with open(classes_path) as f:
    classes = [line.strip() for line in f]

# Classify image
predictions = vgg.predict(image)
```

### ResNet50 Example

```python
from yapocis.lib.resnet50 import ResNet50

resnet = ResNet50()
# Model automatically loads weights from models/resnet50/
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `YAPOCIS_MODELS` | Additional model search directory |

## Adding Custom Models

```bash
export YAPOCIS_MODELS=/path/to/models

# Structure:
# /path/to/models/
#   └── my_model/
#       ├── weights.npz
#       ├── classes.txt
#       └── config.json
```

Project models override built-in models with the same name.

## ModelAssetManager Class

The model-specific asset manager:

```python
from yapocis.models import _model_manager

# List models
_model_manager.list_models()

# Check if model exists
_model_manager.model_exists("vgg16")

# Get all files
files = _model_manager.get_model_files("vgg16")
for name, path in files.items():
    print(f"{name}: {path}")
```
