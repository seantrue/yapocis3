# VGG16 Model Weights

The VGG16 weights file (`weights.npz`) is too large to store in git.

## Download Weights

Run the download script to get the pre-trained ImageNet weights:

```bash
python scripts/download_vgg16.py
```

This will:
- Download VGG16 from PyTorch (138M parameters)
- Convert to yapocis format
- Save to `src/yapocis/models/vgg16/weights.npz` (528 MB)
- Download ImageNet class labels

## Usage

```python
from yapocis.lib.vgg16 import VGG16

# Initialize (weights loaded automatically)
model = VGG16()

# Classify image
top_indices, top_probs = model.predict("image.jpg", top_k=5)
```

## File Size

- `weights.npz`: 527.8 MB
- `classes.txt`: 10 KB (included in repo)
