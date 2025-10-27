# ResNet50 Model Weights

The ResNet50 weights file (`weights.npz`) is too large to store in git.

## Download Weights

Run the download script to get the pre-trained ImageNet weights:

```bash
python scripts/download_resnet50.py
```

This will:
- Download ResNet50 from PyTorch (25.6M parameters)
- Fuse batch normalization into convolution weights
- Save to `src/yapocis/models/resnet50/weights.npz` (97 MB)
- Download ImageNet class labels

## Usage

```python
from yapocis.lib.resnet50 import ResNet50

# Initialize (weights loaded automatically)
model = ResNet50()

# Classify image
top_indices, top_probs = model.predict("image.jpg", top_k=5)
```

## File Size

- `weights.npz`: 97.4 MB
- `classes.txt`: 10 KB (included in repo)
