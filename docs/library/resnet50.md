# ResNet50

ResNet50 deep CNN for ImageNet classification.

## Overview

The `ResNet50` class implements a 50-layer residual network for image classification. It uses GPU-accelerated CNN kernels and supports both classification and feature embedding extraction.

## Architecture

| Stage | Layers | Output Size | Channels |
|-------|--------|-------------|----------|
| Stem | 7x7 conv, maxpool | 56x56 | 64 |
| Stage 1 | 3 bottleneck blocks | 56x56 | 256 |
| Stage 2 | 4 bottleneck blocks | 28x28 | 512 |
| Stage 3 | 6 bottleneck blocks | 14x14 | 1024 |
| Stage 4 | 3 bottleneck blocks | 7x7 | 2048 |
| Output | Global avg pool, FC | 1 | 1000 |

Total parameters: 25.6M (vs VGG16's 138M)

## Class Reference

```python {.no-exec}
from yapocis.lib.resnet50 import ResNet50

resnet = ResNet50(weights_path="path/to/weights.npz")
```

## Methods

### predict

Classify an image.

```python {.no-exec}
top_indices, top_probs = resnet.predict(image, top_k=5)
```

**Parameters:**
- `image` - (H, W, 3) array in [0, 1] or [0, 255]
- `top_k` - Number of predictions to return

**Returns:** (indices, probabilities) arrays

### forward

Full forward pass.

```python {.no-exec}
probs = resnet.forward(preprocessed_image, stop_at=None)
```

**Parameters:**
- `image` - Preprocessed (3, 224, 224) array
- `stop_at` - Optional layer to stop at ('pool' for embeddings)

### extract_embeddings

Get 2048-dimensional feature vector.

```python {.no-exec}
embedding = resnet.extract_embeddings(image)
```

### preprocess_image

Prepare image for inference.

```python {.no-exec}
preprocessed = resnet.preprocess_image(image)
```

Performs:
1. Resize to 256 (shortest edge)
2. Center crop to 224x224
3. Normalize with ImageNet mean/std

## Resident Version

For maximum performance, use `ResNet50Resident`:

```python {.no-exec}
from yapocis.lib.resnet50 import ResNet50Resident

resnet = ResNet50Resident(weights_path="path/to/weights.npz")
resnet.allocate_buffers()  # Pre-allocate GPU buffers

# 3-4x faster inference
probs = resnet.predict(image)
```

Benefits:
- Pre-allocated GPU-resident activation buffers
- Pre-written weights (~100MB kept on GPU)
- Eliminates redundant CPU↔GPU transfers
- ~280 MB total GPU memory

## Usage

### Image Classification

```python {.no-exec}
from yapocis.lib.resnet50 import ResNet50, load_imagenet_labels
import numpy as np

# Load model and labels
resnet = ResNet50()
labels = load_imagenet_labels()

# Load image (as numpy array)
image = np.random.rand(224, 224, 3).astype(np.float32)

# Classify
indices, probs = resnet.predict(image, top_k=5)

for idx, prob in zip(indices, probs):
    print(f"{labels[idx]}: {prob:.3f}")
```

### Feature Extraction

```python {.no-exec}
# Extract 2048-dim embedding for similarity/clustering
embedding = resnet.extract_embeddings(image)
print(f"Embedding shape: {embedding.shape}")  # (2048,)

# Compare two images
emb1 = resnet.extract_embeddings(image1)
emb2 = resnet.extract_embeddings(image2)
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

### Batch Processing with Resident Mode

```python {.no-exec}
from yapocis.lib.resnet50 import ResNet50Resident

resnet = ResNet50Resident()
resnet.allocate_buffers()

# Enable async mode for additional speedup
import os
os.environ['YAPOCIS_ASYNC'] = '1'

# Process many images efficiently
for image in image_batch:
    probs = resnet.predict(image)
```

## Weights Format

Weights file (`.npz`) should contain:

```
stem_weights, stem_bias
block{0-15}_conv{1,2,3}_weights, block{0-15}_conv{1,2,3}_bias
block{0,3,7,13}_downsample_weights, block{0,3,7,13}_downsample_bias
fc_weights, fc_bias
```

## Performance

| Configuration | Inference Time |
|---------------|----------------|
| Base ResNet50 | ~200ms |
| ResNet50Resident | ~60ms |
| Resident + Async | ~50ms |

Memory: ~280 MB GPU (resident mode)

## See Also

- {doc}`vgg16` - VGG16 architecture
