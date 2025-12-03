# VGG16

VGG16 deep CNN for ImageNet classification.

## Overview

The `VGG16` class implements the 16-layer VGG network architecture for image classification. It uses GPU-accelerated CNN kernels and supports both classification and feature embedding extraction.

## Architecture

| Block | Layers | Output Size | Channels |
|-------|--------|-------------|----------|
| Block 1 | conv3-64 x2, maxpool | 112x112 | 64 |
| Block 2 | conv3-128 x2, maxpool | 56x56 | 128 |
| Block 3 | conv3-256 x3, maxpool | 28x28 | 256 |
| Block 4 | conv3-512 x3, maxpool | 14x14 | 512 |
| Block 5 | conv3-512 x3, maxpool | 7x7 | 512 |
| FC6 | Fully connected | 4096 | 4096 |
| FC7 | Fully connected | 4096 | 4096 |
| FC8 | Fully connected | 1000 | 1000 |

Total parameters: 138M

## Class Reference

```python {.no-exec}
from yapocis.lib.vgg16 import VGG16

vgg = VGG16(weights_path="path/to/weights.npz")
```

## Methods

### load_weights

Load weights from npz file.

```python {.no-exec}
vgg.load_weights("vgg16_weights.npz")
```

### predict

Classify an image.

```python {.no-exec}
top_indices, top_probs = vgg.predict(image, top_k=5)
```

**Parameters:**
- `image` - (H, W, 3) array in [0, 1] or [0, 255]
- `top_k` - Number of predictions to return

**Returns:** (indices, probabilities) arrays

### forward

Full forward pass.

```python {.no-exec}
probs = vgg.forward(preprocessed_image, stop_at=None)
```

**Parameters:**
- `image` - Preprocessed (3, 224, 224) array
- `stop_at` - Layer to stop at ('fc6', 'fc7', or None)

### extract_features

Get 4096-dimensional feature vector.

```python {.no-exec}
features = vgg.extract_features(image, layer='fc6')
```

**Parameters:**
- `image` - Input image
- `layer` - 'fc6' or 'fc7' (both are 4096-dim)

### preprocess_image

Prepare image for VGG16 input.

```python {.no-exec}
preprocessed = vgg.preprocess_image(image)
```

Performs:
1. Resize to 224x224
2. Convert to channel-first (3, 224, 224)
3. Normalize with ImageNet mean/std

## Resident Version

For maximum performance, use `VGG16Resident`:

```python {.no-exec}
from yapocis.lib.vgg16 import VGG16Resident

vgg = VGG16Resident(weights_path="path/to/weights.npz")
vgg.allocate_buffers()

# Faster inference
probs = vgg.predict(image)
```

Benefits:
- Pre-allocated activation buffers
- Pre-written weights
- GPU-resident execution throughout
- Async mode support

## Usage

### Image Classification

```python {.no-exec}
from yapocis.lib.vgg16 import VGG16, load_imagenet_labels
import numpy as np

# Load model and labels
vgg = VGG16()
vgg.load_weights("vgg16_weights.npz")
labels = load_imagenet_labels()

# Load image
image = np.random.rand(224, 224, 3).astype(np.float32)

# Classify
indices, probs = vgg.predict(image, top_k=5)

for idx, prob in zip(indices, probs):
    print(f"{labels[idx]}: {prob:.3f}")
```

### Feature Extraction for Transfer Learning

```python {.no-exec}
# Extract FC6 features (4096-dim)
features = vgg.extract_features(image, layer='fc6')

# Use as input to another classifier
# features can be used for:
# - Similarity search
# - Clustering
# - Fine-tuning
```

### Async Execution

```python {.no-exec}
import os
os.environ['YAPOCIS_ASYNC'] = '1'

vgg = VGG16Resident()
vgg.load_weights("vgg16_weights.npz")
vgg.allocate_buffers()

# Each conv block runs in single async context
probs = vgg.predict(image)  # 8-12% speedup
```

## Weights Format

Expected npz file structure:

```
conv0_weights, conv0_bias, ..., conv12_weights, conv12_bias
fc0_weights, fc0_bias  (FC6)
fc1_weights, fc1_bias  (FC7)
fc2_weights, fc2_bias  (FC8)
```

## Performance

| Configuration | Inference Time |
|---------------|----------------|
| Base VGG16 | ~500ms |
| VGG16Resident | ~200ms |
| Resident + Async | ~180ms |

VGG16 is slower than ResNet50 due to more parameters (138M vs 25.6M) despite lower accuracy.

## Comparison with ResNet50

| Feature | VGG16 | ResNet50 |
|---------|-------|----------|
| Parameters | 138M | 25.6M |
| Top-1 Accuracy | 71.3% | 76.1% |
| Memory (resident) | ~500MB | ~280MB |
| Speed | Slower | Faster |

For new projects, ResNet50 is recommended unless VGG16 features are specifically needed (e.g., for compatibility with existing models).

## See Also

- {doc}`resnet50` - ResNet50 architecture (recommended)
