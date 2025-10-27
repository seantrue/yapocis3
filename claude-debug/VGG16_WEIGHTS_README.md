# VGG16 Pre-trained Weights for yapocis

This directory contains tools to download and use pre-trained VGG16 weights from the ONNX Model Zoo with yapocis kernels.

## Quick Start

### 1. Install ONNX (Required)

```bash
pip install onnx
```

### 2. Download VGG16 Weights

```bash
cd claude-debug
python download_vgg16_onnx.py
```

This will:
- Download VGG16 ONNX model (~550 MB) from the ONNX Model Zoo
- Extract all weights (13 conv layers + 3 FC layers)
- Save to `vgg16_weights.npz` (~138 MB)

### 3. Run Example Inference

```bash
# With random input
python vgg16_example.py

# With your own image
python vgg16_example.py --image /path/to/image.jpg
```

Note: Image inference requires PIL/Pillow: `pip install pillow`

## Files

- **download_vgg16_onnx.py** - Downloads and extracts VGG16 weights from ONNX
- **vgg16_example.py** - Demo inference using yapocis kernels
- **vgg16_weights.npz** - Extracted weights (created by download script)
- **vgg16.onnx** - Downloaded ONNX model (created by download script)

## Weight Format

The NPZ file contains weights in yapocis-compatible format:

### Convolutional Layers (13 total)

```
conv0_weights: (64, 3, 3, 3)      conv0_bias: (64,)     # Block 1
conv1_weights: (64, 64, 3, 3)     conv1_bias: (64,)

conv2_weights: (128, 64, 3, 3)    conv2_bias: (128,)    # Block 2
conv3_weights: (128, 128, 3, 3)   conv3_bias: (128,)

conv4_weights: (256, 128, 3, 3)   conv4_bias: (256,)    # Block 3
conv5_weights: (256, 256, 3, 3)   conv5_bias: (256,)
conv6_weights: (256, 256, 3, 3)   conv6_bias: (256,)

conv7_weights: (512, 256, 3, 3)   conv7_bias: (512,)    # Block 4
conv8_weights: (512, 512, 3, 3)   conv8_bias: (512,)
conv9_weights: (512, 512, 3, 3)   conv9_bias: (512,)

conv10_weights: (512, 512, 3, 3)  conv10_bias: (512,)   # Block 5
conv11_weights: (512, 512, 3, 3)  conv11_bias: (512,)
conv12_weights: (512, 512, 3, 3)  conv12_bias: (512,)
```

### Fully Connected Layers (3 total)

```
fc0_weights: (4096, 25088)  fc0_bias: (4096,)   # FC6
fc1_weights: (4096, 4096)   fc1_bias: (4096,)   # FC7
fc2_weights: (1000, 4096)   fc2_bias: (1000,)   # FC8 (classifier)
```

**Total Parameters**: 138,357,544 (~138 million)

## VGG16 Architecture

```
Input: (3, 224, 224) RGB image

Block 1: Conv(64)-ReLU-Conv(64)-ReLU-MaxPool    → (64, 112, 112)
Block 2: Conv(128)-ReLU-Conv(128)-ReLU-MaxPool  → (128, 56, 56)
Block 3: Conv(256)-ReLU-Conv(256)-ReLU-Conv(256)-ReLU-MaxPool → (256, 28, 28)
Block 4: Conv(512)-ReLU-Conv(512)-ReLU-Conv(512)-ReLU-MaxPool → (512, 14, 14)
Block 5: Conv(512)-ReLU-Conv(512)-ReLU-Conv(512)-ReLU-MaxPool → (512, 7, 7)

Flatten: → (25088,)

FC6: (4096,) + ReLU + Dropout
FC7: (4096,) + ReLU + Dropout
FC8: (1000,) + Softmax

Output: (1000,) class probabilities (ImageNet classes)
```

## Usage Example

```python
import numpy as np
from yapocis import core

# Load weights
weights = np.load('vgg16_weights.npz')

# Load VGG16 program
prog = core.load_program("vgg16")

# Prepare input (3, 224, 224)
input_img = np.random.randn(3, 224, 224).astype(np.float32)

# First convolution: 3 -> 64 channels
conv1_out = np.zeros((64, 224, 224), dtype=np.float32)
prog.conv2d_3x3(
    input_img,
    weights['conv0_weights'],
    weights['conv0_bias'],
    conv1_out,
    64  # output_channels
)

# ReLU activation
relu1_out = prog.relu_activation(conv1_out.reshape(-1)).reshape(64, 224, 224)

# Continue with remaining layers...
```

## Image Preprocessing

VGG16 expects specific preprocessing:

```python
from PIL import Image

# 1. Resize to 224x224
img = Image.open('image.jpg').convert('RGB')
img = img.resize((224, 224), Image.BILINEAR)

# 2. Convert to float array
img_array = np.array(img, dtype=np.float32)

# 3. Subtract ImageNet mean (RGB order)
mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
img_array -= mean

# 4. Convert to CHW format (channels first)
img_array = np.transpose(img_array, (2, 0, 1))  # (3, 224, 224)
```

## ImageNet Classes

The FC8 output layer produces 1000 class probabilities corresponding to ImageNet-1K classes.

Download class labels:
```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

Map predictions to class names:
```python
# After running full VGG16 inference
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f]

# Get top-5 predictions
top5_idx = np.argsort(predictions)[-5:][::-1]
for idx in top5_idx:
    print(f"{classes[idx]}: {predictions[idx]:.3f}")
```

## Command-Line Options

### download_vgg16_onnx.py

```bash
python download_vgg16_onnx.py [options]

Options:
  -o, --output PATH      Output NPZ file path (default: vgg16_weights.npz)
  --skip-download        Skip download if ONNX file exists
  --onnx-file PATH       Path to ONNX file (default: vgg16.onnx)
```

### vgg16_example.py

```bash
python vgg16_example.py [options]

Options:
  --weights PATH    Path to weights NPZ file (default: vgg16_weights.npz)
  --image PATH      Input image path (uses random data if not provided)
```

## Troubleshooting

### Download Fails

If automatic download fails, manually download VGG16:

1. Visit: https://github.com/onnx/models/tree/main/vision/classification/vgg
2. Download `vgg16-7.onnx` (or latest version)
3. Save to `claude-debug/vgg16.onnx`
4. Run: `python download_vgg16_onnx.py --skip-download`

### Import Errors

```bash
# For ONNX support
pip install onnx

# For image loading
pip install pillow

# For numpy (should already be installed)
pip install numpy
```

### Metal Backend Required

The VGG16 kernels currently only support the Metal backend (macOS):

```bash
export YAPOCIS_BACKEND=metal
python vgg16_example.py
```

## Model Source

- **Source**: [ONNX Model Zoo](https://github.com/onnx/models)
- **Model**: VGG16-7 (ImageNet-1K pre-trained)
- **License**: BSD 3-Clause (original Caffe model)
- **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2014)

## References

- Original VGG paper: https://arxiv.org/abs/1409.1556
- ONNX Model Zoo: https://github.com/onnx/models
- yapocis VGG16 interface: `src/yapocis/interfaces/vgg16.interface`
- yapocis VGG16 kernels: `src/yapocis/kernels/metal/vgg16.mako`
- Tests: `tests/test_vgg16_kernels.py`
