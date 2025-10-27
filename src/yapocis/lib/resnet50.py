"""
ResNet50 neural network for image classification.

ResNet50 is a 50-layer deep convolutional neural network with residual connections.
It achieves better accuracy than VGG16 with fewer parameters (25.6M vs 138M).

Architecture:
- Stem: 7×7 conv, stride 2, 64 channels
- Stage 1: 3 bottleneck blocks, 256 channels (56×56)
- Stage 2: 4 bottleneck blocks, 512 channels (28×28)
- Stage 3: 6 bottleneck blocks, 1024 channels (14×14)
- Stage 4: 3 bottleneck blocks, 2048 channels (7×7)
- Global average pooling
- Fully connected layer (2048 → 1000)

Each bottleneck block uses: 1×1 conv → 3×3 conv → 1×1 conv with skip connection.

Example:
    from yapocis.lib.resnet50 import ResNet50
    from yapocis.models import get_weights_path, get_classes_path

    # Load model
    weights_path = get_weights_path('resnet50')
    resnet = ResNet50(str(weights_path))

    # Classify image
    image = load_image('dog.jpg')
    top_indices, top_probs = resnet.predict(image, top_k=5)

    # Extract embeddings (2048-dimensional features)
    embeddings = resnet.forward(image, stop_at='pool')
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict

from yapocis import core
from yapocis.models import get_weights_path, get_classes_path


def load_imagenet_labels(labels_path: Optional[str] = None) -> list[str]:
    """
    Load ImageNet class labels.

    Args:
        labels_path: Optional path to classes.txt file.
                    If None, loads from model asset manager.

    Returns:
        List of 1000 class names
    """
    if labels_path is None:
        labels_path_obj = get_classes_path('resnet50')
        if labels_path_obj:
            labels_path = str(labels_path_obj)
        else:
            raise FileNotFoundError(
                "ResNet50 classes.txt not found. "
                "Please provide labels_path or install model assets."
            )

    with open(labels_path, 'r') as f:
        return [line.strip() for line in f]


class ResNet50:
    """
    ResNet50 neural network for image classification.

    This implementation uses the unified CNN kernels and supports:
    - Full classification (1000 ImageNet classes)
    - Embedding extraction (2048-dimensional features from global pooling)

    Args:
        weights_path: Path to weights.npz file. If None, loads from model asset manager.
    """

    def __init__(self, weights_path: Optional[str] = None):
        """Initialize ResNet50 with pre-trained weights."""
        # Load weights
        if weights_path is None:
            weights_path_obj = get_weights_path('resnet50')
            if weights_path_obj:
                weights_path = str(weights_path_obj)
            else:
                raise FileNotFoundError(
                    "ResNet50 weights not found. "
                    "Please provide weights_path or install model assets."
                )

        self.weights = np.load(weights_path)

        # Load CNN kernels
        self.prog = core.load_program('cnn')

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ResNet50.

        Follows PyTorch's standard preprocessing:
        1. Resize to 256 (shortest edge)
        2. Center crop to 224×224
        3. Normalize with ImageNet mean/std

        Args:
            image: Input image as (H, W, 3) array in range [0, 1]

        Returns:
            Preprocessed image as (3, 224, 224) array
        """
        from PIL import Image

        # Convert to PIL for resizing
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        pil_image = Image.fromarray((image * 255).astype(np.uint8))

        # Resize to 256 (shortest edge)
        w, h = pil_image.size
        if h < w:
            new_h, new_w = 256, int(256 * w / h)
        else:
            new_h, new_w = int(256 * h / w), 256
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

        # Center crop to 224×224
        w, h = pil_image.size
        left = (w - 224) // 2
        top = (h - 224) // 2
        pil_image = pil_image.crop((left, top, left + 224, top + 224))

        image = np.array(pil_image).astype(np.float32) / 255.0

        # Normalize with ImageNet mean and std
        image = (image - self.mean) / self.std

        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return np.ascontiguousarray(image.astype(np.float32))

    def _bottleneck_block(
        self,
        x: np.ndarray,
        block_idx: int,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False
    ) -> np.ndarray:
        """
        Single bottleneck residual block: 1×1 → 3×3 → 1×1 with skip connection.

        Args:
            x: Input tensor (C_in, H, W)
            block_idx: Block index for weight lookup
            in_channels: Input channels
            mid_channels: Middle channels (bottleneck)
            out_channels: Output channels
            stride: Stride for 3×3 conv (1 or 2)
            downsample: Whether to use projection shortcut

        Returns:
            Output tensor (C_out, H', W')
        """
        # Get dimensions
        C_in, H, W = x.shape
        H_out = H // stride if stride == 2 else H
        W_out = W // stride if stride == 2 else W

        # Ensure input is contiguous
        x = np.ascontiguousarray(x)

        # Load weights for this block
        conv1_W = np.ascontiguousarray(self.weights[f'block{block_idx}_conv1_weights'])
        conv1_b = np.ascontiguousarray(self.weights[f'block{block_idx}_conv1_bias'])
        conv2_W = np.ascontiguousarray(self.weights[f'block{block_idx}_conv2_weights'])
        conv2_b = np.ascontiguousarray(self.weights[f'block{block_idx}_conv2_bias'])
        conv3_W = np.ascontiguousarray(self.weights[f'block{block_idx}_conv3_weights'])
        conv3_b = np.ascontiguousarray(self.weights[f'block{block_idx}_conv3_bias'])

        # 1×1 conv to reduce channels
        out1 = np.ascontiguousarray(np.zeros((mid_channels, H, W), dtype=np.float32))
        self.prog.conv2d_1x1(x, conv1_W, conv1_b, out1, mid_channels)
        out1 = self.prog.relu_activation(out1)

        # 3×3 conv (with optional stride=2 for downsampling)
        if stride == 2:
            out2 = np.ascontiguousarray(np.zeros((mid_channels, H_out, W_out), dtype=np.float32))
            self.prog.conv2d_3x3_stride2(out1, conv2_W, conv2_b, out2, mid_channels)
        else:
            out2 = np.ascontiguousarray(np.zeros((mid_channels, H, W), dtype=np.float32))
            self.prog.conv2d_3x3(out1, conv2_W, conv2_b, out2, mid_channels)
        out2 = self.prog.relu_activation(out2)

        # 1×1 conv to expand channels
        out3 = np.ascontiguousarray(np.zeros((out_channels, H_out, W_out), dtype=np.float32))
        self.prog.conv2d_1x1(out2, conv3_W, conv3_b, out3, out_channels)

        # Skip connection
        if downsample:
            # Project input to match output dimensions
            proj_W = np.ascontiguousarray(self.weights[f'block{block_idx}_downsample_weights'])
            proj_b = np.ascontiguousarray(self.weights[f'block{block_idx}_downsample_bias'])
            shortcut = np.ascontiguousarray(np.zeros((out_channels, H_out, W_out), dtype=np.float32))
            if stride == 2:
                self.prog.conv2d_1x1_stride2(x, proj_W, proj_b, shortcut, out_channels)
            else:
                self.prog.conv2d_1x1(x, proj_W, proj_b, shortcut, out_channels)
        else:
            # Identity shortcut
            shortcut = x

        # Add residual
        result = self.prog.add_tensors(out3, shortcut)

        # ReLU after addition
        result = self.prog.relu_activation(result)

        return result

    def forward(self, image: np.ndarray, stop_at: Optional[str] = None) -> np.ndarray:
        """
        Forward pass through ResNet50.

        Args:
            image: Preprocessed image as (3, 224, 224) array
            stop_at: Optional layer to stop at for embeddings:
                    - 'pool': Return 2048-dimensional features after global pooling
                    - None: Return 1000-dimensional class probabilities

        Returns:
            Output features or probabilities
        """
        x = np.ascontiguousarray(image)

        # Stem: 7×7 conv, stride 2, 64 channels (224×224 → 112×112)
        stem_W = np.ascontiguousarray(self.weights['stem_weights'])
        stem_b = np.ascontiguousarray(self.weights['stem_bias'])
        x = np.ascontiguousarray(np.zeros((64, 112, 112), dtype=np.float32))
        self.prog.conv2d_7x7_stride2(image, stem_W, stem_b, x, 64)
        x = self.prog.relu_activation(x)

        # Max pool 3×3, stride 2, padding 1 (112×112 → 56×56)
        x_pooled = np.ascontiguousarray(np.zeros((64, 56, 56), dtype=np.float32))
        x = self.prog.max_pool_3x3_stride2(x, x_pooled)
        x = np.ascontiguousarray(x)

        # Stage 1: 3 bottleneck blocks (64 → 256 channels, 56×56)
        # First block has downsample projection
        x = self._bottleneck_block(x, 0, 64, 64, 256, stride=1, downsample=True)
        x = self._bottleneck_block(x, 1, 256, 64, 256, stride=1, downsample=False)
        x = self._bottleneck_block(x, 2, 256, 64, 256, stride=1, downsample=False)

        # Stage 2: 4 bottleneck blocks (256 → 512 channels, 28×28)
        # First block uses stride=2 for downsampling
        x = self._bottleneck_block(x, 3, 256, 128, 512, stride=2, downsample=True)
        x = self._bottleneck_block(x, 4, 512, 128, 512, stride=1, downsample=False)
        x = self._bottleneck_block(x, 5, 512, 128, 512, stride=1, downsample=False)
        x = self._bottleneck_block(x, 6, 512, 128, 512, stride=1, downsample=False)

        # Stage 3: 6 bottleneck blocks (512 → 1024 channels, 14×14)
        # First block uses stride=2 for downsampling
        x = self._bottleneck_block(x, 7, 512, 256, 1024, stride=2, downsample=True)
        x = self._bottleneck_block(x, 8, 1024, 256, 1024, stride=1, downsample=False)
        x = self._bottleneck_block(x, 9, 1024, 256, 1024, stride=1, downsample=False)
        x = self._bottleneck_block(x, 10, 1024, 256, 1024, stride=1, downsample=False)
        x = self._bottleneck_block(x, 11, 1024, 256, 1024, stride=1, downsample=False)
        x = self._bottleneck_block(x, 12, 1024, 256, 1024, stride=1, downsample=False)

        # Stage 4: 3 bottleneck blocks (1024 → 2048 channels, 7×7)
        # First block uses stride=2 for downsampling
        x = self._bottleneck_block(x, 13, 1024, 512, 2048, stride=2, downsample=True)
        x = self._bottleneck_block(x, 14, 2048, 512, 2048, stride=1, downsample=False)
        x = self._bottleneck_block(x, 15, 2048, 512, 2048, stride=1, downsample=False)

        # Global average pooling (7×7 → 1×1)
        pool_out = np.ascontiguousarray(np.zeros(2048, dtype=np.float32))
        self.prog.global_avg_pool(x, pool_out)

        if stop_at == 'pool':
            return pool_out

        # Fully connected layer (2048 → 1000)
        fc_W = np.ascontiguousarray(self.weights['fc_weights'])
        fc_b = np.ascontiguousarray(self.weights['fc_bias'])
        logits = np.ascontiguousarray(np.zeros(1000, dtype=np.float32))
        self.prog.fully_connected(pool_out, fc_W, fc_b, logits)

        # Softmax
        probs = self.prog.softmax(logits)

        return probs

    def predict(self, image: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify image and return top-K predictions.

        Args:
            image: Input image as (H, W, 3) array in range [0, 1] or [0, 255]
            top_k: Number of top predictions to return

        Returns:
            (top_indices, top_probabilities) tuples
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)

        # Run forward pass
        probs = self.forward(preprocessed)

        # Get top-K predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]

        return top_indices, top_probs

    def extract_embeddings(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 2048-dimensional feature embeddings.

        Args:
            image: Input image as (H, W, 3) array in range [0, 1] or [0, 255]

        Returns:
            2048-dimensional feature vector
        """
        preprocessed = self.preprocess_image(image)
        return self.forward(preprocessed, stop_at='pool')


class ResNet50Resident(ResNet50):
    """
    ResNet50 using resident buffers for reduced memory transfers.

    This version keeps intermediate activations on the GPU/compute device
    to minimize data transfer overhead. Useful for batch processing.
    """

    def __init__(self, weights_path: Optional[str] = None):
        """Initialize ResNet50 with resident buffer support."""
        super().__init__(weights_path)
        # TODO: Implement resident buffer version using 'res' extensions
        # Similar to VGG16Resident implementation
