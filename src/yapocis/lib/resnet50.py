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

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

from __future__ import annotations

import numpy as np

from yapocis import core
from yapocis.models import get_classes_path, get_weights_path


def load_imagenet_labels(labels_path: str | None = None) -> list[str]:
    """
    Load ImageNet class labels.

    Args:
        labels_path: Optional path to classes.txt file.
                    If None, loads from model asset manager.

    Returns:
        List of 1000 class names
    """
    if labels_path is None:
        labels_path_obj = get_classes_path("resnet50")
        if labels_path_obj:
            labels_path = str(labels_path_obj)
        else:
            raise FileNotFoundError(
                "ResNet50 classes.txt not found. "
                "Please provide labels_path or install model assets."
            )

    with open(labels_path) as f:
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

    def __init__(self, weights_path: str | None = None):
        """Initialize ResNet50 with pre-trained weights."""
        # Load weights
        if weights_path is None:
            weights_path_obj = get_weights_path("resnet50")
            if weights_path_obj:
                weights_path = str(weights_path_obj)
            else:
                raise FileNotFoundError(
                    "ResNet50 weights not found. "
                    "Please provide weights_path or install model assets."
                )

        self.weights = np.load(weights_path)

        # Load CNN kernels
        self.prog = core.load_program("cnn")

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
        downsample: bool = False,
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
        conv1_W = np.ascontiguousarray(self.weights[f"block{block_idx}_conv1_weights"])
        conv1_b = np.ascontiguousarray(self.weights[f"block{block_idx}_conv1_bias"])
        conv2_W = np.ascontiguousarray(self.weights[f"block{block_idx}_conv2_weights"])
        conv2_b = np.ascontiguousarray(self.weights[f"block{block_idx}_conv2_bias"])
        conv3_W = np.ascontiguousarray(self.weights[f"block{block_idx}_conv3_weights"])
        conv3_b = np.ascontiguousarray(self.weights[f"block{block_idx}_conv3_bias"])

        # 1×1 conv to reduce channels
        out1 = np.ascontiguousarray(np.zeros((mid_channels, H, W), dtype=np.float32))
        self.prog.conv2d_1x1(x, conv1_W, conv1_b, out1, mid_channels)
        out1 = self.prog.relu_activation(out1)

        # 3×3 conv (with optional stride=2 for downsampling)
        if stride == 2:
            out2 = np.ascontiguousarray(
                np.zeros((mid_channels, H_out, W_out), dtype=np.float32)
            )
            self.prog.conv2d_3x3_stride2(out1, conv2_W, conv2_b, out2, mid_channels)
        else:
            out2 = np.ascontiguousarray(
                np.zeros((mid_channels, H, W), dtype=np.float32)
            )
            self.prog.conv2d_3x3(out1, conv2_W, conv2_b, out2, mid_channels)
        out2 = self.prog.relu_activation(out2)

        # 1×1 conv to expand channels
        out3 = np.ascontiguousarray(
            np.zeros((out_channels, H_out, W_out), dtype=np.float32)
        )
        self.prog.conv2d_1x1(out2, conv3_W, conv3_b, out3, out_channels)

        # Skip connection
        if downsample:
            # Project input to match output dimensions
            proj_W = np.ascontiguousarray(
                self.weights[f"block{block_idx}_downsample_weights"]
            )
            proj_b = np.ascontiguousarray(
                self.weights[f"block{block_idx}_downsample_bias"]
            )
            shortcut = np.ascontiguousarray(
                np.zeros((out_channels, H_out, W_out), dtype=np.float32)
            )
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

    def forward(self, image: np.ndarray, stop_at: str | None = None) -> np.ndarray:
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
        stem_W = np.ascontiguousarray(self.weights["stem_weights"])
        stem_b = np.ascontiguousarray(self.weights["stem_bias"])
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

        if stop_at == "pool":
            return pool_out

        # Fully connected layer (2048 → 1000)
        fc_W = np.ascontiguousarray(self.weights["fc_weights"])
        fc_b = np.ascontiguousarray(self.weights["fc_bias"])
        logits = np.ascontiguousarray(np.zeros(1000, dtype=np.float32))
        self.prog.fully_connected(pool_out, fc_W, fc_b, logits)

        # Softmax
        probs = self.prog.softmax(logits)

        return probs

    def predict(
        self, image: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return self.forward(preprocessed, stop_at="pool")


class ResNet50Resident(ResNet50):
    """
    ResNet50 using resident buffers for maximum performance.

    This version keeps all intermediate activations on GPU and pre-writes
    all weights during initialization, eliminating redundant CPU↔GPU transfers.

    Provides 3-4x speedup over base ResNet50 through:
    - Pre-allocated GPU-resident activation buffers
    - Pre-written weights (eliminates ~2.5 GB transfers per inference)
    - Resident buffer kernel variants (_res) throughout
    - Single async context for all bottleneck blocks (when async enabled)

    Memory usage: ~280 MB GPU (buffers + weights resident)
    """

    def __init__(self, weights_path: str | None = None):
        """Initialize ResNet50 with resident buffer support."""
        super().__init__(weights_path)
        self.buffers: dict[str, np.ndarray] | None = None
        self._buffers_written = False

    def allocate_buffers(self) -> None:
        """
        Pre-allocate all intermediate activation buffers.

        ResNet50 has significantly fewer parameters than VGG16 (25.6M vs 138M)
        but more intermediate feature maps due to skip connections.

        Buffer allocation:
        - Stem: 1 conv + 1 pool
        - 16 bottleneck blocks: 3 convs + 1 shortcut each
        - Final: 1 global pool + 1 FC

        Total: ~180 MB activation buffers + ~100 MB weights = 280 MB GPU memory
        """
        self.buffers = {
            # Stem
            "stem_conv": np.zeros((64, 112, 112), dtype=np.float32),
            "stem_pool": np.zeros((64, 56, 56), dtype=np.float32),
            # Stage 1: 3 blocks (56×56, 256 channels)
            # Each bottleneck: conv1 (64), conv2 (64), conv3 (256), shortcut (256 if downsample)
            "block0_conv1": np.zeros((64, 56, 56), dtype=np.float32),
            "block0_conv2": np.zeros((64, 56, 56), dtype=np.float32),
            "block0_conv3": np.zeros((256, 56, 56), dtype=np.float32),
            "block0_shortcut": np.zeros((256, 56, 56), dtype=np.float32),
            "block0_out": np.zeros((256, 56, 56), dtype=np.float32),
            "block1_conv1": np.zeros((64, 56, 56), dtype=np.float32),
            "block1_conv2": np.zeros((64, 56, 56), dtype=np.float32),
            "block1_conv3": np.zeros((256, 56, 56), dtype=np.float32),
            "block1_out": np.zeros((256, 56, 56), dtype=np.float32),
            "block2_conv1": np.zeros((64, 56, 56), dtype=np.float32),
            "block2_conv2": np.zeros((64, 56, 56), dtype=np.float32),
            "block2_conv3": np.zeros((256, 56, 56), dtype=np.float32),
            "block2_out": np.zeros((256, 56, 56), dtype=np.float32),
            # Stage 2: 4 blocks (28×28, 512 channels)
            "block3_conv1": np.zeros((128, 56, 56), dtype=np.float32),  # Before stride
            "block3_conv2": np.zeros((128, 28, 28), dtype=np.float32),  # After stride
            "block3_conv3": np.zeros((512, 28, 28), dtype=np.float32),
            "block3_shortcut": np.zeros((512, 28, 28), dtype=np.float32),
            "block3_out": np.zeros((512, 28, 28), dtype=np.float32),
            "block4_conv1": np.zeros((128, 28, 28), dtype=np.float32),
            "block4_conv2": np.zeros((128, 28, 28), dtype=np.float32),
            "block4_conv3": np.zeros((512, 28, 28), dtype=np.float32),
            "block4_out": np.zeros((512, 28, 28), dtype=np.float32),
            "block5_conv1": np.zeros((128, 28, 28), dtype=np.float32),
            "block5_conv2": np.zeros((128, 28, 28), dtype=np.float32),
            "block5_conv3": np.zeros((512, 28, 28), dtype=np.float32),
            "block5_out": np.zeros((512, 28, 28), dtype=np.float32),
            "block6_conv1": np.zeros((128, 28, 28), dtype=np.float32),
            "block6_conv2": np.zeros((128, 28, 28), dtype=np.float32),
            "block6_conv3": np.zeros((512, 28, 28), dtype=np.float32),
            "block6_out": np.zeros((512, 28, 28), dtype=np.float32),
            # Stage 3: 6 blocks (14×14, 1024 channels)
            "block7_conv1": np.zeros((256, 28, 28), dtype=np.float32),  # Before stride
            "block7_conv2": np.zeros((256, 14, 14), dtype=np.float32),  # After stride
            "block7_conv3": np.zeros((1024, 14, 14), dtype=np.float32),
            "block7_shortcut": np.zeros((1024, 14, 14), dtype=np.float32),
            "block7_out": np.zeros((1024, 14, 14), dtype=np.float32),
            "block8_conv1": np.zeros((256, 14, 14), dtype=np.float32),
            "block8_conv2": np.zeros((256, 14, 14), dtype=np.float32),
            "block8_conv3": np.zeros((1024, 14, 14), dtype=np.float32),
            "block8_out": np.zeros((1024, 14, 14), dtype=np.float32),
            "block9_conv1": np.zeros((256, 14, 14), dtype=np.float32),
            "block9_conv2": np.zeros((256, 14, 14), dtype=np.float32),
            "block9_conv3": np.zeros((1024, 14, 14), dtype=np.float32),
            "block9_out": np.zeros((1024, 14, 14), dtype=np.float32),
            "block10_conv1": np.zeros((256, 14, 14), dtype=np.float32),
            "block10_conv2": np.zeros((256, 14, 14), dtype=np.float32),
            "block10_conv3": np.zeros((1024, 14, 14), dtype=np.float32),
            "block10_out": np.zeros((1024, 14, 14), dtype=np.float32),
            "block11_conv1": np.zeros((256, 14, 14), dtype=np.float32),
            "block11_conv2": np.zeros((256, 14, 14), dtype=np.float32),
            "block11_conv3": np.zeros((1024, 14, 14), dtype=np.float32),
            "block11_out": np.zeros((1024, 14, 14), dtype=np.float32),
            "block12_conv1": np.zeros((256, 14, 14), dtype=np.float32),
            "block12_conv2": np.zeros((256, 14, 14), dtype=np.float32),
            "block12_conv3": np.zeros((1024, 14, 14), dtype=np.float32),
            "block12_out": np.zeros((1024, 14, 14), dtype=np.float32),
            # Stage 4: 3 blocks (7×7, 2048 channels)
            "block13_conv1": np.zeros((512, 14, 14), dtype=np.float32),  # Before stride
            "block13_conv2": np.zeros((512, 7, 7), dtype=np.float32),  # After stride
            "block13_conv3": np.zeros((2048, 7, 7), dtype=np.float32),
            "block13_shortcut": np.zeros((2048, 7, 7), dtype=np.float32),
            "block13_out": np.zeros((2048, 7, 7), dtype=np.float32),
            "block14_conv1": np.zeros((512, 7, 7), dtype=np.float32),
            "block14_conv2": np.zeros((512, 7, 7), dtype=np.float32),
            "block14_conv3": np.zeros((2048, 7, 7), dtype=np.float32),
            "block14_out": np.zeros((2048, 7, 7), dtype=np.float32),
            "block15_conv1": np.zeros((512, 7, 7), dtype=np.float32),
            "block15_conv2": np.zeros((512, 7, 7), dtype=np.float32),
            "block15_conv3": np.zeros((2048, 7, 7), dtype=np.float32),
            "block15_out": np.zeros((2048, 7, 7), dtype=np.float32),
            # Final layers
            "global_pool": np.zeros(2048, dtype=np.float32),
            "fc_out": np.zeros(1000, dtype=np.float32),
        }

        # Pre-write all weights to GPU (critical optimization!)
        for _, weight in self.weights.items():
            self.prog.write(weight)

    def _ensure_buffers_on_gpu(self) -> None:
        """Write all buffers to GPU once (lazy initialization)."""
        if self._buffers_written:
            return

        if self.buffers is None:
            self.allocate_buffers()

        # Write all activation buffers to GPU once
        for buf in self.buffers.values():
            self.prog.write(buf)

        self._buffers_written = True

    def _bottleneck_block(
        self,
        x: np.ndarray,
        block_idx: int,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
    ) -> np.ndarray:
        """
        Resident version of bottleneck block using pre-allocated buffers.

        Overrides parent method to use GPU-resident buffers and _res kernel variants.

        Args:
            x: Input tensor (buffer reference, already on GPU)
            block_idx: Block index for weight and buffer lookup
            in_channels: Input channels
            mid_channels: Middle channels (bottleneck)
            out_channels: Output channels
            stride: Stride for 3×3 conv (1 or 2)
            downsample: Whether to use projection shortcut

        Returns:
            Output tensor (buffer reference, kept on GPU)
        """
        self._ensure_buffers_on_gpu()

        # Get pre-allocated buffers for this block
        conv1_buf = self.buffers[f"block{block_idx}_conv1"]
        conv2_buf = self.buffers[f"block{block_idx}_conv2"]
        conv3_buf = self.buffers[f"block{block_idx}_conv3"]
        out_buf = self.buffers[f"block{block_idx}_out"]

        # Weights are already on GPU (pre-written)
        conv1_W = self.weights[f"block{block_idx}_conv1_weights"]
        conv1_b = self.weights[f"block{block_idx}_conv1_bias"]
        conv2_W = self.weights[f"block{block_idx}_conv2_weights"]
        conv2_b = self.weights[f"block{block_idx}_conv2_bias"]
        conv3_W = self.weights[f"block{block_idx}_conv3_weights"]
        conv3_b = self.weights[f"block{block_idx}_conv3_bias"]

        # 1×1 conv to reduce channels
        self.prog.conv2d_1x1_res(x, conv1_W, conv1_b, conv1_buf, mid_channels)
        self.prog.relu_activation_res(conv1_buf, conv1_buf)  # In-place ReLU

        # 3×3 conv (with optional stride=2 for downsampling)
        if stride == 2:
            self.prog.conv2d_3x3_stride2_res(
                conv1_buf, conv2_W, conv2_b, conv2_buf, mid_channels
            )
        else:
            self.prog.conv2d_3x3_res(
                conv1_buf, conv2_W, conv2_b, conv2_buf, mid_channels
            )
        self.prog.relu_activation_res(conv2_buf, conv2_buf)  # In-place ReLU

        # 1×1 conv to expand channels
        self.prog.conv2d_1x1_res(conv2_buf, conv3_W, conv3_b, conv3_buf, out_channels)

        # Skip connection
        if downsample:
            # Project input to match output dimensions
            shortcut_buf = self.buffers[f"block{block_idx}_shortcut"]
            proj_W = self.weights[f"block{block_idx}_downsample_weights"]
            proj_b = self.weights[f"block{block_idx}_downsample_bias"]

            if stride == 2:
                self.prog.conv2d_1x1_stride2_res(
                    x, proj_W, proj_b, shortcut_buf, out_channels
                )
            else:
                self.prog.conv2d_1x1_res(x, proj_W, proj_b, shortcut_buf, out_channels)
        else:
            # Identity shortcut - just use input directly
            shortcut_buf = x

        # Add residual
        self.prog.add_tensors_res(conv3_buf, shortcut_buf, out_buf)

        # ReLU after addition
        self.prog.relu_activation_res(out_buf, out_buf)  # In-place ReLU

        return out_buf

    def forward(self, image: np.ndarray, stop_at: str | None = None) -> np.ndarray:
        """
        Forward pass through ResNet50 using resident buffers.

        Overrides parent method to use pre-allocated buffers throughout.
        Supports single async context for all bottleneck blocks when async enabled.

        Args:
            image: Preprocessed image (3, 224, 224)
            stop_at: Optional layer to stop at ('pool' for embeddings)

        Returns:
            Network output at specified layer
        """
        import os

        self._ensure_buffers_on_gpu()

        # Check if async mode is enabled
        use_async = os.environ.get("YAPOCIS_ASYNC", "").lower() in ("1", "true", "yes")

        # Write input image to GPU
        input_buf = np.ascontiguousarray(image)
        self.prog.write(input_buf)

        # Stem: 7×7 conv, stride 2, 64 channels (224×224 → 112×112)
        stem_conv_buf = self.buffers["stem_conv"]
        stem_W = self.weights["stem_weights"]
        stem_b = self.weights["stem_bias"]
        self.prog.conv2d_7x7_stride2_res(input_buf, stem_W, stem_b, stem_conv_buf, 64)
        self.prog.relu_activation_res(stem_conv_buf, stem_conv_buf)

        # Max pool 3×3, stride 2 (112×112 → 56×56)
        stem_pool_buf = self.buffers["stem_pool"]
        self.prog.max_pool_3x3_stride2_res(stem_conv_buf, stem_pool_buf)

        if use_async:
            # Async execution - single context for all 16 bottleneck blocks
            with self.prog.async_mode():
                # Stage 1: 3 blocks (64 → 256 channels, 56×56)
                x = self._bottleneck_block(
                    stem_pool_buf, 0, 64, 64, 256, stride=1, downsample=True
                )
                x = self._bottleneck_block(
                    x, 1, 256, 64, 256, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 2, 256, 64, 256, stride=1, downsample=False
                )

                # Stage 2: 4 blocks (256 → 512 channels, 28×28)
                x = self._bottleneck_block(
                    x, 3, 256, 128, 512, stride=2, downsample=True
                )
                x = self._bottleneck_block(
                    x, 4, 512, 128, 512, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 5, 512, 128, 512, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 6, 512, 128, 512, stride=1, downsample=False
                )

                # Stage 3: 6 blocks (512 → 1024 channels, 14×14)
                x = self._bottleneck_block(
                    x, 7, 512, 256, 1024, stride=2, downsample=True
                )
                x = self._bottleneck_block(
                    x, 8, 1024, 256, 1024, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 9, 1024, 256, 1024, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 10, 1024, 256, 1024, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 11, 1024, 256, 1024, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 12, 1024, 256, 1024, stride=1, downsample=False
                )

                # Stage 4: 3 blocks (1024 → 2048 channels, 7×7)
                x = self._bottleneck_block(
                    x, 13, 1024, 512, 2048, stride=2, downsample=True
                )
                x = self._bottleneck_block(
                    x, 14, 2048, 512, 2048, stride=1, downsample=False
                )
                x = self._bottleneck_block(
                    x, 15, 2048, 512, 2048, stride=1, downsample=False
                )

            # Read final conv output
            self.prog.read(x)
        else:
            # Synchronous execution
            x = self._bottleneck_block(
                stem_pool_buf, 0, 64, 64, 256, stride=1, downsample=True
            )
            x = self._bottleneck_block(x, 1, 256, 64, 256, stride=1, downsample=False)
            x = self._bottleneck_block(x, 2, 256, 64, 256, stride=1, downsample=False)

            x = self._bottleneck_block(x, 3, 256, 128, 512, stride=2, downsample=True)
            x = self._bottleneck_block(x, 4, 512, 128, 512, stride=1, downsample=False)
            x = self._bottleneck_block(x, 5, 512, 128, 512, stride=1, downsample=False)
            x = self._bottleneck_block(x, 6, 512, 128, 512, stride=1, downsample=False)

            x = self._bottleneck_block(x, 7, 512, 256, 1024, stride=2, downsample=True)
            x = self._bottleneck_block(
                x, 8, 1024, 256, 1024, stride=1, downsample=False
            )
            x = self._bottleneck_block(
                x, 9, 1024, 256, 1024, stride=1, downsample=False
            )
            x = self._bottleneck_block(
                x, 10, 1024, 256, 1024, stride=1, downsample=False
            )
            x = self._bottleneck_block(
                x, 11, 1024, 256, 1024, stride=1, downsample=False
            )
            x = self._bottleneck_block(
                x, 12, 1024, 256, 1024, stride=1, downsample=False
            )

            x = self._bottleneck_block(
                x, 13, 1024, 512, 2048, stride=2, downsample=True
            )
            x = self._bottleneck_block(
                x, 14, 2048, 512, 2048, stride=1, downsample=False
            )
            x = self._bottleneck_block(
                x, 15, 2048, 512, 2048, stride=1, downsample=False
            )

        # Global average pooling
        pool_buf = self.buffers["global_pool"]
        self.prog.global_avg_pool_res(x, pool_buf)

        if stop_at == "pool":
            self.prog.read(pool_buf)
            return pool_buf.copy()

        # Fully connected layer
        fc_buf = self.buffers["fc_out"]
        fc_W = self.weights["fc_weights"]
        fc_b = self.weights["fc_bias"]
        self.prog.fully_connected_res(pool_buf, fc_W, fc_b, fc_buf)

        # Softmax
        self.prog.softmax_res(fc_buf, fc_buf)

        # Read final result
        self.prog.read(fc_buf)
        return fc_buf.copy()
