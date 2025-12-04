"""
VGG16 neural network implementation using yapocis CNN kernels.

Provides classification and embedding extraction for the VGG16 architecture.
Supports GPU-resident buffers for efficient inference.
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

from __future__ import annotations

import os

import numpy as np

from yapocis import core


class VGG16:
    """
    VGG16 deep convolutional neural network.

    Architecture:
        - Block 1: conv3-64 x2, maxpool
        - Block 2: conv3-128 x2, maxpool
        - Block 3: conv3-256 x3, maxpool
        - Block 4: conv3-512 x3, maxpool
        - Block 5: conv3-512 x3, maxpool
        - FC6: 4096 (embeddings)
        - FC7: 4096
        - FC8: 1000 (ImageNet classes)

    Usage:
        # Classification
        vgg = VGG16()
        vgg.load_weights('vgg16_weights.npz')
        probs = vgg.predict(image)

        # Embedding extraction
        embedding = vgg.extract_features(image)
    """

    def __init__(self, weights_path: str | None = None):
        """
        Initialize VGG16 network.

        Args:
            weights_path: Path to weights file (.npz format). If None, must call load_weights() later.
        """
        self.prog = core.load_program("cnn")
        self.weights: dict | None = None

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path: str) -> None:
        """
        Load VGG16 weights from npz file.

        Expected format: npz file with keys like:
            'conv0_weights', 'conv0_bias', ..., 'conv12_weights', 'conv12_bias'
            'fc0_weights', 'fc0_bias', 'fc1_weights', 'fc1_bias', 'fc2_weights', 'fc2_bias'

        Args:
            path: Path to weights .npz file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")

        self.weights = np.load(path)
        print(f"Loaded VGG16 weights from {path}")
        print(f"Available weight keys: {len(self.weights.files)} tensors")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for VGG16 input.

        Args:
            image: Input image (H, W, 3) in range [0, 255] or [0, 1]

        Returns:
            Preprocessed image (3, 224, 224) in VGG16 format
        """
        # Handle different input formats
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        # Resize to 224x224 if needed
        if image.shape[:2] != (224, 224):
            try:
                from PIL import Image

                pil_image = Image.fromarray((image * 255).astype(np.uint8))
                pil_image = pil_image.resize((224, 224), Image.BILINEAR)
                image = np.array(pil_image).astype(np.float32) / 255.0
            except ImportError:
                # Fallback to simple crop/pad if PIL not available
                h, w = image.shape[:2]
                if h != 224 or w != 224:
                    # Center crop or pad
                    new_img = np.zeros((224, 224, 3), dtype=np.float32)
                    start_h = max(0, (h - 224) // 2)
                    start_w = max(0, (w - 224) // 2)
                    end_h = min(h, start_h + 224)
                    end_w = min(w, start_w + 224)

                    dst_h = end_h - start_h
                    dst_w = end_w - start_w
                    new_img[:dst_h, :dst_w] = image[start_h:end_h, start_w:end_w]
                    image = new_img

        # Convert to channel-first format (3, 224, 224)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))

        # VGG16 preprocessing: ImageNet normalization (PyTorch style)
        # Mean and std values for RGB channels
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image[0] = (image[0] - mean[0]) / std[0]  # R
        image[1] = (image[1] - mean[1]) / std[1]  # G
        image[2] = (image[2] - mean[2]) / std[2]  # B

        # Ensure contiguous for OpenCL
        return np.ascontiguousarray(image.astype(np.float32))

    def _check_weights(self) -> None:
        """Check that weights are loaded."""
        if self.weights is None:
            raise RuntimeError("Weights not loaded. Call load_weights() first.")

    def _get_conv_layer_index(self, block_num: int, layer_in_block: int) -> int:
        """
        Map block/layer to global conv layer index.

        VGG16 has 13 conv layers total (conv0-conv12):
        - Block 1: conv0, conv1 (2 layers)
        - Block 2: conv2, conv3 (2 layers)
        - Block 3: conv4, conv5, conv6 (3 layers)
        - Block 4: conv7, conv8, conv9 (3 layers)
        - Block 5: conv10, conv11, conv12 (3 layers)
        """
        layer_offsets = [0, 2, 4, 7, 10]  # Starting index for each block
        return layer_offsets[block_num - 1] + layer_in_block - 1

    def _conv_block(
        self,
        input_tensor: np.ndarray,
        block_num: int,
        num_layers: int,
        out_channels: int,
    ) -> np.ndarray:
        """
        Execute a VGG convolutional block.

        Args:
            input_tensor: Input feature map (C, H, W)
            block_num: Block number (1-5)
            num_layers: Number of conv layers in block
            out_channels: Number of output channels

        Returns:
            Output feature map after convolutions and pooling
        """
        # Check if async mode is enabled
        use_async = os.environ.get("YAPOCIS_ASYNC", "").lower() in ("1", "true", "yes")

        if use_async:
            return self._conv_block_async(
                input_tensor, block_num, num_layers, out_channels
            )

        # Synchronous path (original implementation)
        x = np.ascontiguousarray(input_tensor)
        C_in, H, W = x.shape

        for i in range(1, num_layers + 1):
            conv_idx = self._get_conv_layer_index(block_num, i)
            weights = np.ascontiguousarray(self.weights[f"conv{conv_idx}_weights"])
            bias = np.ascontiguousarray(self.weights[f"conv{conv_idx}_bias"])

            # Allocate contiguous output buffer
            output_buf = np.ascontiguousarray(
                np.zeros((out_channels, H, W), dtype=np.float32)
            )

            # 3x3 convolution
            x = self.prog.conv2d_3x3(x, weights, bias, output_buf, out_channels)

            # ReLU activation (in-place to avoid copy)
            x = self.prog.relu_activation(x)

            C_in = out_channels

        # Max pooling 2x2
        H_out = H // 2
        W_out = W // 2
        pool_buf = np.ascontiguousarray(
            np.zeros((out_channels, H_out, W_out), dtype=np.float32)
        )
        x = self.prog.max_pool_2x2(x, pool_buf)

        return x

    def _conv_block_async(
        self,
        input_tensor: np.ndarray,
        block_num: int,
        num_layers: int,
        out_channels: int,
    ) -> np.ndarray:
        """
        Execute a VGG convolutional block using async execution.

        Uses resident buffers to eliminate synchronization overhead between kernels.
        Expected speedup: 1.08-1.12x per block.

        Args:
            input_tensor: Input feature map (C, H, W)
            block_num: Block number (1-5)
            num_layers: Number of conv layers in block
            out_channels: Number of output channels

        Returns:
            Output feature map after convolutions and pooling
        """
        C_in, H, W = input_tensor.shape

        # Pre-allocate all intermediate buffers for this block
        # For num_layers convolutions, we need:
        # - conv output buffers (one per layer)
        # - relu output buffers (one per layer)
        # - pool output buffer (one for the block)
        conv_buffers = []
        relu_buffers = []

        for i in range(num_layers):
            conv_buffers.append(
                np.ascontiguousarray(np.zeros((out_channels, H, W), dtype=np.float32))
            )
            relu_buffers.append(
                np.ascontiguousarray(np.zeros((out_channels, H, W), dtype=np.float32))
            )

        H_out = H // 2
        W_out = W // 2
        pool_buf = np.ascontiguousarray(
            np.zeros((out_channels, H_out, W_out), dtype=np.float32)
        )

        # Ensure input is contiguous
        input_buf = np.ascontiguousarray(input_tensor)

        # Write all buffers to GPU
        self.prog.write(input_buf)
        for buf in conv_buffers + relu_buffers:
            self.prog.write(buf)
        self.prog.write(pool_buf)

        # Async execution - enqueue all operations without synchronization
        with self.prog.async_mode():
            current_input = input_buf

            for i in range(num_layers):
                conv_idx = self._get_conv_layer_index(block_num, i + 1)
                # Note: weights and biases are assumed to already be resident on GPU
                # They are passed as-is without writing
                weights = self.weights[f"conv{conv_idx}_weights"]
                bias = self.weights[f"conv{conv_idx}_bias"]

                # Use resident buffer variants to avoid synchronization
                self.prog.conv2d_3x3_res(
                    current_input, weights, bias, conv_buffers[i], out_channels
                )
                self.prog.relu_activation_res(conv_buffers[i], relu_buffers[i])

                # Next layer uses this layer's relu output
                current_input = relu_buffers[i]

            # Max pooling on final relu output
            self.prog.max_pool_2x2_res(relu_buffers[-1], pool_buf)

        # Single synchronization point - read final result
        self.prog.read(pool_buf)
        return pool_buf

    def forward(self, image: np.ndarray, stop_at: str | None = None) -> np.ndarray:
        """
        Forward pass through VGG16 network.

        Args:
            image: Preprocessed image (3, 224, 224)
            stop_at: Optional layer name to stop at ('fc6', 'fc7', 'fc8', or None for full network)

        Returns:
            Network output at specified layer
        """
        self._check_weights()

        # Block 1: 64 channels (conv0, conv1)
        x = self._conv_block(image, 1, 2, 64)  # (64, 112, 112)

        # Block 2: 128 channels (conv2, conv3)
        x = self._conv_block(x, 2, 2, 128)  # (128, 56, 56)

        # Block 3: 256 channels (conv4, conv5, conv6)
        x = self._conv_block(x, 3, 3, 256)  # (256, 28, 28)

        # Block 4: 512 channels (conv7, conv8, conv9)
        x = self._conv_block(x, 4, 3, 512)  # (512, 14, 14)

        # Block 5: 512 channels (conv10, conv11, conv12)
        x = self._conv_block(x, 5, 3, 512)  # (512, 7, 7)

        # Flatten
        x = np.ascontiguousarray(x.reshape(-1))  # (512 * 7 * 7 = 25088,)

        # FC6 (fc0)
        fc6_W = np.ascontiguousarray(self.weights["fc0_weights"])  # (4096, 25088)
        fc6_b = np.ascontiguousarray(self.weights["fc0_bias"])  # (4096,)
        fc6_out = np.ascontiguousarray(np.zeros(4096, dtype=np.float32))
        x = self.prog.fully_connected(x, fc6_W, fc6_b, fc6_out)
        x = self.prog.relu_activation(x)

        if stop_at == "fc6":
            return x

        # FC7 (fc1)
        fc7_W = np.ascontiguousarray(self.weights["fc1_weights"])  # (4096, 4096)
        fc7_b = np.ascontiguousarray(self.weights["fc1_bias"])  # (4096,)
        fc7_out = np.ascontiguousarray(np.zeros(4096, dtype=np.float32))
        x = self.prog.fully_connected(x, fc7_W, fc7_b, fc7_out)
        x = self.prog.relu_activation(x)

        if stop_at == "fc7":
            return x

        # FC8 (fc2)
        fc8_W = np.ascontiguousarray(self.weights["fc2_weights"])  # (1000, 4096)
        fc8_b = np.ascontiguousarray(self.weights["fc2_bias"])  # (1000,)
        fc8_out = np.ascontiguousarray(np.zeros(1000, dtype=np.float32))
        x = self.prog.fully_connected(x, fc8_W, fc8_b, fc8_out)

        # Softmax for probabilities
        x = self.prog.softmax(x)

        return x

    def predict(
        self, image: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify an image and return top predictions.

        Args:
            image: Input image (H, W, 3) in range [0, 255] or [0, 1]
            top_k: Number of top predictions to return

        Returns:
            Tuple of (class_indices, probabilities) for top K predictions
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)

        # Forward pass
        probs = self.forward(preprocessed)

        # Get top K predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_probs = probs[top_indices]

        return top_indices, top_probs

    def extract_features(self, image: np.ndarray, layer: str = "fc6") -> np.ndarray:
        """
        Extract feature embeddings from FC6 or FC7 layer.

        Args:
            image: Input image (H, W, 3) in range [0, 255] or [0, 1]
            layer: Layer to extract features from ('fc6' or 'fc7')

        Returns:
            Feature vector (4096-dimensional)
        """
        if layer not in ["fc6", "fc7"]:
            raise ValueError(f"Layer must be 'fc6' or 'fc7', got: {layer}")

        # Preprocess
        preprocessed = self.preprocess_image(image)

        # Forward pass to specified layer
        features = self.forward(preprocessed, stop_at=layer)

        return features


class VGG16Resident(VGG16):
    """
    VGG16 with GPU-resident buffers for maximum performance.

    This version keeps all intermediate activations on GPU to avoid
    unnecessary data transfers. Pre-allocates all buffers and uses
    resident buffer kernels (_res variants) throughout.

    Provides better throughput for batch inference and works seamlessly
    with async execution mode.
    """

    def __init__(self, weights_path: str | None = None):
        super().__init__(weights_path)
        self.buffers: dict | None = None
        self._buffers_written = False

    def allocate_buffers(self) -> None:
        """Pre-allocate all intermediate activation buffers."""
        self.buffers = {
            # Conv blocks - using naming: conv{block}_{layer}
            "conv1_1": np.zeros((64, 224, 224), dtype=np.float32),
            "conv1_2": np.zeros((64, 224, 224), dtype=np.float32),
            "pool1": np.zeros((64, 112, 112), dtype=np.float32),
            "conv2_1": np.zeros((128, 112, 112), dtype=np.float32),
            "conv2_2": np.zeros((128, 112, 112), dtype=np.float32),
            "pool2": np.zeros((128, 56, 56), dtype=np.float32),
            "conv3_1": np.zeros((256, 56, 56), dtype=np.float32),
            "conv3_2": np.zeros((256, 56, 56), dtype=np.float32),
            "conv3_3": np.zeros((256, 56, 56), dtype=np.float32),
            "pool3": np.zeros((256, 28, 28), dtype=np.float32),
            "conv4_1": np.zeros((512, 28, 28), dtype=np.float32),
            "conv4_2": np.zeros((512, 28, 28), dtype=np.float32),
            "conv4_3": np.zeros((512, 28, 28), dtype=np.float32),
            "pool4": np.zeros((512, 14, 14), dtype=np.float32),
            "conv5_1": np.zeros((512, 14, 14), dtype=np.float32),
            "conv5_2": np.zeros((512, 14, 14), dtype=np.float32),
            "conv5_3": np.zeros((512, 14, 14), dtype=np.float32),
            "pool5": np.zeros((512, 7, 7), dtype=np.float32),
            # FC layers
            "flatten": np.zeros(25088, dtype=np.float32),
            "fc6": np.zeros(4096, dtype=np.float32),
            "fc7": np.zeros(4096, dtype=np.float32),
            "fc8": np.zeros(1000, dtype=np.float32),
        }
        for _, weight in self.weights.items():
            self.prog.write(weight)

    def _ensure_buffers_on_gpu(self) -> None:
        """Write all buffers to GPU once (lazy initialization)."""
        if self._buffers_written:
            return

        if self.buffers is None:
            self.allocate_buffers()

        # Write all buffers to GPU once
        for buf in self.buffers.values():
            self.prog.write(buf)

        self._buffers_written = True

    def _conv_block(
        self,
        input_tensor: np.ndarray,
        block_num: int,
        num_layers: int,
        out_channels: int,
    ) -> np.ndarray:
        """
        Execute a VGG convolutional block using pre-allocated resident buffers.

        Overrides parent method to use VGG16Resident's pre-allocated buffers.
        This method runs synchronously - for async mode, forward() wraps all blocks
        in a single async context.

        Args:
            input_tensor: Input feature map (C, H, W)
            block_num: Block number (1-5)
            num_layers: Number of conv layers in block
            out_channels: Number of output channels

        Returns:
            Output feature map after convolutions and pooling
        """
        self._ensure_buffers_on_gpu()

        # Get buffer names for this block
        buffer_names = []
        for i in range(1, num_layers + 1):
            buffer_names.append(f"conv{block_num}_{i}")

        pool_buffer_name = f"pool{block_num}"

        # Write input to GPU
        input_buf = np.ascontiguousarray(input_tensor)
        self.prog.write(input_buf)

        # Execute block (synchronous)
        current_input = input_buf

        for i in range(num_layers):
            conv_idx = self._get_conv_layer_index(block_num, i + 1)
            weights = self.weights[f"conv{conv_idx}_weights"]
            bias = self.weights[f"conv{conv_idx}_bias"]

            # Conv and ReLU in-place
            conv_buf = self.buffers[buffer_names[i]]
            self.prog.conv2d_3x3_res(
                current_input, weights, bias, conv_buf, out_channels
            )
            self.prog.relu_activation_res(conv_buf, conv_buf)

            current_input = conv_buf

        # Max pooling
        pool_buf = self.buffers[pool_buffer_name]
        self.prog.max_pool_2x2_res(current_input, pool_buf)

        # Read result
        self.prog.read(pool_buf)
        return pool_buf

    def forward(self, image: np.ndarray, stop_at: str | None = None) -> np.ndarray:
        """
        Forward pass through VGG16 using resident buffers.

        Overrides parent method to use pre-allocated buffers throughout.

        Args:
            image: Preprocessed image (3, 224, 224)
            stop_at: Optional layer name to stop at ('fc6', 'fc7', or None for full network)

        Returns:
            Network output at specified layer
        """
        self._check_weights()
        self._ensure_buffers_on_gpu()

        # Check if async mode is enabled
        use_async = os.environ.get("YAPOCIS_ASYNC", "").lower() in ("1", "true", "yes")

        # Write input image
        input_buf = np.ascontiguousarray(image)
        self.prog.write(input_buf)

        if use_async:
            # Async execution - single context for all conv blocks
            with self.prog.async_mode():
                # Block 1: 64 channels (conv0, conv1)
                x = self._run_conv_block_async(input_buf, 1, 2, 64)

                # Block 2: 128 channels (conv2, conv3)
                x = self._run_conv_block_async(x, 2, 2, 128)

                # Block 3: 256 channels (conv4, conv5, conv6)
                x = self._run_conv_block_async(x, 3, 3, 256)

                # Block 4: 512 channels (conv7, conv8, conv9)
                x = self._run_conv_block_async(x, 4, 3, 512)

                # Block 5: 512 channels (conv10, conv11, conv12)
                x = self._run_conv_block_async(x, 5, 3, 512)

            # Read final conv output
            self.prog.read(x)
        else:
            # Synchronous execution - use _conv_block which handles sync mode
            x = self._conv_block(input_buf, 1, 2, 64)
            x = self._conv_block(x, 2, 2, 128)
            x = self._conv_block(x, 3, 3, 256)
            x = self._conv_block(x, 4, 3, 512)
            x = self._conv_block(x, 5, 3, 512)

        # Flatten
        flatten_buf = self.buffers["flatten"]
        flatten_buf[:] = x.reshape(-1)
        # Write flattened data to GPU
        self.prog.write(flatten_buf)

        # FC layers
        fc6_buf = self.buffers["fc6"]
        fc7_buf = self.buffers["fc7"]
        fc8_buf = self.buffers["fc8"]

        # FC6
        fc6_W = self.weights["fc0_weights"]
        fc6_b = self.weights["fc0_bias"]
        self.prog.fully_connected_res(flatten_buf, fc6_W, fc6_b, fc6_buf)
        self.prog.relu_activation_res(fc6_buf, fc6_buf)

        if stop_at == "fc6":
            self.prog.read(fc6_buf)
            return fc6_buf.copy()

        # FC7
        fc7_W = self.weights["fc1_weights"]
        fc7_b = self.weights["fc1_bias"]
        self.prog.fully_connected_res(fc6_buf, fc7_W, fc7_b, fc7_buf)
        self.prog.relu_activation_res(fc7_buf, fc7_buf)

        if stop_at == "fc7":
            self.prog.read(fc7_buf)
            return fc7_buf.copy()

        # FC8
        fc8_W = self.weights["fc2_weights"]
        fc8_b = self.weights["fc2_bias"]
        self.prog.fully_connected_res(fc7_buf, fc8_W, fc8_b, fc8_buf)

        # Softmax
        self.prog.softmax_res(fc8_buf, fc8_buf)

        # Read final result
        self.prog.read(fc8_buf)
        return fc8_buf.copy()

    def _run_conv_block_async(
        self,
        input_tensor: np.ndarray,
        block_num: int,
        num_layers: int,
        out_channels: int,
    ) -> np.ndarray:
        """
        Helper for async conv block execution within a larger async context.

        This is called from within forward()'s async_mode context, so it doesn't
        create its own async context.

        Returns the buffer reference (not a copy) for chaining.
        """
        # Get buffer names for this block
        buffer_names = []
        for i in range(1, num_layers + 1):
            buffer_names.append(f"conv{block_num}_{i}")

        pool_buffer_name = f"pool{block_num}"

        current_input = input_tensor

        for i in range(num_layers):
            conv_idx = self._get_conv_layer_index(block_num, i + 1)
            weights = self.weights[f"conv{conv_idx}_weights"]
            bias = self.weights[f"conv{conv_idx}_bias"]

            # Conv to buffer, then ReLU in-place
            conv_buf = self.buffers[buffer_names[i]]
            self.prog.conv2d_3x3_res(
                current_input, weights, bias, conv_buf, out_channels
            )
            self.prog.relu_activation_res(conv_buf, conv_buf)

            current_input = conv_buf

        # Max pooling
        pool_buf = self.buffers[pool_buffer_name]
        self.prog.max_pool_2x2_res(current_input, pool_buf)

        return pool_buf


def load_imagenet_labels(path: str = None, model_name: str = "vgg16") -> list[str]:
    """
    Load ImageNet class labels.

    Args:
        path: Path to labels file (one class per line). If None, uses model asset manager.
        model_name: Model name to load labels from (default: 'vgg16')

    Returns:
        List of 1000 class names
    """
    if path is None:
        # Try to get from model asset manager
        try:
            from yapocis.models import get_classes_path

            classes_path = get_classes_path(model_name)
            if classes_path:
                path = str(classes_path)
        except ImportError:
            pass

    if path is None or not os.path.exists(path):
        # Return dummy labels if file not found
        return [f"class_{i}" for i in range(1000)]

    with open(path) as f:
        labels = [line.strip() for line in f]

    return labels
