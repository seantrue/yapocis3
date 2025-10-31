"""
Functional tests for VGG16 kernels.

Tests each kernel with simple, verifiable inputs to ensure correctness.
"""

import pytest
import numpy as np
import os
from yapocis import core
from yapocis.core import get_backends


# Check if Metal backend is available (VGG16 kernels are Metal-only for now)
BACKENDS = get_backends()
HAS_METAL = 'metal' in BACKENDS

# Skip all tests if Metal not available
pytestmark = pytest.mark.skipif(not HAS_METAL, reason="VGG16 kernels require Metal backend")


@pytest.fixture(scope="module")
def vgg16_program():
    """Load VGG16 program once for all tests."""
    os.environ['YAPOCIS_BACKEND'] = 'metal'
    return core.load_program("vgg16")


class TestActivations:
    """Test activation function kernels."""

    def test_relu_activation_basic(self, vgg16_program):
        """Test ReLU activation: max(0, x)"""
        # Simple test with known values
        input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)

        output = vgg16_program.relu_activation(input_data)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_relu_activation_large(self, vgg16_program):
        """Test ReLU with larger array."""
        input_data = np.random.randn(1000).astype(np.float32)
        expected = np.maximum(0, input_data)

        output = vgg16_program.relu_activation(input_data)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_relu_activation_3d(self, vgg16_program):
        """Test ReLU with 3D tensor (channels, height, width)."""
        input_data = np.random.randn(3, 4, 4).astype(np.float32)
        expected = np.maximum(0, input_data)

        # Flatten for kernel
        input_flat = input_data.reshape(-1)
        output_flat = vgg16_program.relu_activation(input_flat)
        output = output_flat.reshape(3, 4, 4)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_softmax_simple(self, vgg16_program):
        """Test softmax with simple values."""
        # Simple 3-class example
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        output = vgg16_program.softmax(input_data)

        # Verify properties
        assert output.shape == (3,)
        assert np.all(output >= 0)  # All non-negative
        assert np.all(output <= 1)  # All <= 1
        np.testing.assert_almost_equal(np.sum(output), 1.0, decimal=5)  # Sum to 1

        # Verify monotonicity (larger input -> larger output)
        assert output[2] > output[1] > output[0]

    def test_softmax_numerical_stability(self, vgg16_program):
        """Test softmax with large values (numerical stability)."""
        # Large values that would overflow without max subtraction
        input_data = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)

        output = vgg16_program.softmax(input_data)

        # Should still work (no NaN/Inf)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        np.testing.assert_almost_equal(np.sum(output), 1.0, decimal=5)


class TestPooling:
    """Test pooling layer kernels."""

    def test_max_pool_2x2_simple(self, vgg16_program):
        """Test 2x2 max pooling with simple pattern."""
        # Create 4x4 input with known max values in each 2x2 block
        # Block layout:
        #   [1 0]  [3 0]
        #   [0 0]  [0 0]
        #
        #   [5 0]  [7 0]
        #   [0 0]  [0 0]
        input_data = np.array([
            [1, 0, 3, 0],
            [0, 0, 0, 0],
            [5, 0, 7, 0],
            [0, 0, 0, 0]
        ], dtype=np.float32)

        # Single channel: reshape to (1, 4, 4)
        input_3d = input_data.reshape(1, 4, 4)

        # Pre-allocate output (pooling halves spatial dimensions)
        output_shape = (1, 2, 2)  # (channels, height/2, width/2)
        output = np.zeros(output_shape, dtype=np.float32)

        # Dimensions auto-derived from input shape (channels, height, width)
        # Global size must match output shape for pooling kernels
        vgg16_program.max_pool_2x2(input_3d, output, global_size=(2, 2, 1))

        # Expected output: 2x2 with max from each block
        expected = np.array([
            [1, 3],
            [5, 7]
        ], dtype=np.float32).reshape(1, 2, 2)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_max_pool_2x2_multi_channel(self, vgg16_program):
        """Test 2x2 max pooling with multiple channels."""
        # 2 channels, 4x4 each
        input_data = np.array([
            # Channel 0
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]],
            # Channel 1
            [[16, 15, 14, 13],
             [12, 11, 10, 9],
             [8, 7, 6, 5],
             [4, 3, 2, 1]]
        ], dtype=np.float32)

        # Pre-allocate output
        output = np.zeros((2, 2, 2), dtype=np.float32)
        # Global size must match output shape (height, width, channels)
        vgg16_program.max_pool_2x2(input_data, output, global_size=(2, 2, 2))

        # Expected: max of each 2x2 block per channel
        expected = np.array([
            # Channel 0
            [[6, 8],
             [14, 16]],
            # Channel 1
            [[16, 14],
             [8, 6]]
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_max_pool_2x2_negative_values(self, vgg16_program):
        """Test max pooling with negative values."""
        input_data = np.array([
            [[-1, -2, -3, -4],
             [-5, -6, -7, -8],
             [-9, -10, -11, -12],
             [-13, -14, -15, -16]]
        ], dtype=np.float32)

        # Pre-allocate output
        output = np.zeros((1, 2, 2), dtype=np.float32)
        # Global size must match output shape
        vgg16_program.max_pool_2x2(input_data, output, global_size=(2, 2, 1))

        # Max of negative values
        expected = np.array([
            [[-1, -3],
             [-9, -11]]
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)


class TestFullyConnected:
    """Test fully connected layer kernels."""

    def test_fully_connected_simple(self, vgg16_program):
        """Test fully connected with simple 2x2 example."""
        # Input: 2 features
        input_data = np.array([1.0, 2.0], dtype=np.float32)

        # Weights: 3 output neurons, 2 input features
        # Each row is one neuron's weights
        weights = np.array([
            [1.0, 0.0],  # Neuron 0: picks first input
            [0.0, 1.0],  # Neuron 1: picks second input
            [1.0, 1.0],  # Neuron 2: sums both inputs
        ], dtype=np.float32)

        # Bias: one per output neuron
        bias = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Pre-allocate output
        output = np.zeros(3, dtype=np.float32)
        # Global size must match output size
        vgg16_program.fully_connected(input_data, weights, bias, output, global_size=(3,))

        # Expected: Wx + b
        # [1, 0] · [1, 2] + 0.5 = 1 + 0.5 = 1.5
        # [0, 1] · [1, 2] + 0.5 = 2 + 0.5 = 2.5
        # [1, 1] · [1, 2] + 0.5 = 3 + 0.5 = 3.5
        expected = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_fully_connected_identity(self, vgg16_program):
        """Test fully connected with identity matrix."""
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Identity matrix: output = input
        weights = np.eye(3, dtype=np.float32)
        bias = np.zeros(3, dtype=np.float32)

        # Pre-allocate output
        output = np.zeros(3, dtype=np.float32)
        # Global size must match output size
        vgg16_program.fully_connected(input_data, weights, bias, output, global_size=(3,))

        np.testing.assert_array_almost_equal(output, input_data, decimal=5)

    def test_fully_connected_zero_bias(self, vgg16_program):
        """Test fully connected with zero bias."""
        input_data = np.array([2.0, 3.0], dtype=np.float32)

        weights = np.array([
            [1.0, 2.0],  # 1*2 + 2*3 = 8
            [3.0, 4.0],  # 3*2 + 4*3 = 18
        ], dtype=np.float32)

        bias = np.zeros(2, dtype=np.float32)

        # Pre-allocate output
        output = np.zeros(2, dtype=np.float32)
        # Global size must match output size
        vgg16_program.fully_connected(input_data, weights, bias, output, global_size=(2,))

        expected = np.array([8.0, 18.0], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)


class TestMatrixMultiply:
    """Test matrix multiplication kernels."""

    def test_matrix_multiply_simd_simple(self, vgg16_program):
        """Test SIMD matrix multiply with simple 2x2 example."""
        # A: 2x2, B: 2x3, C: 2x3
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=np.float32)

        B = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float32)

        bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Pre-allocate output (M×N matrix) - shape is (2, 3)
        output = np.zeros((2, 3), dtype=np.float32)
        # M, N, K auto-derived from array shapes
        # Global size automatically computed from dispatch_size trait: (M, N) = (2, 3)
        vgg16_program.matrix_multiply_simd(A, B, bias, output)

        # Manual calculation:
        # C[0,0] = 1*1 + 2*4 + 0.1 = 9.1
        # C[0,1] = 1*2 + 2*5 + 0.2 = 12.2
        # C[0,2] = 1*3 + 2*6 + 0.3 = 15.3
        # C[1,0] = 3*1 + 4*4 + 0.1 = 19.1
        # C[1,1] = 3*2 + 4*5 + 0.2 = 26.2
        # C[1,2] = 3*3 + 4*6 + 0.3 = 33.3
        expected = np.array([
            [9.1, 12.2, 15.3],
            [19.1, 26.2, 33.3]
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_matrix_multiply_simd_identity(self, vgg16_program):
        """Test matrix multiply with identity matrix."""
        A = np.eye(3, dtype=np.float32)
        B = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], dtype=np.float32)
        bias = np.zeros(2, dtype=np.float32)

        # Pre-allocate output (M×N matrix) - shape is (3, 2)
        output = np.zeros((3, 2), dtype=np.float32)
        # M, N, K auto-derived from array shapes
        # Global size automatically computed from dispatch_size trait: (M, N) = (3, 2)
        vgg16_program.matrix_multiply_simd(A, B, bias, output)

        # Identity * B = B
        expected = B

        np.testing.assert_array_almost_equal(output, expected, decimal=5)


class TestConvolution:
    """Test convolution kernels."""

    def test_conv2d_3x3_identity(self, vgg16_program):
        """Test 3x3 convolution with identity kernel."""
        # 4x4 input, single channel
        input_data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.float32).reshape(1, 4, 4)

        # Identity kernel: center weight = 1, others = 0
        weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
        weights[0, 0, 1, 1] = 1.0  # Center pixel

        bias = np.zeros(1, dtype=np.float32)

        # Pre-allocate output (same spatial size due to padding, different channels)
        output = np.zeros((1, 4, 4), dtype=np.float32)
        # Global size must be (height, width, channels) for output shape (C, H, W)
        vgg16_program.conv2d_3x3(input_data, weights, bias, output, 1, global_size=(4, 4, 1))

        # With padding=1, output should be same as input
        np.testing.assert_array_almost_equal(output, input_data, decimal=5)

    def test_conv2d_3x3_uniform_filter(self, vgg16_program):
        """Test 3x3 convolution with uniform (averaging) filter."""
        # 4x4 input, single channel - all ones
        input_data = np.ones((1, 4, 4), dtype=np.float32)

        # Uniform 3x3 filter (average)
        weights = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0
        bias = np.zeros(1, dtype=np.float32)

        # Pre-allocate output (same spatial size due to padding, different channels)
        output = np.zeros((1, 4, 4), dtype=np.float32)
        # Global size must be (height, width, channels) for output shape (C, H, W)
        vgg16_program.conv2d_3x3(input_data, weights, bias, output, 1, global_size=(4, 4, 1))

        # Edges have fewer neighbors due to padding
        # But interior should be 1.0 (average of nine 1.0s)
        assert output[0, 1, 1] == pytest.approx(1.0, abs=1e-5)
        assert output[0, 2, 2] == pytest.approx(1.0, abs=1e-5)

    def test_conv2d_3x3_horizontal_edge(self, vgg16_program):
        """Test 3x3 convolution with horizontal edge detector."""
        # Create horizontal edge: white top, black bottom
        input_data = np.zeros((1, 4, 4), dtype=np.float32)
        input_data[0, 0:2, :] = 1.0  # Top two rows white
        input_data[0, 2:4, :] = 0.0  # Bottom two rows black

        # Horizontal Sobel-like edge detector
        weights = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float32).reshape(1, 1, 3, 3)

        bias = np.zeros(1, dtype=np.float32)

        # Pre-allocate output (same spatial size due to padding, different channels)
        output = np.zeros((1, 4, 4), dtype=np.float32)
        # Global size must be (height, width, channels) for output shape (C, H, W)
        vgg16_program.conv2d_3x3(input_data, weights, bias, output, 1, global_size=(4, 4, 1))

        # Strong response at the edge (row 1-2 boundary)
        # Center of edge should have high magnitude
        assert abs(output[0, 1, 1]) > 0.5  # Significant edge response

    def test_conv2d_3x3_multi_channel(self, vgg16_program):
        """Test 3x3 convolution with multiple input and output channels."""
        # 2 input channels, 4x4 each
        input_data = np.random.randn(2, 4, 4).astype(np.float32)

        # 3 output channels, 2 input channels, 3x3 kernel each
        weights = np.random.randn(3, 2, 3, 3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)

        # Pre-allocate output
        output = np.zeros((3, 4, 4), dtype=np.float32)
        # Global size: (height, width, channels) for output (3, 4, 4)
        vgg16_program.conv2d_3x3(input_data, weights, bias, output, 3, global_size=(4, 4, 3))

        # Check output shape
        assert output.shape == (3, 4, 4)

        # Check no NaN/Inf
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_conv2d_3x3_vs_optimized(self, vgg16_program):
        """Test that basic and optimized convolution produce same results."""
        # Random input
        input_data = np.random.randn(3, 8, 8).astype(np.float32)
        weights = np.random.randn(4, 3, 3, 3).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)

        # Pre-allocate outputs
        output_basic = np.zeros((4, 8, 8), dtype=np.float32)
        output_optimized = np.zeros((4, 8, 8), dtype=np.float32)

        # Global size: (height, width, channels) for output (4, 8, 8)
        # Run basic version (dimensions auto-derived)
        vgg16_program.conv2d_3x3(input_data, weights, bias, output_basic, 4, global_size=(8, 8, 4))

        # Run optimized version (dimensions auto-derived)
        vgg16_program.conv2d_3x3_optimized(input_data, weights, bias, output_optimized, 4, global_size=(8, 8, 4))

        # Should produce identical results
        np.testing.assert_array_almost_equal(
            output_basic, output_optimized,
            decimal=4,  # Allow small floating point differences
            err_msg="Basic and optimized convolution should match"
        )


class TestBatchNorm:
    """Test batch normalization kernels."""

    def test_batch_norm_standardize(self, vgg16_program):
        """Test batch norm standardization (mean=0, var=1)."""
        # Single channel, 3x3 spatial
        input_data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32).reshape(1, 3, 3)

        # Learned parameters
        gamma = np.array([1.0], dtype=np.float32)  # Scale = 1
        beta = np.array([0.0], dtype=np.float32)   # Shift = 0

        # Statistics (pre-computed for this channel)
        mean = np.array([5.0], dtype=np.float32)   # Mean of 1-9
        variance = np.array([6.666667], dtype=np.float32)  # Var of 1-9

        epsilon = 1e-5

        # Dimensions auto-derived from input shape
        # Global size: (height, width, channels) matching input shape
        output = vgg16_program.batch_norm(
            input_data, gamma, beta, mean, variance, epsilon,
            global_size=(input_data.shape[1], input_data.shape[2], input_data.shape[0])
        )

        # Output should be standardized: (x - mean) / sqrt(var + eps)
        expected = (input_data - mean[0]) / np.sqrt(variance[0] + epsilon)

        np.testing.assert_array_almost_equal(output, expected, decimal=4)

    def test_batch_norm_scale_shift(self, vgg16_program):
        """Test batch norm with scale and shift."""
        input_data = np.ones((1, 2, 2), dtype=np.float32) * 5.0

        # Transform: 2x + 3
        gamma = np.array([2.0], dtype=np.float32)
        beta = np.array([3.0], dtype=np.float32)

        # Zero-centered, unit variance
        mean = np.array([5.0], dtype=np.float32)
        variance = np.array([0.0], dtype=np.float32)

        epsilon = 1e-5

        # Dimensions auto-derived from input shape
        # Global size: (height, width, channels) matching input shape
        output = vgg16_program.batch_norm(
            input_data, gamma, beta, mean, variance, epsilon,
            global_size=(input_data.shape[1], input_data.shape[2], input_data.shape[0])
        )

        # (5 - 5) / sqrt(0 + eps) * 2 + 3 ≈ 3
        expected = np.ones((1, 2, 2), dtype=np.float32) * 3.0

        np.testing.assert_array_almost_equal(output, expected, decimal=3)

    def test_batch_norm_multi_channel(self, vgg16_program):
        """Test batch norm with multiple channels."""
        # 2 channels, 2x2 each
        input_data = np.array([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]
        ], dtype=np.float32)

        # Different parameters per channel
        gamma = np.array([1.0, 2.0], dtype=np.float32)
        beta = np.array([0.5, 1.0], dtype=np.float32)
        mean = np.array([2.5, 6.5], dtype=np.float32)
        variance = np.array([1.25, 1.25], dtype=np.float32)

        epsilon = 1e-5

        # Dimensions auto-derived from input shape
        # Global size: (height, width, channels) matching input shape
        output = vgg16_program.batch_norm(
            input_data, gamma, beta, mean, variance, epsilon,
            global_size=(input_data.shape[1], input_data.shape[2], input_data.shape[0])
        )

        # Check shape preserved
        assert output.shape == (2, 2, 2)

        # Check no NaN/Inf
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestIntegration:
    """Integration tests combining multiple kernels."""

    def test_conv_relu_pool_pipeline(self, vgg16_program):
        """Test typical CNN layer: Conv -> ReLU -> Pool."""
        # Small input: 1 channel, 4x4
        input_data = np.random.randn(1, 4, 4).astype(np.float32)

        # Conv: 1 in channel, 2 out channels
        weights = np.random.randn(2, 1, 3, 3).astype(np.float32)
        bias = np.zeros(2, dtype=np.float32)

        # Step 1: Convolution (pre-allocate output)
        conv_out = np.zeros((2, 4, 4), dtype=np.float32)
        # Global size: (height, width, channels) for output (2, 4, 4)
        vgg16_program.conv2d_3x3(input_data, weights, bias, conv_out, 2, global_size=(4, 4, 2))
        assert conv_out.shape == (2, 4, 4)

        # Step 2: ReLU
        conv_out_flat = conv_out.reshape(-1)
        relu_out_flat = vgg16_program.relu_activation(conv_out_flat)
        relu_out = relu_out_flat.reshape(2, 4, 4)

        # All values should be non-negative after ReLU
        assert np.all(relu_out >= 0)

        # Step 3: Max Pool (pre-allocate output)
        pool_out = np.zeros((2, 2, 2), dtype=np.float32)
        # Global size must match output shape for pooling
        vgg16_program.max_pool_2x2(relu_out, pool_out, global_size=(2, 2, 2))
        assert pool_out.shape == (2, 2, 2)

        # No NaN/Inf through pipeline
        assert not np.any(np.isnan(pool_out))
        assert not np.any(np.isinf(pool_out))

    def test_fc_relu_fc_pipeline(self, vgg16_program):
        """Test typical classifier: FC -> ReLU -> FC."""
        # Input features
        input_data = np.random.randn(10).astype(np.float32)

        # First FC layer: 10 -> 5
        weights1 = np.random.randn(5, 10).astype(np.float32)
        bias1 = np.zeros(5, dtype=np.float32)

        # Pre-allocate output
        fc1_out = np.zeros(5, dtype=np.float32)
        # Global size must match output size
        vgg16_program.fully_connected(input_data, weights1, bias1, fc1_out, global_size=(5,))
        assert fc1_out.shape == (5,)

        # ReLU
        relu_out = vgg16_program.relu_activation(fc1_out)
        assert np.all(relu_out >= 0)

        # Second FC layer: 5 -> 3
        weights2 = np.random.randn(3, 5).astype(np.float32)
        bias2 = np.zeros(3, dtype=np.float32)

        # Pre-allocate output
        fc2_out = np.zeros(3, dtype=np.float32)
        # Global size must match output size
        vgg16_program.fully_connected(relu_out, weights2, bias2, fc2_out, global_size=(3,))
        assert fc2_out.shape == (3,)

        # Softmax for probabilities
        probs = vgg16_program.softmax(fc2_out)
        assert probs.shape == (3,)
        np.testing.assert_almost_equal(np.sum(probs), 1.0, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
