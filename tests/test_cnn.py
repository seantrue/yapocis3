"""
Unified CNN test suite.

Tests all CNN layers (VGG16 + modern CNN) in a single consolidated package.
Verifies compilation, numerical correctness, and cross-backend consistency.
"""

import numpy as np
import pytest
import os
from yapocis import core
from yapocis.core.manager import get_backends


# =================================================================
# Test Helpers
# =================================================================

def get_test_backends():
    """Get all available backends for testing."""
    return get_backends()


# =================================================================
# Convolution Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_conv2d_3x3_loads(backend):
    """Test that 3x3 convolution kernel compiles."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")
    assert hasattr(prog, 'conv2d_3x3')


@pytest.mark.parametrize("backend", get_test_backends())
def test_conv2d_1x1_loads(backend):
    """Test that 1x1 convolution kernel compiles."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")
    assert hasattr(prog, 'conv2d_1x1')


@pytest.mark.parametrize("backend", get_test_backends())
def test_depthwise_conv2d_3x3_loads(backend):
    """Test that depthwise separable convolution kernel compiles."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")
    assert hasattr(prog, 'depthwise_conv2d_3x3')


# =================================================================
# Activation Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_relu_activation(backend):
    """Test ReLU activation."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    output = prog.relu_activation(input_arr)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("backend", get_test_backends())
def test_gelu_activation(backend):
    """Test GELU activation."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.array([0.0], dtype=np.float32)
    output = prog.gelu(input_arr)

    # GELU(0) should be approximately 0
    assert abs(output[0]) < 1e-6


@pytest.mark.parametrize("backend", get_test_backends())
def test_softmax_activation(backend):
    """Test softmax activation."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    output = prog.softmax(input_arr)

    # Softmax output should sum to 1
    assert abs(output.sum() - 1.0) < 1e-5
    # All values should be positive
    assert np.all(output > 0)


# =================================================================
# Pooling Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_max_pool_2x2(backend):
    """Test 2x2 max pooling."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    # Simple 4x4 input, expect 2x2 output
    input_arr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    output_buf = np.zeros((1, 2, 2), dtype=np.float32)

    output = prog.max_pool_2x2(input_arr, output_buf)

    # Check shape
    assert output.shape == (1, 2, 2)
    # Max pooling should select max from each 2x2 block
    assert output[0, 0, 0] == 5.0  # max of [0,1,4,5]
    assert output[0, 0, 1] == 7.0  # max of [2,3,6,7]


@pytest.mark.parametrize("backend", get_test_backends())
def test_global_avg_pool(backend):
    """Test global average pooling."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.ones((2, 4, 4), dtype=np.float32)
    output_buf = np.zeros(2, dtype=np.float32)

    output = prog.global_avg_pool(input_arr, output_buf)

    assert output.shape == (2,)
    # Average of all ones should be 1
    np.testing.assert_allclose(output, [1.0, 1.0], rtol=1e-5)


@pytest.mark.parametrize("backend", get_test_backends())
def test_global_max_pool(backend):
    """Test global max pooling."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.random.randn(2, 4, 4).astype(np.float32)
    output_buf = np.zeros(2, dtype=np.float32)

    output = prog.global_max_pool(input_arr, output_buf)

    assert output.shape == (2,)
    # Check that output is the max of each channel
    for c in range(2):
        expected_max = input_arr[c].max()
        np.testing.assert_allclose(output[c], expected_max, rtol=1e-5)


# =================================================================
# Normalization Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_batch_norm(backend):
    """Test batch normalization."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    channels = 2
    input_arr = np.random.randn(channels, 4, 4).astype(np.float32)
    gamma = np.ones(channels, dtype=np.float32)
    beta = np.zeros(channels, dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    variance = np.array([1.0, 1.0], dtype=np.float32)
    epsilon = 1e-5

    output = prog.batch_norm(input_arr, gamma, beta, mean, variance, epsilon)

    assert output.shape == input_arr.shape


@pytest.mark.parametrize("backend", get_test_backends())
def test_layer_norm(backend):
    """Test layer normalization."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    channels = 4
    input_arr = np.random.randn(channels, 8, 8).astype(np.float32)
    gamma = np.ones(channels, dtype=np.float32)
    beta = np.zeros(channels, dtype=np.float32)
    epsilon = 1e-5
    output_buf = np.zeros((channels, 8, 8), dtype=np.float32)

    output = prog.layer_norm(input_arr, gamma, beta, output_buf, epsilon)

    assert output.shape == input_arr.shape


# =================================================================
# Fully Connected Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_fully_connected(backend):
    """Test fully connected layer."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_size = 10
    output_size = 5
    input_arr = np.random.randn(input_size).astype(np.float32)
    weights = np.random.randn(output_size, input_size).astype(np.float32)
    bias = np.random.randn(output_size).astype(np.float32)
    output_buf = np.zeros(output_size, dtype=np.float32)

    output = prog.fully_connected(input_arr, weights, bias, output_buf)

    assert output.shape == (output_size,)


# =================================================================
# Regularization Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_dropout_inference_mode(backend):
    """Test dropout in inference mode (should be identity)."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.random.randn(100).astype(np.float32)
    output_buf = np.zeros(100, dtype=np.float32)
    rate = 0.5
    seed = 42
    training = 0  # Inference mode

    output = prog.dropout(input_arr, output_buf, rate, seed, training)

    # In inference mode, output should equal input
    np.testing.assert_array_equal(output, input_arr)


@pytest.mark.parametrize("backend", get_test_backends())
def test_dropout_training_mode(backend):
    """Test dropout in training mode."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    input_arr = np.ones(1000, dtype=np.float32)
    output_buf = np.zeros(1000, dtype=np.float32)
    rate = 0.5
    seed = 42
    training = 1  # Training mode

    output = prog.dropout(input_arr, output_buf, rate, seed, training)

    # Check that some elements are zeroed
    num_zeros = np.sum(output == 0.0)
    assert num_zeros > 0, "Dropout should zero out some elements"

    # Check approximately correct fraction is dropped
    drop_fraction = num_zeros / len(output)
    assert 0.35 < drop_fraction < 0.65, f"Expected ~50% dropout, got {drop_fraction*100:.1f}%"


# =================================================================
# Cross-Backend Consistency Tests
# =================================================================

def test_cross_backend_conv2d_1x1():
    """Test that Metal and OpenCL produce identical results for 1x1 conv."""
    backends = get_test_backends()
    if len(backends) < 2:
        pytest.skip("Need at least 2 backends for cross-backend testing")

    input_arr = np.random.randn(4, 16, 16).astype(np.float32)
    weights = np.random.randn(8, 4, 1, 1).astype(np.float32)
    bias = np.random.randn(8).astype(np.float32)

    outputs = {}
    for backend in backends[:2]:
        os.environ['YAPOCIS_BACKEND'] = backend
        prog = core.load_program("cnn")

        if backend == 'ocl':
            input_arr = np.ascontiguousarray(input_arr)

        output_buf = np.zeros((8, 16, 16), dtype=np.float32)
        outputs[backend] = prog.conv2d_1x1(input_arr, weights, bias, output_buf, 8)

    backend1, backend2 = list(outputs.keys())
    np.testing.assert_allclose(
        outputs[backend1], outputs[backend2],
        rtol=1e-5, atol=1e-6,
        err_msg=f"Cross-backend mismatch: {backend1} vs {backend2}"
    )


def test_cross_backend_gelu():
    """Test that Metal and OpenCL produce identical GELU results."""
    backends = get_test_backends()
    if len(backends) < 2:
        pytest.skip("Need at least 2 backends for cross-backend testing")

    input_arr = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)

    outputs = {}
    for backend in backends[:2]:
        os.environ['YAPOCIS_BACKEND'] = backend
        prog = core.load_program("cnn")
        outputs[backend] = prog.gelu(input_arr)

    backend1, backend2 = list(outputs.keys())
    np.testing.assert_allclose(
        outputs[backend1], outputs[backend2],
        rtol=1e-5, atol=1e-6,
        err_msg=f"Cross-backend mismatch: {backend1} vs {backend2}"
    )


# =================================================================
# Integration Tests
# =================================================================

@pytest.mark.parametrize("backend", get_test_backends())
def test_all_kernels_load(backend):
    """Test that all expected kernels load successfully."""
    os.environ['YAPOCIS_BACKEND'] = backend
    prog = core.load_program("cnn")

    expected_kernels = [
        # Convolution
        'conv2d_3x3', 'conv2d_3x3_optimized', 'conv2d_1x1', 'depthwise_conv2d_3x3',
        # Activation
        'relu_activation', 'gelu', 'softmax',
        # Pooling
        'max_pool_2x2', 'global_avg_pool', 'global_max_pool',
        # Normalization
        'batch_norm', 'layer_norm', 'layer_norm_1d',
        # Fully Connected
        'fully_connected', 'matrix_multiply_simd',
        # Regularization
        'dropout', 'spatial_dropout'
    ]

    missing = [k for k in expected_kernels if not hasattr(prog, k)]
    assert not missing, f"Missing kernels: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
