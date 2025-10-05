"""
Example test showing validation framework usage.

This demonstrates how to modify existing tests to use the validation framework.
"""

import pytest
import numpy as np
from yapocis.lib.graphics4 import Graphics4
from yapocis.utils.deterministic import generate_test_field
from yapocis.core.manager import get_backends


@pytest.mark.validation
@pytest.mark.parametrize("backend", get_backends())
def test_graphics4_circles_validation(backend, validate_result, monkeypatch):
    """Example test with validation support."""
    # Set backend
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create test canvas
    canvas = np.zeros((100, 100, 4), dtype=np.float32)
    g4 = Graphics4()

    # Draw some circles
    circles = [(25, 25, 10), (75, 75, 15), (50, 25, 8)]
    colors = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    )

    g4.circles(canvas, circles, colors=colors)
    result = g4.read(canvas)

    # Validate result against stored reference
    validate_result("graphics4_circles_small", result, backend)

    # Test assertions can still be used alongside validation
    assert np.sum(result) > 0, "Canvas should not be empty"


@pytest.mark.validation
def test_manual_validation_context():
    """Example showing manual validation context usage."""
    from yapocis.utils.validation import ValidationContext, VALIDATION_DIR
    from pathlib import Path

    # Clean up any existing reference to ensure test is deterministic
    ref_file = VALIDATION_DIR / "manual_validation_test.npz"
    if ref_file.exists():
        ref_file.unlink()

    # Generate test data (deterministic based on test_name)
    test_data = generate_test_field(
        (50, 50), test_name="manual_validation", distribution="random"
    )

    # Use validation context with auto_promote
    with ValidationContext("manual_validation_test", auto_promote=True) as validator:
        # Perform validation - should auto-promote since no reference exists
        is_valid = validator.validate(test_data)
        assert is_valid, "Validation should pass (auto-promote if no reference)"

    # Second validation should also pass (now comparing against promoted reference)
    with ValidationContext("manual_validation_test", auto_promote=False) as validator:
        # Generate same deterministic data
        test_data2 = generate_test_field(
            (50, 50), test_name="manual_validation", distribution="random"
        )
        is_valid2 = validator.validate(test_data2)
        assert is_valid2, "Validation should pass against promoted reference"


def test_no_validation():
    """Test without validation marker runs normally."""
    # This test will not participate in validation even if --validate is passed
    result = np.ones((10, 10), dtype=np.float32)
    assert np.sum(result) == 100
