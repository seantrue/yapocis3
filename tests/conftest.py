import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add yapocis to path if needed
if str(os.path.dirname(os.path.dirname(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yapocis.utils.validation import compare_with_validation


def pytest_ignore_collect(collection_path: Path, config):
    """Conditionally skip collecting CUDA PyCUDA test if pycuda is missing.

    The file cukernels/test_convolve.py imports pycuda.autoinit unconditionally.
    When PyCUDA is not installed, let pytest skip collecting just that file.
    """
    if collection_path.as_posix().endswith("cukernels/test_convolve.py"):
        try:
            importlib.import_module("pycuda.autoinit")
            return False
        except Exception:
            return True
    return False


def pytest_addoption(parser):
    """Add command line options for validation testing."""
    parser.addoption(
        "--validate",
        action="store_true",
        default=False,
        help="Enable validation against stored references",
    )
    parser.addoption(
        "--update-validation",
        action="store_true",
        default=False,
        help="Update validation references with current test results",
    )
    parser.addoption(
        "--validation-tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for validation comparisons (default: 1e-6)",
    )


@pytest.fixture
def validation_enabled(request):
    """Check if validation is enabled for this test run."""
    return request.config.getoption("--validate")


@pytest.fixture
def update_validation(request):
    """Check if validation references should be updated."""
    return request.config.getoption("--update-validation")


@pytest.fixture
def validation_tolerance(request):
    """Get validation tolerance setting."""
    return request.config.getoption("--validation-tolerance")


@pytest.fixture
def validate_result(
    request, validation_enabled, update_validation, validation_tolerance
):
    """
    Fixture that returns a function to validate test results.

    Usage in tests:
        def test_something(validate_result):
            result = compute_something()
            validate_result("test_something", result)
    """

    def _validate(test_name: str, result: np.ndarray, backend: str | None = None):
        if not validation_enabled and not update_validation:
            # Validation not requested, skip silently
            return True

        if update_validation:
            # Update mode: save current result as new reference
            from yapocis.utils.validation import save_validation_reference

            try:
                save_validation_reference(test_name, result, backend, overwrite=True)
                print(f"Updated validation reference for {test_name}")
                return True
            except Exception as e:
                pytest.fail(
                    f"Failed to update validation reference for {test_name}: {e}"
                )
        else:
            # Validation mode: compare against existing reference
            matches, info = compare_with_validation(
                test_name, result, backend, atol=validation_tolerance
            )

            if not matches:
                if info["status"] == "no_reference":
                    pytest.skip(f"No validation reference found for {test_name}")
                else:
                    pytest.fail(f"Validation failed for {test_name}: {info['status']}")

            return True

    return _validate


class ValidationHelper:
    """Helper class for validation operations in tests."""

    def __init__(self, enabled: bool, update_mode: bool, tolerance: float):
        self.enabled = enabled
        self.update_mode = update_mode
        self.tolerance = tolerance

    def check(
        self, test_name: str, result: np.ndarray, backend: str | None = None
    ) -> bool:
        """Check result against validation reference."""
        if not self.enabled and not self.update_mode:
            return True

        if self.update_mode:
            from yapocis.utils.validation import save_validation_reference

            save_validation_reference(test_name, result, backend, overwrite=True)
            return True
        else:
            matches, info = compare_with_validation(
                test_name, result, backend, atol=self.tolerance
            )
            return matches

    def assert_valid(
        self, test_name: str, result: np.ndarray, backend: str | None = None
    ):
        """Assert that result matches validation reference."""
        if not self.check(test_name, result, backend):
            pytest.fail(f"Validation failed for {test_name}")


@pytest.fixture
def validation_helper(validation_enabled, update_validation, validation_tolerance):
    """Validation helper instance."""
    return ValidationHelper(validation_enabled, update_validation, validation_tolerance)


# Mark for skipping tests that require backends but dependencies are not installed
def pytest_runtest_setup(item):
    """Skip backend tests when dependencies are not available."""
    # Check for CUDA marker
    cuda_marker = item.get_closest_marker("cuda")
    if cuda_marker is not None:
        try:
            import pycuda  # noqa: F401
        except ImportError:
            pytest.skip("pycuda not installed - skipping CUDA tests")

    # Check for Metal marker
    metal_marker = item.get_closest_marker("metal")
    if metal_marker is not None:
        try:
            import pymetallic  # noqa: F401
        except ImportError:
            pytest.skip("pymetallic not installed - skipping Metal tests")

    # Check for OCL marker
    ocl_marker = item.get_closest_marker("ocl")
    if ocl_marker is not None:
        try:
            import pyopencl  # noqa: F401
        except ImportError:
            pytest.skip("pyopencl not installed - skipping OpenCL tests")

    # Check for Numba marker
    numba_marker = item.get_closest_marker("numba")
    if numba_marker is not None:
        try:
            import numba  # noqa: F401
        except ImportError:
            pytest.skip("numba not installed - skipping Numba tests")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "smoke: marks quick validation tests that cover all library modules"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests that support validation checking"
    )
    config.addinivalue_line(
        "markers", "track_memory: marks tests that should track memory usage"
    )
    config.addinivalue_line(
        "markers",
        "performance: marks performance tests that should only run when explicitly requested",
    )


def get_backend_device_combinations():
    """
    Generate all (backend, device) combinations for testing.

    This is used for parametrizing tests that need to run across all
    backend/device combinations. Respects YAPOCIS_TEST_BACKENDS for filtering.

    Returns:
        list: List of (backend, device_name) tuples

    Example:
        @pytest.mark.parametrize("backend,device", get_backend_device_combinations())
        def test_something(backend, device):
            ...
    """
    try:
        from yapocis.core import get_test_backends, get_test_devices
    except ImportError:
        # Fallback if imports fail
        return [("ocl", "default")]

    combinations = []
    for backend in get_test_backends():
        try:
            devices = get_test_devices(backend)
            for device_name in devices.keys():
                combinations.append((backend, device_name))
        except Exception:
            # If we can't get devices for a backend, skip it
            continue

    return combinations


@pytest.fixture(autouse=True)
def clean_yapocis_device():
    """Clean up YAPOCIS_DEVICE environment variable before each test."""
    # Store original value
    original_device = os.environ.get("YAPOCIS_DEVICE")

    # Clear YAPOCIS_DEVICE to avoid pollution from previous tests
    if "YAPOCIS_DEVICE" in os.environ:
        del os.environ["YAPOCIS_DEVICE"]

    yield

    # Restore original value after test
    if original_device is not None:
        os.environ["YAPOCIS_DEVICE"] = original_device
    elif "YAPOCIS_DEVICE" in os.environ:
        del os.environ["YAPOCIS_DEVICE"]


# Performance tracking fixtures
@pytest.fixture
def perf_tracker(request):
    """Automatic performance tracking for test functions."""
    from . import PerformanceTracker

    test_name = request.node.name
    backend = os.environ.get("YAPOCIS_BACKEND", "unknown")

    # Extract memory tracking preference from markers
    track_memory = False
    if hasattr(request.node, "get_closest_marker"):
        memory_marker = request.node.get_closest_marker("track_memory")
        if memory_marker:
            track_memory = True

    tracker = PerformanceTracker(f"{test_name}_{backend}", track_memory=track_memory)

    def save_results():
        """Save performance results after test completion."""
        try:
            filepath = tracker.save_results()
            print(f"\n📊 Performance data saved to: {filepath}")
        except Exception as e:
            print(f"\n⚠️  Failed to save performance data: {e}")

    request.addfinalizer(save_results)
    return tracker


@pytest.fixture
def backend_comparison(request):
    """Run the same test across multiple backends for comparison."""
    import time

    from . import PerformanceTracker

    available_backends = []

    try:
        from yapocis.core import get_test_backends

        available_backends = get_test_backends()
    except ImportError:
        available_backends = ["ocl"]  # fallback

    original_backend = os.environ.get("YAPOCIS_BACKEND")
    test_name = request.node.name
    tracker = PerformanceTracker(f"{test_name}_comparison", track_memory=True)

    def run_on_backends(test_func, *args, **kwargs):
        """Run test function on all available backends."""
        results = {}

        for backend in available_backends:
            os.environ["YAPOCIS_BACKEND"] = backend
            try:
                print(f"\n🔄 Running on {backend}...")
                start_time = time.time()
                result = test_func(*args, **kwargs)
                elapsed = time.time() - start_time

                tracker.record(f"{backend}_execution", elapsed)
                results[backend] = result

            except Exception as e:
                print(f"❌ Failed on {backend}: {e}")
                results[backend] = None

        # Restore original backend
        if original_backend:
            os.environ["YAPOCIS_BACKEND"] = original_backend

        return results

    def cleanup():
        """Print comparison results and save data."""
        if len(tracker.results) > 0:
            print("\n" + tracker.compare_backends())
            tracker.save_results()

    request.addfinalizer(cleanup)

    # Return both the runner function and the tracker
    return run_on_backends, tracker
