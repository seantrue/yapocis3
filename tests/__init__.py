#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import inspect
import os
import random
import string
import time
from random import shuffle

import numpy as np
from numpy.testing import assert_allclose

random.seed(42)

import pytest

from yapocis.images import find_image
from yapocis.utils.basic import imread, show_array
from yapocis.core import get_backend, get_manager, get_backend_device

ASIZE = 980


def image_path_test(image_name="test.jpg"):
    return find_image(image_name)


def image_test():
    test_image = image_path_test()
    img = imread(test_image)
    assert img.shape[0] != img.shape[1], "Test image must not be square"
    return img


def text_test_lines(text_name="test.txt"):
    dirname = os.path.dirname(__file__)
    lines = [
        line.strip()
        for line in open(os.path.join(dirname, "test.txt")).read().splitlines()
    ]
    return lines


def text_test():
    dirname = os.path.dirname(__file__)
    s = open(os.path.join(dirname, "test.txt")).read()
    uninteresting = set(string.printable) - set(
        string.ascii_lowercase + string.ascii_uppercase
    )
    translator = s.maketrans({c: " " for c in uninteresting})
    s = s.translate(translator)
    words = sorted(set(word for word in s.split() if len(word) > 2))
    return words


def interesting_words(n=10):
    words = text_test()
    shuffle(words)
    return words[:n]


def double_shuffle(lst, n):
    while len(lst) < n:
        lst += lst
    lst = lst[:n]
    shuffle(lst)
    return lst


@pytest.fixture(scope="session", autouse=True)
def show_snapshots(request):
    from yapocis.ocl.device_manager import display_snapshots

    request.addfinalizer(display_snapshots)


def strip_test_prefix(s):
    if s.startswith("test_"):
        s = s[5:]
    return s


def name_of_test(depth=1, full_path=False):
    """
    Extract test name from call stack.

    Args:
        depth: How many frames up the stack to look
        full_path: If True, return full module.class.function path and file path
                   If False (default), return (function_name, module_name) for backward compatibility

    Returns:
        If full_path=False: (function_name, module_name)  # backward compatible
        If full_path=True: (full_test_path, file_path)   # e.g., ("test_graphics4.TestClass.test_method", "/path/to/test_graphics4.py")
    """
    current_frame = inspect.currentframe()
    call_frames = inspect.getouterframes(current_frame, depth + 1)
    fi = call_frames[depth]

    caller_name = fi.function
    file_name = fi.filename
    file_path = os.path.abspath(file_name)
    module_name = os.path.splitext(os.path.basename(os.path.split(file_name)[-1]))[0]

    if not full_path:
        # Backward compatible behavior
        return strip_test_prefix(caller_name), strip_test_prefix(module_name)

    # Build full path: module.class.function
    parts = [strip_test_prefix(module_name)]

    # Check if this is a method call by looking for 'self' or 'cls' in frame locals
    frame_locals = fi.frame.f_locals
    if "self" in frame_locals:
        # Instance method - get class name
        obj = frame_locals["self"]
        class_name = obj.__class__.__name__
        parts.append(class_name)
    elif "cls" in frame_locals:
        # Class method - get class name
        cls = frame_locals["cls"]
        class_name = cls.__name__
        parts.append(class_name)

    # Add function name
    parts.append(strip_test_prefix(caller_name))

    full_test_path = ".".join(parts)
    return full_test_path, file_path


def show(*args, **kwargs):
    tname, fname = name_of_test(2)
    array = args[-1]
    other_args = args[:-1]
    show_array(array, f"{fname}: {tname}", *other_args, **kwargs)


def snapshot(snapper):
    tname, fname = name_of_test(2)
    snapper.snapshot(name=f"{fname}/{tname}")


def snapshow(
    snapper: "BoundKernel", a: np.ndarray, *args, depth=2, checkzero=True, **kwargs
):
    tname, fname = name_of_test(depth)
    snapper.snapshot(name=f"{fname}/{tname}")
    show_array(a, f"{fname} {tname}", *args, checkzero=checkzero, **kwargs)


class _CheckpointContext:
    """Context manager for timing a checkpoint block."""

    def __init__(self, stopwatcher, name):
        self.stopwatcher = stopwatcher
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Start timing this checkpoint block."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record the checkpoint with the time elapsed in this block."""
        if self.start_time is not None:
            now = time.time()
            elapsed = now - self.start_time
            self.stopwatcher.checkpoints[self.name] = elapsed
            self.stopwatcher.last_checkpoint = now
        return False


YAPOCIS_PERFORMANCE = "YAPOCIS_PERFORMANCE" in os.environ


class StopWatcher:
    def __init__(self, *args, log=None, save_perf=False, kernel=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.log = log
        self.save_perf = save_perf or YAPOCIS_PERFORMANCE
        self.checkpoints = {}
        self.last_checkpoint = None
        self.kernel = kernel
        self._test_name = None
        self._test_file = None
        self._checkpoint_name = None

    def __enter__(self):
        self.t = time.time()
        self.last_checkpoint = self.t

        # Capture test information for performance tracking
        if self.save_perf:
            try:
                self._test_name, self._test_file = name_of_test(depth=2, full_path=True)
            except Exception:
                # If we can't get test info, disable performance saving
                self.save_perf = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, optionally log the summary, and save performance data."""
        # Log summary if requested
        if self.log is not None and callable(self.log):
            self.log(self.summary())

        # Save performance data if requested
        if self.save_perf and self._test_name and self._test_file:
            try:
                from yapocis.utils.performance import save_performance_data

                backend = get_backend()
                mgr = get_manager(backend)
                device_index, device_name = get_backend_device()
                driver_version = None

                # Save the data
                save_performance_data(
                    test_name=self._test_name,
                    file_path=self._test_file,
                    backend=backend,
                    timing_data=self.checkpoints.copy(),
                    test_start_time=self.t,
                    device_name=device_name,
                    driver_version=driver_version,
                )
            except Exception as e:
                # Don't fail the test if performance tracking fails
                import sys

                print(f"Warning: Failed to save performance data: {e}", file=sys.stderr)

        return False

    def __call__(
        self,
        array: np.ndarray,
        *args,
        checkzero=False,
        depth=0,
        kernel=None,
    ):
        t = time.time() - self.t
        kernel = kernel if kernel else self.kernel
        snapshow(
            kernel,
            array,
            *self.args,
            *args,
            depth=3 + depth,
            t=f"{t:.03f}",
            checkzero=checkzero,
        )

    show = __call__

    def showcheck(
        self, array: np.ndarray, *args, checkzero=False, depth=0, kernel=None
    ):
        kernel = kernel if kernel else self.kernel
        self.show(
            array,
            self._checkpoint_name,
            *args,
            checkzero=checkzero,
            depth=depth + 1,
            kernel=kernel,
        )

    def checkpoint(self, name: str):
        """
        Add a timing checkpoint. Can be used in two ways:

        1. As a regular method call (records time since last checkpoint):
           sw.checkpoint("operation")

        2. As a context manager (records time within the block):
           with sw.checkpoint("operation"):
               # code to time

        Returns a context manager that can optionally be used with 'with'.
        """
        # For backward compatibility: record checkpoint immediately when called as regular method
        now = time.time()
        if self.last_checkpoint is None:
            self.last_checkpoint = self.t
        elapsed = now - self.last_checkpoint
        self.checkpoints[name] = elapsed
        self.last_checkpoint = now
        self._checkpoint_name = name

        # Return context manager for optional 'with' usage
        # If used with 'with', it will overwrite the checkpoint with block timing
        return _CheckpointContext(self, name)

    @property
    def stats(self) -> dict:
        """Return timing statistics."""
        if not self.checkpoints:
            total_time = time.time() - self.t if hasattr(self, "t") else 0
            return {"total": total_time, "checkpoints": {}}

        return {
            "total": sum(self.checkpoints.values()),
            "checkpoints": self.checkpoints.copy(),
            "slowest": (
                max(self.checkpoints.items(), key=lambda x: x[1])
                if self.checkpoints
                else None
            ),
            "fastest": (
                min(self.checkpoints.items(), key=lambda x: x[1])
                if self.checkpoints
                else None
            ),
        }

    def summary(self) -> str:
        """Human readable timing summary."""
        if not self.checkpoints:
            total_time = time.time() - self.t if hasattr(self, "t") else 0
            return f"Total time: {total_time:.3f}s (no checkpoints)"

        lines = [f"Timing summary (total: {self.stats['total']:.3f}s):"]
        for name, elapsed in self.checkpoints.items():
            percentage = (
                (elapsed / self.stats["total"] * 100) if self.stats["total"] > 0 else 0
            )
            lines.append(f"  {name}: {elapsed:.3f}s ({percentage:.1f}%)")

        slowest_name, slowest_time = self.stats["slowest"]
        lines.append(f"Slowest operation: {slowest_name} ({slowest_time:.3f}s)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Debug representation showing current timing state."""
        status = "active" if hasattr(self, "t") else "inactive"
        checkpoint_count = len(self.checkpoints)

        if hasattr(self, "t") and self.checkpoints:
            elapsed = time.time() - self.t
            return f"<StopWatcher {status}: {checkpoint_count} checkpoints, {elapsed:.3f}s total>"
        elif hasattr(self, "t"):
            elapsed = time.time() - self.t
            return f"<StopWatcher {status}: {elapsed:.3f}s elapsed>"
        else:
            return f"<StopWatcher {status}>"


class PerformanceTracker:
    """Advanced performance tracking with backend comparison and regression detection."""

    def __init__(self, test_name: str, track_memory: bool = False):
        self.test_name = test_name
        self.track_memory = track_memory
        self.results = {}  # backend -> {operation -> [timing_data]}
        self.current_backend = self._get_backend()
        self.environment = self._capture_environment()

        if track_memory:
            try:
                import psutil

                self.process = psutil.Process()
                self.memory_enabled = True
            except ImportError:
                self.memory_enabled = False
                print("Warning: psutil not available, memory tracking disabled")
        else:
            self.memory_enabled = False

    def _get_backend(self) -> str:
        """Get the current backend from environment."""
        import os

        return os.environ.get("YAPOCIS_BACKEND", "unknown")

    def _capture_environment(self) -> dict:
        """Capture relevant environment variables for performance analysis."""
        import os

        env_vars = {}

        # Capture BENCHMARKING flag (disables image saving)
        benchmarking = os.environ.get("BENCHMARKING")
        if benchmarking is not None:
            env_vars["BENCHMARKING"] = benchmarking

        # Capture all YAPOCIS_* variables
        for key, value in os.environ.items():
            if key.startswith("YAPOCIS_"):
                env_vars[key] = value

        return env_vars

    def record(self, operation: str, elapsed: float, memory_mb: float = None):
        """Record timing (and optionally memory) data for backend comparison."""
        backend = self.current_backend

        if backend not in self.results:
            self.results[backend] = {}
        if operation not in self.results[backend]:
            self.results[backend][operation] = []

        entry = {"time": elapsed}
        if memory_mb is not None:
            entry["memory_mb"] = memory_mb

        self.results[backend][operation].append(entry)

    def time_operation(self, operation_name: str, func, *args, **kwargs):
        """Time a function call and record the results."""
        memory_before = None
        if self.memory_enabled:
            memory_before = self.process.memory_info().rss / 1024 / 1024

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        memory_after = None
        if self.memory_enabled and memory_before:
            memory_after = self.process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before
            self.record(operation_name, elapsed, memory_delta)
        else:
            self.record(operation_name, elapsed)

        return result

    def get_stats(self, operation: str = None) -> dict:
        """Get statistics for a specific operation or all operations."""
        import numpy as np

        stats = {}
        backends = self.results.keys() if operation is None else [self.current_backend]

        for backend in backends:
            if backend not in self.results:
                continue

            backend_stats = {}
            operations = (
                self.results[backend].keys() if operation is None else [operation]
            )

            for op in operations:
                if op not in self.results[backend]:
                    continue

                times = [entry["time"] for entry in self.results[backend][op]]
                memories = [
                    entry.get("memory_mb", 0) for entry in self.results[backend][op]
                ]

                op_stats = {
                    "count": len(times),
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "median": np.median(times),
                }

                if self.memory_enabled and any(memories):
                    op_stats["memory"] = {
                        "mean_mb": np.mean(memories),
                        "max_mb": np.max(memories),
                    }

                backend_stats[op] = op_stats

            stats[backend] = backend_stats

        return stats

    def compare_backends(self) -> str:
        """Generate backend comparison report."""
        stats = self.get_stats()

        if len(stats) < 2:
            return "Need at least 2 backends for comparison"

        lines = [f"Backend Performance Comparison - {self.test_name}"]
        lines.append("=" * 60)

        # Get all operations across all backends
        all_operations = set()
        for backend_stats in stats.values():
            all_operations.update(backend_stats.keys())

        for operation in sorted(all_operations):
            lines.append(f"\nOperation: {operation}")
            lines.append("-" * 30)

            # Find fastest backend for this operation
            backend_times = {}
            for backend, backend_stats in stats.items():
                if operation in backend_stats:
                    backend_times[backend] = backend_stats[operation]["mean"]

            if not backend_times:
                continue

            fastest_backend = min(backend_times.keys(), key=lambda b: backend_times[b])
            fastest_time = backend_times[fastest_backend]

            for backend in sorted(backend_times.keys()):
                time_val = backend_times[backend]
                speedup = fastest_time / time_val if time_val > 0 else float("inf")

                status = (
                    "🥇" if backend == fastest_backend else f"{speedup:.2f}x slower"
                )
                lines.append(f"  {backend:12}: {time_val*1000:8.2f}ms  {status}")

                # Add memory info if available
                if (
                    backend in stats
                    and operation in stats[backend]
                    and "memory" in stats[backend][operation]
                ):
                    mem_info = stats[backend][operation]["memory"]
                    lines.append(
                        f"               Memory: {mem_info['mean_mb']:6.1f}MB avg"
                    )

        return "\n".join(lines)

    def save_results(self, filename: str = None):
        """Save performance results to JSON file for historical tracking."""
        import json
        import os
        from datetime import datetime

        if filename is None:
            filename = f"performance_{self.test_name}_{self.current_backend}.json"

        # Create performance directory if it doesn't exist
        perf_dir = os.path.join(os.path.dirname(__file__), "..", "performance")
        os.makedirs(perf_dir, exist_ok=True)
        filepath = os.path.join(perf_dir, filename)

        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Add current results with timestamp
        timestamp = datetime.now().isoformat()
        if "runs" not in existing_data:
            existing_data["runs"] = []

        existing_data["runs"].append(
            {
                "timestamp": timestamp,
                "test_name": self.test_name,
                "results": self.results,
                "environment": self.environment,
            }
        )

        # Keep only last 100 runs to prevent files from growing too large
        existing_data["runs"] = existing_data["runs"][-100:]

        with open(filepath, "w") as f:
            json.dump(existing_data, f, indent=2)

        return filepath

    def detect_regressions(
        self, baseline_file: str = None, threshold: float = 1.2
    ) -> dict:
        """Detect performance regressions by comparing with historical data.

        Args:
            baseline_file: Path to baseline performance file (optional)
            threshold: Regression threshold (1.2 = 20% slower is flagged)

        Returns:
            Dictionary with regression analysis results
        """
        import json
        import os
        from statistics import median

        if baseline_file is None:
            # Use default performance file
            perf_dir = os.path.join(os.path.dirname(__file__), "..", "performance")
            baseline_file = os.path.join(
                perf_dir, f"performance_{self.test_name}_{self.current_backend}.json"
            )

        if not os.path.exists(baseline_file):
            return {
                "status": "no_baseline",
                "message": f"No baseline file found: {baseline_file}",
            }

        try:
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return {"status": "error", "message": f"Failed to load baseline: {e}"}

        # Get baseline statistics (median of last 10 runs for stability)
        baseline_stats = {}
        for run in baseline_data.get("runs", [])[-10:]:
            for backend, backend_data in run.get("results", {}).items():
                if backend not in baseline_stats:
                    baseline_stats[backend] = {}

                for operation, measurements in backend_data.items():
                    if operation not in baseline_stats[backend]:
                        baseline_stats[backend][operation] = []

                    # Extract times from measurements
                    times = [entry["time"] for entry in measurements]
                    if times:
                        baseline_stats[backend][operation].extend(times)

        # Calculate baseline medians
        baseline_medians = {}
        for backend, operations in baseline_stats.items():
            baseline_medians[backend] = {}
            for operation, times in operations.items():
                if times:
                    baseline_medians[backend][operation] = median(times)

        # Compare current results against baseline
        regressions = {}
        current_stats = self.get_stats()

        for backend, operations in current_stats.items():
            if backend not in baseline_medians:
                continue

            backend_regressions = {}

            for operation, stats in operations.items():
                if operation not in baseline_medians[backend]:
                    continue

                baseline_time = baseline_medians[backend][operation]
                current_time = stats["median"]

                if baseline_time > 0:
                    slowdown = current_time / baseline_time

                    if slowdown >= threshold:
                        backend_regressions[operation] = {
                            "baseline_ms": baseline_time * 1000,
                            "current_ms": current_time * 1000,
                            "slowdown": slowdown,
                            "regression_pct": (slowdown - 1) * 100,
                            "threshold": threshold,
                        }

            if backend_regressions:
                regressions[backend] = backend_regressions

        return {
            "status": "complete",
            "regressions": regressions,
            "threshold": threshold,
            "baseline_runs": len(baseline_data.get("runs", [])),
            "has_regressions": len(regressions) > 0,
        }

    def regression_report(self, threshold: float = 1.2) -> str:
        """Generate a human-readable regression analysis report."""
        analysis = self.detect_regressions(threshold=threshold)

        if analysis["status"] != "complete":
            return f"⚠️  Regression analysis failed: {analysis['message']}"

        lines = [f"Performance Regression Analysis - {self.test_name}"]
        lines.append("=" * 60)
        lines.append(f"Threshold: {threshold:.1f}x slower = regression")
        lines.append(f"Baseline: {analysis['baseline_runs']} historical runs")

        if not analysis["has_regressions"]:
            lines.append("\n✅ No performance regressions detected!")
            return "\n".join(lines)

        lines.append(
            f"\n🚨 {len(analysis['regressions'])} backend(s) with regressions:"
        )

        for backend, regressions in analysis["regressions"].items():
            lines.append(f"\n{backend} backend:")
            lines.append("-" * 30)

            for operation, regression in regressions.items():
                lines.append(f"  {operation}:")
                lines.append(f"    Baseline: {regression['baseline_ms']:8.2f}ms")
                lines.append(f"    Current:  {regression['current_ms']:8.2f}ms")
                lines.append(
                    f"    Slowdown: {regression['slowdown']:8.2f}x ({regression['regression_pct']:+5.1f}%)"
                )

        return "\n".join(lines)


# Pytest integration fixtures
@pytest.fixture
def perf_tracker(request):
    """Automatic performance tracking for test functions."""
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
    available_backends = []

    from yapocis.core import get_backends

    try:
        available_backends = get_backends()
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


def assert_closeenough(a, b):
    assert a.shape == b.shape
    assert_allclose(a, b, atol=0.0001)
