"""
Validation framework for using stored test results as regression tests.

This module provides utilities for:
1. Storing test results as validation references
2. Loading validation references for comparison
3. Comparing test results against stored validation data
4. Managing validation file lifecycle
"""

from pathlib import Path

import numpy as np

from .basic import _current_backend
from .internals import get_viewing_queue

# Default paths
QUEUE_DIR = Path(get_viewing_queue())
RAW_DIR = QUEUE_DIR / "raw"
VALIDATION_DIR = QUEUE_DIR / "validation"


def get_validation_key(
    test_name: str, backend: str | None = None, backend_specific: bool = False
) -> str:
    """
    Generate a standardized validation key for test results.

    Args:
        test_name: Name of the test (e.g., "graphics4 circles")
        backend: Backend name (only used if backend_specific=True)
        backend_specific: If True, creates backend-specific key

    Returns:
        Standardized key for validation lookup
    """
    if backend_specific and backend:
        return f"{test_name} [{backend}]"
    return test_name


def save_validation_reference(
    test_name: str,
    array: np.ndarray,
    backend: str | None = None,
    backend_specific: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Save an array as a validation reference.

    Args:
        test_name: Name of the test
        array: NumPy array to save as reference
        backend: Backend name (only used if backend_specific=True)
        backend_specific: If True, creates backend-specific reference
        overwrite: Whether to overwrite existing reference

    Returns:
        Path to saved validation file

    Raises:
        FileExistsError: If reference exists and overwrite=False
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    key = get_validation_key(test_name, backend, backend_specific)
    validation_file = VALIDATION_DIR / f"{key}.npz"

    if validation_file.exists() and not overwrite:
        raise FileExistsError(f"Validation reference already exists: {validation_file}")

    np.savez_compressed(validation_file, array)
    print(f"Saved validation reference: {validation_file}")
    return validation_file


def load_validation_reference(
    test_name: str, backend: str | None = None
) -> np.ndarray | None:
    """
    Load a validation reference array.

    Checks for agnostic reference first, then backend-specific reference.

    Args:
        test_name: Name of the test
        backend: Backend name for fallback to backend-specific reference

    Returns:
        Validation reference array, or None if not found
    """
    # Try agnostic reference first
    agnostic_key = get_validation_key(test_name, backend, backend_specific=False)
    agnostic_file = VALIDATION_DIR / f"{agnostic_key}.npz"

    if agnostic_file.exists():
        data = np.load(agnostic_file)
        return data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]

    # Fallback to backend-specific reference
    if backend:
        specific_key = get_validation_key(test_name, backend, backend_specific=True)
        specific_file = VALIDATION_DIR / f"{specific_key}.npz"

        if specific_file.exists():
            data = np.load(specific_file)
            return data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]

    return None


def compare_with_validation(
    test_name: str,
    result: np.ndarray,
    backend: str | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[bool, dict]:
    """
    Compare test result with stored validation reference.

    Args:
        test_name: Name of the test
        result: Test result array
        backend: Backend name for fallback to backend-specific reference
        atol: Absolute tolerance for comparison (default: 1e-5 for ~5 significant digits)
        rtol: Relative tolerance for comparison (default: 1e-5 for ~5 significant digits)

    Returns:
        tuple of (matches, info_dict) where info_dict contains comparison details
    """
    reference = load_validation_reference(test_name, backend)

    info = {
        "test_name": test_name,
        "backend": backend or _current_backend(),
        "reference_exists": reference is not None,
        "result_shape": result.shape,
        "result_dtype": result.dtype,
        "result_sum": float(np.sum(result)),
    }

    if reference is None:
        info["status"] = "no_reference"
        info["matches"] = False
        return False, info

    info["reference_shape"] = reference.shape
    info["reference_dtype"] = reference.dtype
    info["reference_sum"] = float(np.sum(reference))

    # Shape and dtype checks
    if result.shape != reference.shape:
        info["status"] = "shape_mismatch"
        info["matches"] = False
        return False, info

    if result.dtype != reference.dtype:
        info["status"] = "dtype_mismatch"
        info["matches"] = False
        return False, info

    # Numerical comparison
    matches = np.allclose(result, reference, atol=atol, rtol=rtol)
    info["matches"] = matches
    info["max_abs_diff"] = float(np.max(np.abs(result - reference)))
    info["status"] = "matches" if matches else "numerical_mismatch"

    return matches, info


def promote_to_validation(
    test_name: str,
    backend: str | None = None,
    source_dir: Path | None = None,
    source_file: Path | None = None,
    backend_specific: bool = False,
) -> bool:
    """
    Promote a test result from the raw queue to validation reference.

    Args:
        test_name: Name of the test
        backend: Backend name (needed to find source file in raw)
        source_dir: Source directory, defaults to RAW_DIR
        source_file: Optional direct path to source .npz file (bypasses search)
        backend_specific: If True, creates backend-specific validation reference

    Returns:
        True if promotion successful, False otherwise
    """
    if backend is None:
        backend = _current_backend()

    # If source_file is provided directly, use it
    if source_file is not None:
        if not source_file.exists():
            print(f"Source file not found: {source_file}")
            return False
    else:
        # Fall back to searching for the file
        if source_dir is None:
            source_dir = RAW_DIR

        # Look for source file - search for files matching test_name with any qualifiers
        # This handles files like "test_name [100] [backend].npz" or "test [ref] name [backend].npz"
        # We iterate through all .npz files and match by removing all bracket qualifiers
        import re

        source_file = None
        all_files = list(source_dir.glob("*.npz"))
        for f in all_files:
            fname = f.stem  # filename without extension

            # Extract backend from filename (last bracketed term)
            backend_match = re.search(r"\[(\w+)\]$", fname)
            if not backend_match or backend_match.group(1) != backend:
                continue  # Wrong backend or no backend marker

            # Remove all bracket qualifiers to get the base test name
            fname_base = re.sub(r"\s*\[\w+\]", "", fname).strip()

            # Compare with test_name
            if fname_base == test_name:
                source_file = f
                break

        if source_file is None:
            print(
                f"Source file not found for test_name='{test_name}' backend='{backend}'"
            )
            print(f"  Searched in: {source_dir}")
            return False

    # Load and re-save to validation directory
    data = np.load(source_file)
    array = data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]

    try:
        save_validation_reference(
            test_name, array, backend, backend_specific, overwrite=True
        )
        return True
    except Exception as e:
        print(f"Failed to promote {test_name}: {e}")
        return False


def list_validation_references(backend: str | None = None) -> list[str]:
    """
    list available validation references.

    Args:
        backend: Filter by backend, or None for all

    Returns:
        list of test names with validation references
    """
    if not VALIDATION_DIR.exists():
        return []

    references = []
    pattern = f"*[{backend}].npz" if backend else "*.npz"

    for file in VALIDATION_DIR.glob(pattern):
        # Extract test name from filename
        name = file.stem  # Remove .npz extension
        if backend:
            # Remove " [backend]" suffix
            name = name.replace(f" [{backend}]", "")
        references.append(name)

    return sorted(references)


def cleanup_validation_references(
    backend: str | None = None, confirm: bool = True
) -> int:
    """
    Clean up validation references.

    Args:
        backend: Backend to clean up, or None for all
        confirm: Whether to confirm before deletion

    Returns:
        Number of files deleted
    """
    if not VALIDATION_DIR.exists():
        return 0

    pattern = f"*[{backend}].npz" if backend else "*.npz"
    files = list(VALIDATION_DIR.glob(pattern))

    if not files:
        return 0

    if confirm:
        print(f"Found {len(files)} validation files to delete:")
        for file in files[:5]:  # Show first 5
            print(f"  {file.name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")

        response = input("Delete these files? (y/N): ")
        if response.lower() != "y":
            return 0

    deleted = 0
    for file in files:
        try:
            file.unlink()
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

    return deleted


class ValidationContext:
    """Context manager for validation testing."""

    def __init__(
        self,
        test_name: str,
        backend: str | None = None,
        auto_promote: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.test_name = test_name
        self.backend = backend or _current_backend()
        self.auto_promote = auto_promote
        self.atol = atol
        self.rtol = rtol
        self.validation_result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def validate(self, result: np.ndarray) -> bool:
        """
        Validate result against reference.

        Args:
            result: Test result to validate

        Returns:
            True if validation passes, False otherwise
        """
        matches, info = compare_with_validation(
            self.test_name, result, self.backend, self.atol, self.rtol
        )

        self.validation_result = info

        if not matches and info["status"] == "no_reference" and self.auto_promote:
            print(
                f"No validation reference found for {self.test_name}, auto-promoting..."
            )
            save_validation_reference(
                self.test_name, result, self.backend, overwrite=True
            )
            return True

        return matches

    def assert_valid(self, result: np.ndarray) -> None:
        """Assert that result matches validation reference."""
        if not self.validate(result):
            info = self.validation_result
            raise AssertionError(
                f"Validation failed for {self.test_name}: {info['status']}"
            )
