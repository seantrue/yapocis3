#!/usr/bin/env python
"""
Validation management script for yapocis test results.

This script helps manage validation references by:
- Promoting test results from queue/raw to validation references
- Listing available validation references
- Cleaning up old validation data
- Bulk operations across backends
"""

import argparse
import sys
from pathlib import Path

# Add yapocis to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re

from yapocis.core import get_backends
from yapocis.utils.validation import (
    RAW_DIR,
    VALIDATION_DIR,
    cleanup_validation_references,
    list_validation_references,
    load_validation_reference,
    promote_to_validation,
)


def parse_test_name(filename):
    """
    Parse a filename to extract test name and backend, handling [ref]/[test] qualifiers.

    Args:
        filename: Filename (without path)

    Returns:
        tuple: (test_name, backend) or (None, None) if parsing fails
    """
    # Remove file extension
    name = Path(filename).stem

    # Extract backend from the last [backend] pattern (actual backend like ocl, metal, numba)
    backend_match = re.search(r"\[(\w+)\]$", name)
    backend = None
    if backend_match:
        potential_backend = backend_match.group(1)
        # Only treat as backend if it's a known backend (not ref/test)
        if potential_backend.lower() in ["ocl", "metal", "numba", "cuda", "ref"]:
            backend = potential_backend

    # Remove all qualifiers: [ref], [test], and backend qualifiers like [ocl]
    test_name = re.sub(r"\s*\[(?:ref|test|\w+)\]", "", name)
    test_name = test_name.strip()

    return test_name, backend


def promote_command(args):
    """Handle promote subcommand."""
    if args.test_name == "all":
        # Promote all available test results
        promoted = 0
        failed = 0

        for backend in args.backends or get_backends():
            print(f"\nPromoting results for backend: {backend}")

            # Find all .npz files for this backend
            for npz_file in RAW_DIR.glob("*.npz"):
                # Parse the filename to extract test name and backend
                parsed_test_name, parsed_backend = parse_test_name(npz_file.name)

                # Skip if this file doesn't match the target backend
                if parsed_backend != backend:
                    continue

                test_name = parsed_test_name

                if promote_to_validation(
                    test_name, backend, backend_specific=args.backend_specific
                ):
                    ref_type = f"[{backend}]" if args.backend_specific else "(agnostic)"
                    print(f"  ✓ {test_name} {ref_type}")
                    promoted += 1
                else:
                    print(f"  ✗ {test_name}")
                    failed += 1

        print(f"\nSummary: {promoted} promoted, {failed} failed")
    else:
        # Promote specific test
        backends = args.backends or [None]  # None means current backend
        for backend in backends:
            if promote_to_validation(
                args.test_name, backend, backend_specific=args.backend_specific
            ):
                ref_type = f"[{backend}]" if args.backend_specific else "(agnostic)"
                print(f"✓ Promoted {args.test_name} {ref_type}")
            else:
                print(f"✗ Failed to promote {args.test_name}")


def list_command(args):
    """Handle list subcommand."""
    if args.backends:
        for backend in args.backends:
            print(f"\nValidation references for {backend}:")
            references = list_validation_references(backend)
            if references:
                for ref in references:
                    print(f"  {ref}")
            else:
                print("  (none)")
    else:
        print("All validation references:")
        references = list_validation_references()
        if references:
            for ref in references:
                print(f"  {ref}")
        else:
            print("  (none)")


def clean_command(args):
    """Handle clean subcommand."""
    backends = args.backends or [None]
    total_deleted = 0

    for backend in backends:
        deleted = cleanup_validation_references(backend, not args.force)
        print(
            f"Deleted {deleted} validation files"
            + (f" for {backend}" if backend else "")
        )
        total_deleted += deleted

    print(f"Total deleted: {total_deleted}")


def info_command(args):
    """Handle info subcommand."""
    backends = args.backends or get_backends()

    print("Validation system status:")
    print(f"  Validation directory: {VALIDATION_DIR}")
    print(f"  Raw queue directory: {RAW_DIR}")
    print(f"  Available backends: {', '.join(get_backends())}")
    print()

    for backend in backends:
        references = list_validation_references(backend)
        # Count raw files ending with [backend].npz
        raw_files = len(
            [f for f in RAW_DIR.glob("*.npz") if f.name.endswith(f"[{backend}].npz")]
        )

        print(f"Backend: {backend}")
        print(f"  Validation references: {len(references)}")
        print(f"  Raw test files: {raw_files}")

        if args.verbose and references:
            print("  Available references:")
            for ref in references[:10]:  # Show first 10
                print(f"    {ref}")
            if len(references) > 10:
                print(f"    ... and {len(references) - 10} more")
        print()


def inspect_command(args):
    """Handle inspect subcommand."""
    backends = args.backends or get_backends()

    for backend in backends:
        print(f"Inspecting {args.test_name} for backend {backend}:")

        reference = load_validation_reference(args.test_name, backend)
        if reference is not None:
            print(
                f"  Validation reference: shape={reference.shape}, dtype={reference.dtype}"
            )
            print(
                f"  Sum: {reference.sum():.6f}, Min: {reference.min():.6f}, Max: {reference.max():.6f}"
            )
        else:
            print("  No validation reference found")

        # Check if raw file exists
        key = f"{args.test_name} [{backend}]"
        raw_file = RAW_DIR / f"{key}.npz"
        if raw_file.exists():
            import numpy as np

            data = np.load(raw_file)
            array = data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]
            print(f"  Raw file: shape={array.shape}, dtype={array.dtype}")
            print(
                f"  Sum: {array.sum():.6f}, Min: {array.min():.6f}, Max: {array.max():.6f}"
            )
        else:
            print("  No raw file found")
        print()


def validate_command(args):
    """Handle validate subcommand - systematically check all test results against validation references."""
    import numpy as np

    from yapocis.utils.validation import compare_with_validation

    if not RAW_DIR.exists():
        print(f"Raw directory not found: {RAW_DIR}")
        return

    print(f"Systematic validation check with tolerance: {args.tolerance}")
    print(f"Raw directory: {RAW_DIR}")
    print(f"Validation directory: {VALIDATION_DIR}")
    print()

    # Collect all test results by parsing raw files
    all_tests = {}
    for npz_file in RAW_DIR.glob("*.npz"):
        test_name, backend = parse_test_name(npz_file.name)
        if test_name and backend:
            if test_name not in all_tests:
                all_tests[test_name] = {}
            all_tests[test_name][backend] = npz_file

    # Statistics
    total_tests = 0
    validation_found = 0
    validation_passed = 0
    validation_failed = 0
    no_validation = 0

    # Group results by status
    results = {"passed": [], "failed": [], "no_validation": [], "errors": []}

    print(
        f"Found {len(all_tests)} unique tests across {sum(len(backends) for backends in all_tests.values())} backend results\n"
    )

    for test_name in sorted(all_tests.keys()):
        backends = all_tests[test_name]

        if not args.summary_only:
            print(f"📋 {test_name}")

        for backend in sorted(backends.keys()):
            total_tests += 1
            npz_file = backends[backend]

            try:
                # Load test result
                data = np.load(npz_file)
                test_result = (
                    data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]
                )

                # Compare with validation reference
                matches, info = compare_with_validation(
                    test_name,
                    test_result,
                    backend,
                    atol=args.tolerance,
                    rtol=args.tolerance,
                )

                if info["reference_exists"]:
                    validation_found += 1
                    if matches:
                        validation_passed += 1
                        status = "✅ PASS"
                        results["passed"].append((test_name, backend, info))
                    else:
                        validation_failed += 1
                        status = f"❌ FAIL ({info['status']})"
                        if info["status"] == "numerical_mismatch":
                            max_diff = info.get("max_abs_diff", 0)
                            status += f" max_diff={max_diff:.2e}"
                        results["failed"].append((test_name, backend, info))
                else:
                    no_validation += 1
                    status = "⚪ NO_REF"
                    results["no_validation"].append((test_name, backend, info))

                if not args.summary_only:
                    ref_type = "(agnostic)" if info.get("reference_exists") else ""
                    print(f"  [{backend:6s}] {status} {ref_type}")

            except Exception as e:
                results["errors"].append((test_name, backend, str(e)))
                if not args.summary_only:
                    print(f"  [{backend:6s}] ❌ ERROR: {str(e)[:50]}")

        if not args.summary_only:
            print()

    # Summary statistics
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total test results:        {total_tests}")
    print(
        f"With validation reference: {validation_found} ({validation_found/total_tests*100:.1f}%)"
    )
    print(
        f"Without validation ref:    {no_validation} ({no_validation/total_tests*100:.1f}%)"
    )
    print()
    print("Validation results:")
    print(
        f"  ✅ Passed:  {validation_passed}/{validation_found} ({validation_passed/max(validation_found,1)*100:.1f}%)"
    )
    print(
        f"  ❌ Failed:  {validation_failed}/{validation_found} ({validation_failed/max(validation_found,1)*100:.1f}%)"
    )
    print(f"  🚨 Errors:  {len(results['errors'])}")

    # Show failed validations if any
    if results["failed"] and not args.summary_only:
        print("\n" + "=" * 60)
        print("FAILED VALIDATIONS")
        print("=" * 60)
        for test_name, backend, info in results["failed"]:
            print(f"❌ {test_name} [{backend}]: {info['status']}")
            if info["status"] == "numerical_mismatch":
                print(f"   Max difference: {info.get('max_abs_diff', 0):.2e}")

    # Show tests without validation references
    if (
        results["no_validation"]
        and not args.summary_only
        and len(results["no_validation"]) <= 10
    ):
        print("\n" + "=" * 60)
        print("TESTS WITHOUT VALIDATION REFERENCES")
        print("=" * 60)
        for test_name, backend, info in results["no_validation"]:
            print(f"⚪ {test_name} [{backend}]")
    elif len(results["no_validation"]) > 10 and not args.summary_only:
        print(
            f"\n({len(results['no_validation'])} tests without validation references - use --summary-only for list)"
        )

    # Return status code for CI/CD
    return 0 if validation_failed == 0 and len(results["errors"]) == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Manage yapocis validation references")

    # Global options
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=get_backends(),
        help="Target backends (default: all available)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Promote command
    promote_parser = subparsers.add_parser(
        "promote", help="Promote test results to validation references"
    )
    promote_parser.add_argument(
        "test_name", help='Test name to promote, or "all" for bulk promotion'
    )
    promote_parser.add_argument(
        "--backend-specific",
        action="store_true",
        help="Create backend-specific references (default: agnostic)",
    )
    promote_parser.set_defaults(func=promote_command)

    # list command
    list_parser = subparsers.add_parser("list", help="list validation references")
    list_parser.set_defaults(func=list_command)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up validation references")
    clean_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    clean_parser.set_defaults(func=clean_command)

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show validation system information"
    )
    info_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information"
    )
    info_parser.set_defaults(func=info_command)

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a specific test validation data"
    )
    inspect_parser.add_argument("test_name", help="Test name to inspect")
    inspect_parser.set_defaults(func=inspect_command)

    # Validate command - systematic validation checking
    validate_parser = subparsers.add_parser(
        "validate", help="Systematically validate all test results against references"
    )
    validate_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Validation tolerance (default: 1e-5 for ~5 significant digits)",
    )
    validate_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only summary statistics, not individual results",
    )
    validate_parser.set_defaults(func=validate_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
