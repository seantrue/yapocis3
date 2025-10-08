
#!/usr/bin/env python3
"""Analyze NPZ files in queue/raw for dtype, min/max, and NaN status"""

import argparse
import numpy as np
import os
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Analyze NPZ files for dtype, min/max, and NaN status"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="queue/raw",
        help="Directory containing NPZ files (default: queue/raw)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        help="Only show files matching this pattern",
    )
    parser.add_argument(
        "--nan-only",
        action="store_true",
        help="Only show files containing NaN values",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all array details even for files without issues",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Show only summary statistics, not individual file details",
    )
    parser.add_argument(
        "--dtype",
        "-d",
        help="Only show arrays with this dtype (e.g., float32, uint8)",
    )

    args = parser.parse_args()

    # Resolve directory path
    queue_dir = Path(args.directory)
    if not queue_dir.is_absolute():
        # Try relative to current directory first
        if not queue_dir.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent
            queue_dir = script_dir / args.directory

    if not queue_dir.exists():
        print(f"Error: Directory {queue_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Get list of NPZ files
    npz_files = sorted(queue_dir.glob("*.npz"))

    if args.filter:
        npz_files = [f for f in npz_files if args.filter in f.name]

    if not npz_files:
        print(f"No NPZ files found in {queue_dir}")
        sys.exit(0)

    # Statistics tracking
    total_files = len(npz_files)
    files_with_nan = 0
    files_with_inf = 0
    total_arrays = 0
    dtype_counts = {}  # Track arrays by dtype

    if args.summary:
        print(f"Analyzing {total_files} NPZ files from {queue_dir}")
        print("="*80)
    elif not args.nan_only:
        print(f"Found {total_files} NPZ files\n")
        print("="*80)

    for npz_file in npz_files:
        file_has_nan = False
        file_has_inf = False

        try:
            data = np.load(npz_file)

            # list all arrays in the file
            array_names = list(data.keys())

            # Check for NaN/Inf first and collect dtype stats
            arrays_to_show = []
            for name in array_names:
                arr = data[name]

                # Track dtype
                dtype_str = str(arr.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

                # Filter by dtype if specified
                if args.dtype and dtype_str != args.dtype:
                    continue

                arrays_to_show.append(name)
                total_arrays += 1

                if np.issubdtype(arr.dtype, np.floating):
                    if np.any(np.isnan(arr)):
                        file_has_nan = True
                    if np.any(np.isinf(arr)):
                        file_has_inf = True

            # Skip file if no arrays match filters
            if not arrays_to_show:
                data.close()
                continue

            # Skip file if --nan-only and no NaN
            if args.nan_only and not file_has_nan:
                data.close()
                continue

            # Skip detailed output if --summary
            if args.summary:
                if file_has_nan:
                    files_with_nan += 1
                if file_has_inf:
                    files_with_inf += 1
                data.close()
                continue

            # Print file header
            print(f"\nFile: {npz_file.name}")
            print("-"*80)
            print(f"Arrays: {arrays_to_show}")

            for name in arrays_to_show:
                arr = data[name]
                print(f"\n  Array '{name}':")
                print(f"    Shape: {arr.shape}")
                print(f"    Dtype: {arr.dtype}")

                if np.issubdtype(arr.dtype, np.floating):
                    has_nan = np.any(np.isnan(arr))
                    has_inf = np.any(np.isinf(arr))

                    if has_nan or has_inf or args.verbose:
                        if has_nan or has_inf:
                            print(f"    ⚠️  Has NaN: {has_nan}")
                            print(f"    ⚠️  Has Inf: {has_inf}")
                            if has_nan:
                                nan_count = np.sum(np.isnan(arr))
                                print(f"    NaN count: {nan_count} ({100*nan_count/arr.size:.2f}%)")
                            if has_inf:
                                inf_count = np.sum(np.isinf(arr))
                                print(f"    Inf count: {inf_count} ({100*inf_count/arr.size:.2f}%)")

                        # Get min/max excluding NaN/Inf
                        finite_vals = arr[np.isfinite(arr)]
                        if len(finite_vals) > 0:
                            print(f"    Min (finite): {np.min(finite_vals)}")
                            print(f"    Max (finite): {np.max(finite_vals)}")
                            if args.verbose:
                                print(f"    Mean (finite): {np.mean(finite_vals):.6f}")
                        else:
                            print(f"    ⚠️  No finite values!")
                else:
                    if args.verbose:
                        print(f"    Min: {np.min(arr)}")
                        print(f"    Max: {np.max(arr)}")

            if file_has_nan:
                files_with_nan += 1
            if file_has_inf:
                files_with_inf += 1

            data.close()

        except Exception as e:
            print(f"\n  Error reading file {npz_file.name}: {e}")

    # Print summary
    print("\n" + "="*80)
    if args.summary or args.nan_only or args.dtype:
        print(f"\nSummary:")
        print(f"  Total files analyzed: {total_files}")
        print(f"  Total arrays shown: {total_arrays}")
        print(f"  Files with NaN: {files_with_nan}")
        print(f"  Files with Inf: {files_with_inf}")
        print(f"\n  Arrays by dtype:")
        for dtype_name in sorted(dtype_counts.keys()):
            count = dtype_counts[dtype_name]
            print(f"    {dtype_name}: {count}")

if __name__ == "__main__":
    main()
