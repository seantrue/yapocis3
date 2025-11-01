#!/usr/bin/env python3
"""
Multi-backend median filter performance comparison.

Benchmarks 3x3 median filter using async resident buffers across all available
backends (OpenCL, Metal, CUDA, Numba) and devices, comparing to scipy baseline.

Features:
- Automatic backend and device discovery
- Backend and device filtering via command-line args
- Async resident buffer ping-pong for maximum performance
- Elegant seaborn visualization
- Logarithmic scaling for image sizes

Usage:
    python examples/median_performance.py
    python examples/median_performance.py --backends ocl,cuda
    python examples/median_performance.py --devices "GeForce GTX","Quadro"
    python examples/median_performance.py --backends cuda --devices "Quadro GP100"
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import argparse
import os
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import matplotlib
#matplotlib.use("module://yapocis.lib.mplbackend")

try:
    from scipy.ndimage.filters import median_filter
    from matplotlib import pyplot as plt
    import seaborn as sns
except ImportError:
    print("Please install scipy, matplotlib, and seaborn:")
    print("  pip install scipy matplotlib seaborn")
    sys.exit(1)

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yapocis.core import get_test_backends, get_test_devices
from yapocis.lib.median import Median


def median3x3_scipy(input_img: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Reference scipy median filter implementation."""
    result = input_img
    for _ in range(iterations):
        result = median_filter(result, 3)
    return result


def benchmark_median(
    median_filter,
    images: list[np.ndarray],
    widths: np.ndarray,
    iterations: int,
    label: str,
    use_async: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benchmark median filter on a series of images.

    Args:
        median_filter: Function to call for filtering
        images: List of test images
        widths: Array of image widths
        iterations: Number of iterations per test
        label: Label for this benchmark
        use_async: Whether to enable async mode

    Returns:
        (areas, times) where areas are in kilopixels and times in seconds
    """
    # Set async mode
    os.environ['YAPOCIS_ASYNC'] = '1' if use_async else '0'

    areas = []
    times = []

    print(f"\nBenchmarking {label} ({'async' if use_async else 'sync'})...")
    for width, img in zip(widths, images):
        try:
            # Warmup
            _ = median_filter(img, iterations=1)

            # Timed run
            t_start = time.time()
            _ = median_filter(img, iterations=iterations)
            t_elapsed = time.time() - t_start

            areas.append((width * width) / 1000.0)  # kilopixels
            times.append(t_elapsed)

            print(f"  {width:4d}x{width:4d}: {t_elapsed:.4f}s")

        except Exception as e:
            print(f"  {width:4d}x{width:4d}: FAILED - {e}")
            continue

    return np.array(areas), np.array(times)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Multi-backend median filter performance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all backends and devices
  python median_performance.py

  # Test only CUDA and OpenCL backends
  python median_performance.py --backends cuda,ocl

  # Test only devices with "GTX" or "Quadro" in name
  python median_performance.py --devices "GTX,Quadro"

  # Test CUDA backend on Quadro device only
  python median_performance.py --backends cuda --devices Quadro
        """
    )
    parser.add_argument('--backends', type=str, help='Comma-separated list of backends to test (e.g., ocl,cuda,metal)')
    parser.add_argument('--devices', type=str, help='Comma-separated list of device name filters (substring match)')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations per test (default: 20)')
    parser.add_argument('--samples', type=int, default=20, help='Number of image size samples (default: 20)')

    args = parser.parse_args()

    # Parse filters
    backend_filters = [b.strip() for b in args.backends.split(',')] if args.backends else []
    device_filters = [d.strip() for d in args.devices.split(',')] if args.devices else []

    # Configuration
    ITERATIONS = args.iterations
    SAMPLES = args.samples
    SMALLEST = 100
    LARGEST = 1000

    print("=" * 70)
    print("Multi-Backend Median Filter Performance Comparison")
    print("=" * 70)
    print(f"Image sizes: {SMALLEST}x{SMALLEST} to {LARGEST}x{LARGEST} ({SAMPLES} samples)")
    print(f"Iterations per test: {ITERATIONS}")
    print(f"Testing: Async vs Sync resident buffers (ping-pong)")
    if backend_filters:
        print(f"Backend filter: {', '.join(backend_filters)}")
    if device_filters:
        print(f"Device filter: {', '.join(device_filters)}")
    print()

    # Generate test images (logarithmically spaced sizes)
    widths = np.logspace(
        np.log2(SMALLEST),
        np.log2(LARGEST),
        num=SAMPLES,
        base=2
    ).astype(int)

    images = [
        np.float32(np.random.random_sample((w, w)))
        for w in widths
    ]

    # Benchmark scipy baseline
    print("\n" + "=" * 70)
    print("Scipy Baseline")
    print("=" * 70)
    scipy_areas, scipy_times = benchmark_median(
        median3x3_scipy,
        images,
        widths,
        ITERATIONS,
        "scipy median_filter"
    )

    # Discover backends and devices using new selection functions
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    print("\n" + "=" * 70)
    print(f"Testing backends: {', '.join(available_backends)}")
    print("=" * 70)

    # Collect results for all backend/device combinations
    results = []

    for backend in available_backends:
        # Set backend environment variable
        os.environ['YAPOCIS_BACKEND'] = backend

        devices = get_test_devices(backend, device_filters if device_filters else None)

        if not devices:
            print(f"\nNo devices found for {backend}, skipping...")
            continue

        for device_name, device_id in devices.items():
            label = f"{backend.upper()}: {device_name}"

            try:
                # Create median filter instance for this device
                # Extract just device name (after colon)
                device_only = device_name
                median = Median(device_name=device_only)

                # Benchmark with ASYNC resident buffers
                areas_async, times_async = benchmark_median(
                    median.median3x3fast,
                    images,
                    widths,
                    ITERATIONS,
                    label,
                    use_async=True
                )

                # Benchmark with SYNC resident buffers
                areas_sync, times_sync = benchmark_median(
                    median.median3x3fast,
                    images,
                    widths,
                    ITERATIONS,
                    label,
                    use_async=False
                )

                if len(areas_async) > 0 and len(areas_sync) > 0:
                    # Calculate speedup over scipy
                    speedup_async = scipy_times[:len(times_async)] / times_async
                    speedup_sync = scipy_times[:len(times_sync)] / times_sync

                    # Calculate async speedup (async vs sync)
                    async_speedup = times_sync / times_async

                    results.append({
                        'backend': backend.upper(),
                        'device': device_name,
                        'label': label,
                        'areas': areas_async,
                        'times_async': times_async,
                        'times_sync': times_sync,
                        'speedup_async': speedup_async,
                        'speedup_sync': speedup_sync,
                        'async_speedup': async_speedup
                    })

            except Exception as e:
                print(f"\n  ERROR: {label} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    if not results:
        print("\nNo successful benchmarks, cannot plot.")
        return 1

    # Create visualization with seaborn
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="husl")
    sns.set_context("notebook", font_scale=1.0)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # Plot 1: Execution time comparison (async vs sync)
    ax1.plot(scipy_areas, scipy_times,
             linewidth=2.5,
             linestyle='--',
             marker='o',
             markersize=6,
             label='Scipy (CPU baseline)',
             color='gray',
             alpha=0.7,
             zorder=1)

    # Plot async and sync for each backend
    for i, result in enumerate(results):
        # Async (solid line)
        ax1.plot(result['areas'], result['times_async'],
                linewidth=2,
                marker='o',
                markersize=5,
                linestyle='-',
                label=f"{result['label']} (async)",
                zorder=2)
        # Sync (dashed line, same color)
        ax1.plot(result['areas'], result['times_sync'],
                linewidth=2,
                marker='s',
                markersize=4,
                linestyle='--',
                alpha=0.6,
                label=f"{result['label']} (sync)",
                zorder=2)

    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title(
        f'3x3 Median Filter Performance ({ITERATIONS} iterations)\n'
        f'Async vs Sync Resident Buffers',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Speedup over scipy (async mode)
    for result in results:
        ax2.plot(result['areas'], result['speedup_async'],
                linewidth=2,
                marker='o',
                markersize=5,
                label=f"{result['label']} (async)")

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                label='Scipy baseline', alpha=0.7)
    ax2.set_ylabel('Speedup vs Scipy', fontsize=11, fontweight='bold')
    ax2.set_title('GPU Acceleration (async mode, higher is better)',
                  fontsize=11, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Async benefit (async speedup over sync)
    for result in results:
        ax3.plot(result['areas'], result['async_speedup'],
                linewidth=2,
                marker='d',
                markersize=5,
                label=result['label'])

    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                label='No benefit', alpha=0.7)
    ax3.set_xlabel('Image Size (kilopixels)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Async Speedup', fontsize=11, fontweight='bold')
    ax3.set_title('Async vs Sync Performance Gain (higher is better)',
                  fontsize=11, fontweight='bold', pad=15)
    ax3.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.99, 0.01, f'Generated: {timestamp}',
             ha='right', va='bottom', fontsize=8, alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "median_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Performance Summary (at largest image size)")
    print("=" * 70)
    print(f"{'Backend:Device':<35} {'Async (s)':>11} {'Sync (s)':>11} {'vs Scipy':>10} {'Async Gain':>12}")
    print("-" * 70)
    print(f"{'Scipy baseline':<35} {'-':>11} {scipy_times[-1]:>11.4f} {'1.00x':>10} {'-':>12}")

    for result in results:
        if len(result['times_async']) > 0 and len(result['times_sync']) > 0:
            time_async = result['times_async'][-1]
            time_sync = result['times_sync'][-1]
            speedup = result['speedup_async'][-1]
            async_gain = result['async_speedup'][-1]
            print(f"{result['label']:<35} {time_async:>11.4f} {time_sync:>11.4f} "
                  f"{speedup:>9.2f}x {async_gain:>11.2f}x")

    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
