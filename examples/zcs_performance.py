#!/usr/bin/env python3
"""
Multi-backend multiscale zero-crossing performance comparison.

Benchmarks Difference of Gaussians (DoG) + zero-crossing detection using async
resident buffers across all available backends (OpenCL, Metal, CUDA, Numba),
comparing to scipy baseline.

Features:
- Automatic backend and device discovery
- Async resident buffer operations for maximum performance
- Multiple scale levels (edge detection at different resolutions)
- Elegant seaborn visualization
- Logarithmic scaling for image sizes

Usage:
    python examples/zcs_performance.py
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import os
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import matplotlib

try:
    from scipy.ndimage import gaussian_filter
    from matplotlib import pyplot as plt
    import seaborn as sns
except ImportError:
    print("Please install scipy, matplotlib, and seaborn:")
    print("  pip install scipy matplotlib seaborn")
    sys.exit(1)

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yapocis.core import get_backends
from yapocis.lib.gaussian import Gaussian


def zcsdog_scipy(input_img: np.ndarray, scale: int = 0, iterations: int = 1) -> tuple:
    """Reference scipy DoG + zero-crossing implementation."""
    sigma1 = 1.6 ** scale
    sigma2 = 1.6 ** (scale + 1)

    for _ in range(iterations):
        # Compute Difference of Gaussians
        smaller = gaussian_filter(input_img, sigma=sigma1)
        larger = gaussian_filter(input_img, sigma=sigma2)
        dog = smaller - larger

        # Simple zero-crossing detection
        # Detect sign changes in horizontal and vertical directions
        zc = np.zeros_like(dog, dtype=np.float32)

        # Horizontal zero crossings (convert bool to float)
        zc[:-1, :] = np.maximum(zc[:-1, :], ((dog[:-1, :] * dog[1:, :]) < 0).astype(np.float32))

        # Vertical zero crossings (convert bool to float)
        zc[:, :-1] = np.maximum(zc[:, :-1], ((dog[:, :-1] * dog[:, 1:]) < 0).astype(np.float32))

        # Simple gradient approximation
        grad = np.abs(np.gradient(smaller)[0]) + np.abs(np.gradient(smaller)[1])
        theta = np.arctan2(np.gradient(smaller)[0], np.gradient(smaller)[1])

    return zc.astype(np.float32), grad.astype(np.float32), theta.astype(np.float32)


def get_backend_devices(backend: str) -> dict[str, str]:
    """
    Get available devices for a given backend.

    Returns:
        Dict mapping device name to full identifier (backend:device)
    """
    devices = {}

    try:
        if backend == 'ocl':
            from yapocis.ocl.device import get_available_devices
            dev_dict = get_available_devices(aliases=False)
            for name in dev_dict.keys():
                if isinstance(name, str):
                    devices[name] = f"ocl:{name}"

        elif backend == 'metal':
            from yapocis.metallic.device import get_available_devices
            dev_dict = get_available_devices(aliases=False)
            for name in dev_dict.keys():
                if isinstance(name, str):
                    devices[name] = f"metal:{name}"

        elif backend == 'cuda':
            from yapocis.cu.device import get_available_devices
            dev_dict = get_available_devices(aliases=False)
            for name in dev_dict.keys():
                if isinstance(name, str):
                    devices[name] = f"cuda:{name}"

        elif backend == 'numbax':
            # Numba is CPU-based, single "device"
            devices['CPU (Numba JIT)'] = 'numbax:cpu'

    except Exception as e:
        print(f"Warning: Could not enumerate {backend} devices: {e}")

    return devices


def benchmark_zcsdog(
    gaussian_filter,
    images: list[np.ndarray],
    widths: np.ndarray,
    scale: int,
    iterations: int,
    label: str,
    use_async: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benchmark multiscale zero-crossing on a series of images.

    Args:
        gaussian_filter: Gaussian filter instance to call zcsdog on
        images: List of test images
        widths: Array of image widths
        scale: Scale parameter for DoG
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

    print(f"\nBenchmarking {label} (scale={scale}, {'async' if use_async else 'sync'})...")
    for width, img in zip(widths, images):
        try:
            # Warmup
            _ = gaussian_filter.zcsdog(img, scale=scale, res=True)

            # Timed run
            t_start = time.time()
            for _ in range(iterations):
                _ = gaussian_filter.zcsdog(img, scale=scale, res=True)
            t_elapsed = time.time() - t_start

            areas.append((width * width) / 1000.0)  # kilopixels
            times.append(t_elapsed)

            print(f"  {width:4d}x{width:4d}: {t_elapsed:.4f}s")

        except Exception as e:
            print(f"  {width:4d}x{width:4d}: FAILED - {e}")
            continue

    return np.array(areas), np.array(times)


def benchmark_scipy(
    images: list[np.ndarray],
    widths: np.ndarray,
    scale: int,
    iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark scipy baseline."""
    areas = []
    times = []

    print(f"\nBenchmarking scipy (scale={scale})...")
    for width, img in zip(widths, images):
        try:
            # Warmup
            _ = zcsdog_scipy(img, scale=scale, iterations=1)

            # Timed run
            t_start = time.time()
            _ = zcsdog_scipy(img, scale=scale, iterations=iterations)
            t_elapsed = time.time() - t_start

            areas.append((width * width) / 1000.0)  # kilopixels
            times.append(t_elapsed)

            print(f"  {width:4d}x{width:4d}: {t_elapsed:.4f}s")

        except Exception as e:
            print(f"  {width:4d}x{width:4d}: FAILED - {e}")
            continue

    return np.array(areas), np.array(times)


def main():
    # Configuration
    ITERATIONS = 10
    SAMPLES = 20
    SMALLEST = 100
    LARGEST = 1000
    SCALE = 2  # DoG scale level (0=finest detail, higher=coarser features)

    print("=" * 70)
    print("Multi-Backend Multiscale Zero-Crossing Performance Comparison")
    print("=" * 70)
    print(f"Image sizes: {SMALLEST}x{SMALLEST} to {LARGEST}x{LARGEST} ({SAMPLES} samples)")
    print(f"Iterations per test: {ITERATIONS}")
    print(f"DoG scale level: {SCALE} (sigma ratio: 1.6^{SCALE} vs 1.6^{SCALE+1})")
    print(f"Testing: Async vs Sync resident buffers")
    print()

    # Generate test images (logarithmically spaced sizes)
    widths = np.logspace(
        np.log2(SMALLEST),
        np.log2(LARGEST),
        num=SAMPLES,
        base=2
    ).astype(int)

    # Create test images with some structure (not just noise)
    images = []
    for w in widths:
        img = np.zeros((w, w), dtype=np.float32)
        # Add some geometric features
        center = w // 2
        radius = w // 4
        y, x = np.ogrid[:w, :w]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        img[mask] = 1.0
        # Add some lines
        img[center-5:center+5, :] = 0.5
        img[:, center-5:center+5] = 0.5
        images.append(img)

    # Benchmark scipy baseline
    print("\n" + "=" * 70)
    print("Scipy Baseline")
    print("=" * 70)
    scipy_areas, scipy_times = benchmark_scipy(images, widths, SCALE, ITERATIONS)

    # Discover backends and devices
    available_backends = get_backends()
    print("\n" + "=" * 70)
    print(f"Available backends: {', '.join(available_backends)}")
    print("=" * 70)

    # Collect results for all backend/device combinations
    results = []

    for backend in available_backends:
        # Set backend environment variable
        os.environ['YAPOCIS_BACKEND'] = backend

        devices = get_backend_devices(backend)

        if not devices:
            print(f"\nNo devices found for {backend}, skipping...")
            continue

        for device_name, device_id in devices.items():
            label = f"{backend.upper()}: {device_name}"

            try:
                # Create Gaussian filter instance for this device
                gaussian = Gaussian(device_name=device_name)

                # Benchmark with ASYNC resident buffers
                areas_async, times_async = benchmark_zcsdog(
                    gaussian,
                    images,
                    widths,
                    SCALE,
                    ITERATIONS,
                    label,
                    use_async=True
                )

                # Benchmark with SYNC resident buffers
                areas_sync, times_sync = benchmark_zcsdog(
                    gaussian,
                    images,
                    widths,
                    SCALE,
                    ITERATIONS,
                    label,
                    use_async=False
                )

                if len(areas_async) > 0 and len(areas_sync) > 0:
                    # Calculate speedup over scipy (if available)
                    if len(scipy_times) > 0:
                        # Align arrays - use minimum length
                        min_len = min(len(scipy_times), len(times_async))
                        speedup_async = scipy_times[:min_len] / times_async[:min_len]
                        speedup_sync = scipy_times[:min_len] / times_sync[:min_len]
                    else:
                        # No scipy baseline - set to NaN
                        speedup_async = np.full_like(times_async, np.nan)
                        speedup_sync = np.full_like(times_sync, np.nan)

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
    if len(scipy_times) > 0:
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
        f'Multiscale Zero-Crossing Performance (DoG scale={SCALE}, {ITERATIONS} iterations)\n'
        f'Async vs Sync Resident Buffers',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Speedup over scipy (async mode) - only if scipy baseline exists
    if len(scipy_times) > 0:
        for result in results:
            # Filter out NaN values
            valid = ~np.isnan(result['speedup_async'])
            if np.any(valid):
                ax2.plot(result['areas'][valid], result['speedup_async'][valid],
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
    else:
        # No scipy baseline - show relative performance instead
        for result in results:
            # Normalize to first backend's performance
            if len(results) > 0:
                baseline_time = results[0]['times_async']
                min_len = min(len(baseline_time), len(result['times_async']))
                relative_perf = baseline_time[:min_len] / result['times_async'][:min_len]
                ax2.plot(result['areas'][:min_len], relative_perf,
                        linewidth=2,
                        marker='o',
                        markersize=5,
                        label=f"{result['label']} (async)")

        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                    label=f'{results[0]["label"]} baseline', alpha=0.7)
        ax2.set_ylabel('Relative Performance', fontsize=11, fontweight='bold')
        ax2.set_title('Backend Comparison (async mode, higher is better)',
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
    output_path = Path(__file__).parent / "zcs_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Performance Summary (at largest image size)")
    print("=" * 70)

    if len(scipy_times) > 0:
        print(f"{'Backend:Device':<35} {'Async (s)':>11} {'Sync (s)':>11} {'vs Scipy':>10} {'Async Gain':>12}")
        print("-" * 70)
        print(f"{'Scipy baseline':<35} {'-':>11} {scipy_times[-1]:>11.4f} {'1.00x':>10} {'-':>12}")

        for result in results:
            if len(result['times_async']) > 0 and len(result['times_sync']) > 0:
                time_async = result['times_async'][-1]
                time_sync = result['times_sync'][-1]
                speedup = result['speedup_async'][-1]
                async_gain = result['async_speedup'][-1]
                speedup_str = f"{speedup:>9.2f}x" if not np.isnan(speedup) else "N/A"
                print(f"{result['label']:<35} {time_async:>11.4f} {time_sync:>11.4f} "
                      f"{speedup_str:>10} {async_gain:>11.2f}x")
    else:
        print(f"{'Backend:Device':<35} {'Async (s)':>11} {'Sync (s)':>11} {'Async Gain':>12}")
        print("-" * 70)

        for result in results:
            if len(result['times_async']) > 0 and len(result['times_sync']) > 0:
                time_async = result['times_async'][-1]
                time_sync = result['times_sync'][-1]
                async_gain = result['async_speedup'][-1]
                print(f"{result['label']:<35} {time_async:>11.4f} {time_sync:>11.4f} "
                      f"{async_gain:>11.2f}x")

    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
