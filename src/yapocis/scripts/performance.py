#!/usr/bin/env python3
"""
Multi-backend performance testing for yapocis library functions.

Consolidated performance testing tool that benchmarks async vs sync execution
across all available backends and devices for various yapocis library modules.

Supported modules:
- anisodiff: Anisotropic diffusion (Perona-Malik filtering)
- gaussian: Separable Gaussian convolution
- vgg16: VGG16 CNN inference
- resnet50: ResNet50 CNN inference
- graphics4: KernelDraw4 display list rendering
- graphics-aa: Graphics anti-aliasing performance comparison

Features:
- Automatic backend and device discovery
- Backend and device filtering
- Async vs sync comparison
- Elegant matplotlib/seaborn visualizations
- Configurable output directory via YAPOCIS_PERFORMANCE_DIR
- Optional yapocis matplotlib backend for GPU-accelerated plot rendering

Environment Variables:
- YAPOCIS_PERFORMANCE_DIR: Output directory for plots (default: ./performance)
- YAPOCIS_MPLBACKEND: Set to '1' to use yapocis matplotlib backend for rendering

Usage:
    yapocis-performance anisodiff
    yapocis-performance gaussian --backends cuda --devices Quadro
    yapocis-performance vgg16 --iterations 10
    yapocis-performance resnet50 --test-base
    yapocis-performance graphics-aa --backends metal,ocl
    yapocis-performance --list

    # Use yapocis backend for plot rendering (GPU-accelerated)
    YAPOCIS_MPLBACKEND=1 yapocis-performance graphics4
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib

    # Use yapocis backend if requested
    if os.environ.get("YAPOCIS_MPLBACKEND", "").lower() in ("1", "true", "yes"):
        matplotlib.use("module://yapocis.lib.mplbackend", force=True)
        print("Using yapocis matplotlib backend for plot rendering")
    import seaborn as sns
    from matplotlib import pyplot as plt
except ImportError:
    print("Please install matplotlib and seaborn:")
    print("  pip install matplotlib seaborn")
    sys.exit(1)

from yapocis.core import get_test_backends, get_test_devices
from yapocis.images import find_image
from yapocis.utils.basic import imread


def get_output_dir() -> Path:
    """Get output directory for performance plots."""
    perf_dir = os.environ.get("YAPOCIS_PERFORMANCE_DIR")
    if perf_dir:
        output_dir = Path(perf_dir)
    else:
        output_dir = Path.cwd() / "performance"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_test_image(size=None):
    """Load and prepare test image."""
    img_path = find_image("test.jpg")
    img = imread(img_path)

    # Convert to float32
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Resize if requested
    if size is not None:
        from PIL import Image

        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        img = np.array(pil_img).astype(np.float32) / 255.0

    return img


# ============================================================================
# Anisodiff Performance Testing
# ============================================================================


def benchmark_anisodiff(
    anisodiff,
    image: np.ndarray,
    iteration_counts: list[int],
    scale: float,
    step_size: float,
    use_async: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark anisotropic diffusion with varying iteration counts."""
    os.environ["YAPOCIS_ASYNC"] = "1" if use_async else "0"

    iterations = []
    times = []

    for n_iter in iteration_counts:
        try:
            # Warmup
            _ = anisodiff.anisodiff_n(image, n=1, scale=scale, step_size=step_size)

            # Timed run
            t_start = time.time()
            _ = anisodiff.anisodiff_n(image, n=n_iter, scale=scale, step_size=step_size)
            t_elapsed = time.time() - t_start

            iterations.append(n_iter)
            times.append(t_elapsed)

            print(f"  {n_iter:3d} iterations: {t_elapsed:.4f}s")

        except Exception as e:
            print(f"  {n_iter:3d} iterations: FAILED - {e}")
            continue

    return np.array(iterations), np.array(times)


def test_anisodiff(args):
    """Test anisotropic diffusion performance."""
    from yapocis.lib.anisodiff import Anisodiff

    ITERATION_COUNTS = [1, 2, 5, 10, 20, 50, 100]
    IMAGE_SIZE = args.size
    SCALE = 3.0
    STEP_SIZE = 0.2

    print("=" * 70)
    print("Anisotropic Diffusion Performance Test")
    print("=" * 70)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Iteration counts: {ITERATION_COUNTS}")
    print(f"Scale: {SCALE}, Step size: {STEP_SIZE}")
    print()

    # Load test image
    test_image = load_test_image(size=IMAGE_SIZE)
    if len(test_image.shape) == 3:
        test_image = test_image[:, :, 0]
    test_image = (test_image * 256.0).astype(np.float32)

    # Discover backends
    backend_filters = args.backends.split(",") if args.backends else []
    device_filters = args.devices.split(",") if args.devices else []
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    results = []

    for backend in available_backends:
        os.environ["YAPOCIS_BACKEND"] = backend
        devices = get_test_devices(backend, device_filters if device_filters else None)

        if not devices:
            continue

        for device_name, device_id in devices.items():
            label = f"{backend.upper()}: {device_name}"

            try:
                anisodiff = Anisodiff(device_name=device_name)

                print(f"\nBenchmarking {label} (ASYNC)...")
                iters_async, times_async = benchmark_anisodiff(
                    anisodiff,
                    test_image,
                    ITERATION_COUNTS,
                    SCALE,
                    STEP_SIZE,
                    use_async=True,
                )

                print(f"\nBenchmarking {label} (SYNC)...")
                iters_sync, times_sync = benchmark_anisodiff(
                    anisodiff,
                    test_image,
                    ITERATION_COUNTS,
                    SCALE,
                    STEP_SIZE,
                    use_async=False,
                )

                if len(iters_async) > 0 and len(iters_sync) > 0:
                    async_speedup = times_sync / times_async
                    results.append(
                        {
                            "backend": backend.upper(),
                            "device": device_name,
                            "label": label,
                            "iterations": iters_async,
                            "times_async": times_async,
                            "times_sync": times_sync,
                            "async_speedup": async_speedup,
                        }
                    )

            except Exception as e:
                print(f"\n  ERROR: {label} failed: {e}")
                continue

    if not results:
        print("\nNo successful benchmarks.")
        return 1

    # Create visualization
    plot_anisodiff_results(results, IMAGE_SIZE, ITERATION_COUNTS)
    return 0


def plot_anisodiff_results(results, image_size, iteration_counts):
    """Create visualization for anisodiff results."""
    sns.set_theme(style="whitegrid", palette="husl")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for i, result in enumerate(results):
        ax1.plot(
            result["iterations"],
            result["times_async"],
            linewidth=2,
            marker="o",
            markersize=5,
            linestyle="-",
            label=f"{result['label']} (async)",
            zorder=2,
        )
        ax1.plot(
            result["iterations"],
            result["times_sync"],
            linewidth=2,
            marker="s",
            markersize=4,
            linestyle="--",
            alpha=0.6,
            label=f"{result['label']} (sync)",
            zorder=2,
        )

    ax1.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Anisotropic Diffusion Performance ({image_size}x{image_size})\nAsync vs Sync",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    for result in results:
        time_per_iter_async = result["times_async"] / result["iterations"]
        time_per_iter_sync = result["times_sync"] / result["iterations"]
        ax2.plot(
            result["iterations"],
            time_per_iter_async * 1000,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{result['label']} (async)",
        )
        ax2.plot(
            result["iterations"],
            time_per_iter_sync * 1000,
            linewidth=2,
            marker="s",
            markersize=4,
            linestyle="--",
            alpha=0.6,
            label=f"{result['label']} (sync)",
        )

    ax2.set_ylabel("Time per Iteration (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Efficiency", fontsize=11, fontweight="bold", pad=15)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    for result in results:
        ax3.plot(
            result["iterations"],
            result["async_speedup"],
            linewidth=2,
            marker="d",
            markersize=5,
            label=result["label"],
        )

    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.set_xlabel("Number of Iterations", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Async Speedup", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Async vs Sync Performance Gain", fontsize=11, fontweight="bold", pad=15
    )
    ax3.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    plt.tight_layout()
    output_path = get_output_dir() / "anisodiff_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


# ============================================================================
# Gaussian Performance Testing
# ============================================================================


def benchmark_gauss_image(
    gaussian,
    images: list[np.ndarray],
    widths: np.ndarray,
    scale: int,
    iterations: int,
    use_async: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Benchmark gauss_image across different image sizes."""
    os.environ["YAPOCIS_ASYNC"] = "1" if use_async else "0"

    areas = []
    times = []

    for width, img in zip(widths, images, strict=False):
        try:
            # Warmup
            _ = gaussian.gauss_image(img, scale)

            # Timed run
            t_start = time.time()
            for _ in range(iterations):
                _ = gaussian.gauss_image(img, scale)
            t_elapsed = time.time() - t_start

            areas.append((width * width) / 1000.0)
            times.append(t_elapsed)

            print(f"  {width:4d}x{width:4d}: {t_elapsed:.4f}s")

        except Exception as e:
            print(f"  {width:4d}x{width:4d}: FAILED - {e}")
            continue

    return np.array(areas), np.array(times)


def test_gaussian(args):
    """Test Gaussian blur performance."""
    from yapocis.lib.gaussian import Gaussian

    ITERATIONS = args.iterations
    SAMPLES = 15
    NUM_SCALES = 3
    SMALLEST = 100
    LARGEST = 1000

    print("=" * 70)
    print("Gaussian Blur Performance Test")
    print("=" * 70)
    print(
        f"Image sizes: {SMALLEST}x{SMALLEST} to {LARGEST}x{LARGEST} ({SAMPLES} samples)"
    )
    print(f"Iterations per test: {ITERATIONS}")
    print(f"Testing scales: 0 to {NUM_SCALES-1}")
    print()

    # Generate test images
    widths = np.logspace(
        np.log2(SMALLEST), np.log2(LARGEST), num=SAMPLES, base=2
    ).astype(int)
    images = [np.float32(np.random.random_sample((w, w))) * 256.0 for w in widths]

    # Discover backends
    backend_filters = args.backends.split(",") if args.backends else []
    device_filters = args.devices.split(",") if args.devices else []
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    results = []

    for backend in available_backends:
        os.environ["YAPOCIS_BACKEND"] = backend
        devices = get_test_devices(backend, device_filters if device_filters else None)

        if not devices:
            continue

        for device_name, device_id in devices.items():
            label = f"{backend.upper()}: {device_name}"

            try:
                gaussian = Gaussian(device_name=device_name)
                available_scales = gaussian.get_scales()
                test_scales = min(NUM_SCALES, available_scales)

                scale_results = []

                for scale in range(test_scales):
                    print(f"\n{label} - Scale {scale}")
                    print("ASYNC mode...")
                    areas_async, times_async = benchmark_gauss_image(
                        gaussian, images, widths, scale, ITERATIONS, use_async=True
                    )

                    print("SYNC mode...")
                    areas_sync, times_sync = benchmark_gauss_image(
                        gaussian, images, widths, scale, ITERATIONS, use_async=False
                    )

                    if len(areas_async) > 0 and len(areas_sync) > 0:
                        async_speedup = times_sync / times_async
                        scale_results.append(
                            {
                                "scale": scale,
                                "areas": areas_async,
                                "times_async": times_async,
                                "times_sync": times_sync,
                                "async_speedup": async_speedup,
                            }
                        )

                if scale_results:
                    results.append(
                        {
                            "backend": backend.upper(),
                            "device": device_name,
                            "label": label,
                            "scales": scale_results,
                        }
                    )

            except Exception as e:
                print(f"\n  ERROR: {label} failed: {e}")
                continue

    if not results:
        print("\nNo successful benchmarks.")
        return 1

    plot_gaussian_results(results, ITERATIONS, NUM_SCALES)
    return 0


def plot_gaussian_results(results, iterations, num_scales):
    """Create visualization for gaussian results."""
    sns.set_theme(style="whitegrid", palette="husl")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=False)
    colors = sns.color_palette("husl", len(results))

    # Plot 1: Execution time (scale 0)
    for i, result in enumerate(results):
        scale_data = result["scales"][0]
        ax1.plot(
            scale_data["areas"],
            scale_data["times_async"],
            linewidth=2,
            marker="o",
            markersize=5,
            linestyle="-",
            color=colors[i],
            label=f"{result['label']} (async)",
            zorder=2,
        )
        ax1.plot(
            scale_data["areas"],
            scale_data["times_sync"],
            linewidth=2,
            marker="s",
            markersize=4,
            linestyle="--",
            color=colors[i],
            alpha=0.6,
            label=f"{result['label']} (sync)",
            zorder=2,
        )

    ax1.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"Gaussian Blur Performance (scale 0, {iterations} iterations)\nAsync vs Sync",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Speedup range
    for i, result in enumerate(results):
        areas = result["scales"][0]["areas"]
        speedup_data = np.array(
            [scale_data["async_speedup"] for scale_data in result["scales"]]
        )
        speedup_min = np.min(speedup_data, axis=0)
        speedup_max = np.max(speedup_data, axis=0)
        speedup_mean = np.mean(speedup_data, axis=0)

        ax2.plot(
            areas,
            speedup_mean,
            linewidth=2,
            marker="o",
            markersize=4,
            color=colors[i],
            label=f"{result['label']} (mean)",
            zorder=3,
        )
        ax2.fill_between(
            areas,
            speedup_min,
            speedup_max,
            color=colors[i],
            alpha=0.2,
            label=f"{result['label']} (range)",
        )

    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    ax2.set_xlabel("Image Area (kilopixels)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Async Speedup", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Async vs Sync Performance Gain (range across {num_scales} scales)",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Speedup by scale
    for i, result in enumerate(results):
        scales = []
        avg_speedups = []
        for scale_data in result["scales"]:
            scales.append(scale_data["scale"])
            avg_speedups.append(np.mean(scale_data["async_speedup"]))
        ax3.plot(
            scales,
            avg_speedups,
            linewidth=2,
            marker="o",
            markersize=6,
            color=colors[i],
            label=result["label"],
        )

    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.set_xlabel("Gaussian Scale", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Average Async Speedup", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Async Speedup vs Gaussian Scale", fontsize=11, fontweight="bold", pad=15
    )
    ax3.legend(loc="best", framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(num_scales))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    plt.tight_layout()
    output_path = get_output_dir() / "gaussian_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


# ============================================================================
# VGG16 Performance Testing
# ============================================================================


def benchmark_vgg16(vgg, preprocessed_image, iterations, use_async=True):
    """Benchmark VGG16 forward pass."""
    os.environ["YAPOCIS_ASYNC"] = "1" if use_async else "0"
    _ = vgg.forward(preprocessed_image)  # Warmup

    t_start = time.time()
    for _ in range(iterations):
        _ = vgg.forward(preprocessed_image)
    return time.time() - t_start


def test_vgg16(args):
    """Test VGG16 performance."""
    from yapocis.lib.vgg16 import VGG16, VGG16Resident
    from yapocis.models import get_weights_path

    weights_path = get_weights_path("vgg16")
    if weights_path is None:
        print("ERROR: VGG16 weights not found.")
        return 1

    ITERATIONS = args.iterations
    test_image = load_test_image()

    print("=" * 70)
    print("VGG16 Performance Test")
    print("=" * 70)
    print(f"Iterations: {ITERATIONS}")
    print()

    backend_filters = args.backends.split(",") if args.backends else []
    device_filters = args.devices.split(",") if args.devices else []
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    results = []

    for backend in available_backends:
        os.environ["YAPOCIS_BACKEND"] = backend
        devices = get_test_devices(backend, device_filters if device_filters else None)

        for device_name in devices.keys():
            label = f"{backend.upper()}: {device_name}"

            try:
                print(f"\nTesting {label} - VGG16")
                vgg16_base = VGG16(str(weights_path))
                preprocessed = vgg16_base.preprocess_image(test_image)

                vgg16_sync = benchmark_vgg16(
                    vgg16_base, preprocessed, ITERATIONS, False
                )
                print(f"  Sync: {vgg16_sync:.4f}s")
                vgg16_async = benchmark_vgg16(
                    vgg16_base, preprocessed, ITERATIONS, True
                )
                print(f"  Async: {vgg16_async:.4f}s")

                print(f"Testing {label} - VGG16Resident")
                vgg16_resident = VGG16Resident(str(weights_path))

                vgg16_res_sync = benchmark_vgg16(
                    vgg16_resident, preprocessed, ITERATIONS, False
                )
                print(f"  Sync: {vgg16_res_sync:.4f}s")
                vgg16_res_async = benchmark_vgg16(
                    vgg16_resident, preprocessed, ITERATIONS, True
                )
                print(f"  Async: {vgg16_res_async:.4f}s")

                results.append(
                    {
                        "label": label,
                        "vgg16_sync": vgg16_sync,
                        "vgg16_async": vgg16_async,
                        "vgg16_res_sync": vgg16_res_sync,
                        "vgg16_res_async": vgg16_res_async,
                    }
                )

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    if not results:
        print("\nNo successful benchmarks.")
        return 1

    plot_vgg16_results(results, ITERATIONS)
    return 0


def plot_vgg16_results(results, iterations):
    """Create visualization for VGG16 results."""
    sns.set_theme(style="whitegrid", palette="husl")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

    x_pos = np.arange(len(results))
    width = 0.2
    labels = [r["label"] for r in results]

    # Plot 1: Time comparison
    times_base_sync = [r["vgg16_sync"] / iterations for r in results]
    times_base_async = [r["vgg16_async"] / iterations for r in results]
    times_res_sync = [r["vgg16_res_sync"] / iterations for r in results]
    times_res_async = [r["vgg16_res_async"] / iterations for r in results]

    ax1.bar(x_pos - 1.5 * width, times_base_sync, width, label="VGG16 Sync", alpha=0.8)
    ax1.bar(
        x_pos - 0.5 * width, times_base_async, width, label="VGG16 Async", alpha=0.8
    )
    ax1.bar(
        x_pos + 0.5 * width,
        times_res_sync,
        width,
        label="VGG16Resident Sync",
        alpha=0.8,
    )
    ax1.bar(
        x_pos + 1.5 * width,
        times_res_async,
        width,
        label="VGG16Resident Async",
        alpha=0.8,
    )

    ax1.set_ylabel("Time per Inference (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"VGG16 Performance ({iterations} iterations)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Async speedup
    speedup_base = [r["vgg16_sync"] / r["vgg16_async"] for r in results]
    speedup_res = [r["vgg16_res_sync"] / r["vgg16_res_async"] for r in results]

    ax2.bar(x_pos - width / 2, speedup_base, width, label="VGG16", alpha=0.8)
    ax2.bar(x_pos + width / 2, speedup_res, width, label="VGG16Resident", alpha=0.8)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax2.set_ylabel("Async Speedup", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Async vs Sync Performance Gain", fontsize=11, fontweight="bold", pad=15
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Overall speedup
    best_speedup = [r["vgg16_sync"] / r["vgg16_res_async"] for r in results]

    ax3.bar(x_pos, best_speedup, width * 2, alpha=0.8)
    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax3.set_ylabel("Overall Speedup", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Backend:Device", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Best Overall Speedup vs Base VGG16 (sync)",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    plt.tight_layout()
    output_path = get_output_dir() / "vgg16_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


# ============================================================================
# ResNet50 Performance Testing
# ============================================================================


def benchmark_resnet50(resnet, preprocessed_image, iterations, use_async=True):
    """Benchmark ResNet50 forward pass."""
    os.environ["YAPOCIS_ASYNC"] = "1" if use_async else "0"
    _ = resnet.forward(preprocessed_image)  # Warmup

    t_start = time.time()
    for _ in range(iterations):
        _ = resnet.forward(preprocessed_image)
    return time.time() - t_start


def test_resnet50(args):
    """Test ResNet50 performance."""
    from yapocis.lib.resnet50 import ResNet50Resident
    from yapocis.models import get_weights_path

    weights_path = get_weights_path("resnet50")
    if weights_path is None:
        print("ERROR: ResNet50 weights not found.")
        return 1

    ITERATIONS = args.iterations
    test_image = load_test_image()

    print("=" * 70)
    print("ResNet50 Performance Test")
    print("=" * 70)
    print(f"Iterations: {ITERATIONS}")
    print()

    backend_filters = args.backends.split(",") if args.backends else []
    device_filters = args.devices.split(",") if args.devices else []
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    results = []

    for backend in available_backends:
        os.environ["YAPOCIS_BACKEND"] = backend
        devices = get_test_devices(backend, device_filters if device_filters else None)

        for device_name in devices.keys():
            label = f"{backend.upper()}: {device_name}"

            try:
                print(f"\nTesting {label} - ResNet50Resident")
                resnet = ResNet50Resident(str(weights_path))
                preprocessed = resnet.preprocess_image(test_image)

                sync_time = benchmark_resnet50(resnet, preprocessed, ITERATIONS, False)
                print(f"  Sync: {sync_time:.4f}s")
                async_time = benchmark_resnet50(resnet, preprocessed, ITERATIONS, True)
                print(f"  Async: {async_time:.4f}s")

                results.append({"label": label, "sync": sync_time, "async": async_time})

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    if not results:
        print("\nNo successful benchmarks.")
        return 1

    plot_resnet50_results(results, ITERATIONS)
    return 0


def plot_resnet50_results(results, iterations):
    """Create visualization for ResNet50 results."""
    sns.set_theme(style="whitegrid", palette="husl")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x_pos = np.arange(len(results))
    width = 0.35
    labels = [r["label"] for r in results]

    # Plot 1: Time comparison
    sync_times = [r["sync"] / iterations for r in results]
    async_times = [r["async"] / iterations for r in results]

    ax1.bar(x_pos - width / 2, sync_times, width, label="Sync", alpha=0.8)
    ax1.bar(x_pos + width / 2, async_times, width, label="Async", alpha=0.8)

    ax1.set_ylabel("Time per Inference (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"ResNet50 Performance ({iterations} iterations)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Async speedup
    speedups = [r["sync"] / r["async"] for r in results]

    ax2.bar(x_pos, speedups, width * 2, alpha=0.8)
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax2.set_ylabel("Async Speedup", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Backend:Device", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Async vs Sync Performance Gain", fontsize=11, fontweight="bold", pad=15
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    plt.tight_layout()
    output_path = get_output_dir() / "resnet50_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


# ============================================================================
# Graphics4 Performance Testing
# ============================================================================


def benchmark_graphics4(
    kdraw,
    primitive_count: int,
    canvas_size: int,
    iterations: int,
    use_async: bool = True,
) -> float:
    """
    Benchmark graphics4 display list rendering.

    Args:
        kdraw: KernelDraw4 instance with resident canvas
        primitive_count: Number of primitives to render
        canvas_size: Canvas dimension (square)
        iterations: Number of render iterations
        use_async: Whether to use async mode

    Returns:
        Elapsed time in seconds
    """
    os.environ["YAPOCIS_ASYNC"] = "1" if use_async else "0"

    # Generate random primitives
    np.random.seed(42)  # Reproducible data

    # Mix of primitive types (proportions typical of UI rendering)
    n_lines = int(primitive_count * 0.35)
    n_circles = int(primitive_count * 0.25)
    n_rects = int(primitive_count * 0.25)
    n_triangles = int(primitive_count * 0.10)
    n_ellipses = int(primitive_count * 0.05)

    # Generate coordinates scaled to canvas
    lines = np.random.randint(0, canvas_size, size=(n_lines, 4))
    circles = np.column_stack(
        [
            np.random.randint(0, canvas_size, size=n_circles),
            np.random.randint(0, canvas_size, size=n_circles),
            np.random.randint(5, 50, size=n_circles),
        ]
    )
    rects = np.random.randint(0, canvas_size, size=(n_rects, 4))
    triangles = np.random.randint(0, canvas_size, size=(n_triangles, 6))
    ellipses = np.column_stack(
        [
            np.random.randint(0, canvas_size, size=n_ellipses),
            np.random.randint(0, canvas_size, size=n_ellipses),
            np.random.randint(10, 60, size=n_ellipses),
            np.random.randint(10, 60, size=n_ellipses),
        ]
    )

    # Generate random colors
    colors_lines = np.random.rand(n_lines, 4).astype(np.float32)
    colors_circles = np.random.rand(n_circles, 4).astype(np.float32)
    colors_rects = np.random.rand(n_rects, 4).astype(np.float32)
    colors_triangles = np.random.rand(n_triangles, 4).astype(np.float32)
    colors_ellipses = np.random.rand(n_ellipses, 4).astype(np.float32)

    # Warmup iteration
    kdraw.lines(lines, colors=colors_lines)
    kdraw.circles(circles, colors=colors_circles)
    kdraw.rects(rects, colors=colors_rects)
    kdraw.triangles(triangles, colors=colors_triangles)
    kdraw.ellipses(ellipses, colors=colors_ellipses)
    kdraw.render()

    # Timed iterations
    t_start = time.time()
    for _ in range(iterations):
        kdraw.lines(lines, colors=colors_lines)
        kdraw.circles(circles, colors=colors_circles)
        kdraw.rects(rects, colors=colors_rects)
        kdraw.triangles(triangles, colors=colors_triangles)
        kdraw.ellipses(ellipses, colors=colors_ellipses)
        kdraw.render()
    t_elapsed = time.time() - t_start

    return t_elapsed


def test_graphics4(args):
    """Test Graphics4/KernelDraw4 display list rendering performance."""
    from yapocis.lib.kerneldraw4 import KernelDraw4

    PRIMITIVE_COUNTS = [100, 500, 1000, 5000, 10000]
    CANVAS_SIZES = [512, 1024, 2048]
    ITERATIONS = args.iterations

    print("=" * 70)
    print("Graphics4 Display List Rendering Performance Test")
    print("=" * 70)
    print(f"Primitive counts: {PRIMITIVE_COUNTS}")
    print(f"Canvas sizes: {CANVAS_SIZES}")
    print(f"Iterations per test: {ITERATIONS}")
    print()

    # Discover backends
    backend_filters = args.backends.split(",") if args.backends else []
    device_filters = args.devices.split(",") if args.devices else []
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    results = []

    for backend in available_backends:
        os.environ["YAPOCIS_BACKEND"] = backend
        devices = get_test_devices(backend, device_filters if device_filters else None)

        if not devices:
            continue

        for device_name, device_id in devices.items():
            label = f"{backend.upper()}: {device_name}"

            try:
                # Test across canvas sizes
                for canvas_size in CANVAS_SIZES:
                    print(f"\n{label} - Canvas {canvas_size}x{canvas_size}")

                    # Create canvas and KernelDraw4 instance
                    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.float32)
                    kdraw = KernelDraw4(canvas, device_name=device_name)

                    counts = []
                    times_async = []
                    times_sync = []

                    for prim_count in PRIMITIVE_COUNTS:
                        try:
                            print(f"  {prim_count:5d} primitives (ASYNC)...", end=" ")
                            t_async = benchmark_graphics4(
                                kdraw,
                                prim_count,
                                canvas_size,
                                ITERATIONS,
                                use_async=True,
                            )
                            times_async.append(t_async)
                            print(f"{t_async:.4f}s")

                            print(f"  {prim_count:5d} primitives (SYNC)...", end=" ")
                            t_sync = benchmark_graphics4(
                                kdraw,
                                prim_count,
                                canvas_size,
                                ITERATIONS,
                                use_async=False,
                            )
                            times_sync.append(t_sync)
                            print(f"{t_sync:.4f}s")

                            counts.append(prim_count)

                        except Exception as e:
                            print(f"FAILED - {e}")
                            continue

                    if len(counts) > 0:
                        counts = np.array(counts)
                        times_async = np.array(times_async)
                        times_sync = np.array(times_sync)
                        async_speedup = times_sync / times_async

                        results.append(
                            {
                                "backend": backend.upper(),
                                "device": device_name,
                                "label": label,
                                "canvas_size": canvas_size,
                                "counts": counts,
                                "times_async": times_async,
                                "times_sync": times_sync,
                                "async_speedup": async_speedup,
                            }
                        )

            except Exception as e:
                print(f"\n  ERROR: {label} failed: {e}")
                import traceback

                traceback.print_exc()
                continue

    if not results:
        print("\nNo successful benchmarks.")
        return 1

    plot_graphics4_results(results, ITERATIONS, PRIMITIVE_COUNTS, CANVAS_SIZES)
    return 0


def plot_graphics4_results(results, iterations, primitive_counts, canvas_sizes):
    """Create visualization for graphics4 results."""
    sns.set_theme(style="whitegrid", palette="husl")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))

    # Group results by backend/device
    by_device = {}
    for result in results:
        key = result["label"]
        if key not in by_device:
            by_device[key] = []
        by_device[key].append(result)

    colors = sns.color_palette("husl", len(by_device))

    # Plot 1: Execution time vs primitive count (largest canvas)
    largest_canvas = max(canvas_sizes)
    ax1.set_title(
        f"Graphics4 Display List Performance ({largest_canvas}x{largest_canvas}, {iterations} iterations)\n"
        f"Async vs Sync",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    for i, (device_label, device_results) in enumerate(by_device.items()):
        # Find result for largest canvas
        canvas_result = [
            r for r in device_results if r["canvas_size"] == largest_canvas
        ]
        if canvas_result:
            result = canvas_result[0]
            ax1.plot(
                result["counts"],
                result["times_async"],
                linewidth=2,
                marker="o",
                markersize=5,
                linestyle="-",
                color=colors[i],
                label=f"{device_label} (async)",
                zorder=2,
            )
            ax1.plot(
                result["counts"],
                result["times_sync"],
                linewidth=2,
                marker="s",
                markersize=4,
                linestyle="--",
                color=colors[i],
                alpha=0.6,
                label=f"{device_label} (sync)",
                zorder=2,
            )

    ax1.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Primitives", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Plot 2: Throughput (primitives/sec) vs canvas size
    ax2.set_title(
        "Rendering Throughput (async mode)", fontsize=11, fontweight="bold", pad=15
    )

    for i, (device_label, device_results) in enumerate(by_device.items()):
        # Calculate throughput for each canvas size
        canvas_sizes_actual = []
        throughputs = []

        for result in sorted(device_results, key=lambda r: r["canvas_size"]):
            canvas_sizes_actual.append(result["canvas_size"])
            # Use largest primitive count for throughput calculation
            max_count = result["counts"][-1]
            time_per_iter = result["times_async"][-1] / iterations
            throughput = max_count / time_per_iter
            throughputs.append(throughput / 1000.0)  # K primitives/sec

        if canvas_sizes_actual:
            ax2.plot(
                canvas_sizes_actual,
                throughputs,
                linewidth=2,
                marker="o",
                markersize=6,
                color=colors[i],
                label=device_label,
            )

    ax2.set_ylabel("Throughput (K primitives/sec)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Canvas Size (pixels)", fontsize=11, fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Async speedup vs primitive count
    ax3.set_title(
        "Async vs Sync Performance Gain", fontsize=11, fontweight="bold", pad=15
    )

    for i, (device_label, device_results) in enumerate(by_device.items()):
        # Find result for largest canvas
        canvas_result = [
            r for r in device_results if r["canvas_size"] == largest_canvas
        ]
        if canvas_result:
            result = canvas_result[0]
            ax3.plot(
                result["counts"],
                result["async_speedup"],
                linewidth=2,
                marker="d",
                markersize=5,
                color=colors[i],
                label=device_label,
            )

    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.set_ylabel("Async Speedup", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of Primitives", fontsize=11, fontweight="bold")
    ax3.legend(loc="best", framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    plt.tight_layout()
    output_path = get_output_dir() / "graphics4_performance.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


# ============================================================================
# Graphics Anti-Aliasing Performance Testing
# ============================================================================


def benchmark_graphics_aa(
    kdraw,
    primitive_type: str,
    primitive_count: int,
    canvas_size: int,
    aa_width: float,
    iterations: int,
    use_async: bool = True,
) -> float:
    """
    Benchmark graphics rendering with anti-aliasing.

    Args:
        kdraw: KernelDraw4 instance with resident canvas
        primitive_type: Type of primitive ('circles', 'rects', 'triangles', 'ellipses')
        primitive_count: Number of primitives to render
        canvas_size: Canvas dimension (square)
        aa_width: Anti-aliasing width (0 = disabled)
        iterations: Number of render iterations
        use_async: Whether to use async mode

    Returns:
        Elapsed time in seconds
    """
    os.environ["YAPOCIS_ASYNC"] = "1" if use_async else "0"

    # Generate random primitives
    np.random.seed(42)  # Reproducible data

    # Generate coordinates and colors based on primitive type
    if primitive_type == "circles":
        coords = np.column_stack(
            [
                np.random.randint(0, canvas_size, size=primitive_count),
                np.random.randint(0, canvas_size, size=primitive_count),
                np.random.randint(10, 50, size=primitive_count),
            ]
        )
    elif primitive_type == "rects":
        coords = np.random.randint(0, canvas_size, size=(primitive_count, 4))
    elif primitive_type == "triangles":
        coords = np.random.randint(0, canvas_size, size=(primitive_count, 6))
    elif primitive_type == "ellipses":
        coords = np.column_stack(
            [
                np.random.randint(0, canvas_size, size=primitive_count),
                np.random.randint(0, canvas_size, size=primitive_count),
                np.random.randint(15, 60, size=primitive_count),
                np.random.randint(15, 60, size=primitive_count),
            ]
        )
    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    colors = np.random.rand(primitive_count, 4).astype(np.float32)

    # Get the drawing method
    draw_method = getattr(kdraw, primitive_type)

    # Warmup iteration
    draw_method(coords, colors=colors, aa_width=aa_width)
    kdraw.render()

    # Timed iterations
    t_start = time.time()
    for _ in range(iterations):
        draw_method(coords, colors=colors, aa_width=aa_width)
        kdraw.render()
    t_elapsed = time.time() - t_start

    return t_elapsed


def test_graphics_aa(args):
    """Test Graphics4 anti-aliasing performance across backends and async modes."""
    from yapocis.lib.kerneldraw4 import KernelDraw4

    # Configuration
    AA_WIDTHS = [0.0, 0.25, 0.5, 1.0]
    PRIMITIVE_TYPES = ["circles", "rects", "triangles", "ellipses"]
    PRIMITIVE_COUNT = 5000
    CANVAS_SIZE = 1024
    ITERATIONS = args.iterations

    print("=" * 70)
    print("Graphics Anti-Aliasing Performance Test")
    print("=" * 70)
    print(f"AA widths: {AA_WIDTHS}")
    print(f"Primitive types: {PRIMITIVE_TYPES}")
    print(f"Primitives per test: {PRIMITIVE_COUNT}")
    print(f"Canvas size: {CANVAS_SIZE}x{CANVAS_SIZE}")
    print(f"Iterations per test: {ITERATIONS}")
    print()

    # Discover backends
    backend_filters = args.backends.split(",") if args.backends else []
    device_filters = args.devices.split(",") if args.devices else []
    available_backends = get_test_backends(backend_filters if backend_filters else None)

    if not available_backends:
        print("No backends available for testing")
        return 1

    results = []

    for backend in available_backends:
        os.environ["YAPOCIS_BACKEND"] = backend
        devices = get_test_devices(backend, device_filters if device_filters else None)

        if not devices:
            continue

        for device_name, device_id in devices.items():
            label = f"{backend.upper()}: {device_name}"

            try:
                print(f"\n{label}")

                # Create canvas and KernelDraw4 instance
                canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.float32)
                kdraw = KernelDraw4(canvas, device_name=device_name)

                for primitive_type in PRIMITIVE_TYPES:
                    print(f"\n  {primitive_type.capitalize()}:")

                    for aa_width in AA_WIDTHS:
                        # Skip AA tests for numbax (CPU baseline only)
                        if backend == "numbax" and aa_width > 0.0:
                            continue

                        try:
                            # Test with async
                            t_async = benchmark_graphics_aa(
                                kdraw,
                                primitive_type,
                                PRIMITIVE_COUNT,
                                CANVAS_SIZE,
                                aa_width,
                                ITERATIONS,
                                use_async=True,
                            )

                            # Test without async
                            t_sync = benchmark_graphics_aa(
                                kdraw,
                                primitive_type,
                                PRIMITIVE_COUNT,
                                CANVAS_SIZE,
                                aa_width,
                                ITERATIONS,
                                use_async=False,
                            )

                            baseline_note = (
                                " (CPU baseline)" if backend == "numbax" else ""
                            )
                            print(
                                f"    aa={aa_width:4.2f}: async={t_async:.4f}s, sync={t_sync:.4f}s, "
                                f"speedup={t_sync/t_async:.2f}x{baseline_note}"
                            )

                            results.append(
                                {
                                    "backend": backend.upper(),
                                    "device": device_name,
                                    "label": label,
                                    "primitive_type": primitive_type,
                                    "aa_width": aa_width,
                                    "time_async": t_async,
                                    "time_sync": t_sync,
                                    "async_speedup": t_sync / t_async,
                                }
                            )

                        except Exception as e:
                            print(f"    aa={aa_width:4.2f}: FAILED - {e}")
                            continue

            except Exception as e:
                print(f"\n  ERROR: {label} failed: {e}")
                import traceback

                traceback.print_exc()
                continue

    if not results:
        print("\nNo successful benchmarks.")
        return 1

    plot_graphics_aa_results(
        results, ITERATIONS, AA_WIDTHS, PRIMITIVE_TYPES, PRIMITIVE_COUNT, CANVAS_SIZE
    )
    return 0


def plot_graphics_aa_results(
    results, iterations, aa_widths, primitive_types, primitive_count, canvas_size
):
    """Create visualization for anti-aliasing performance results."""
    sns.set_theme(style="whitegrid", palette="husl")
    fig = plt.figure(figsize=(16, 12))

    # Create grid: 2 rows x 2 columns for each primitive type
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Group results by backend/device
    by_device = {}
    for result in results:
        key = result["label"]
        if key not in by_device:
            by_device[key] = []
        by_device[key].append(result)

    colors = sns.color_palette("husl", len(by_device))

    # Create subplot for each primitive type
    for idx, primitive_type in enumerate(primitive_types):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        ax.set_title(
            f"{primitive_type.capitalize()} Anti-Aliasing Performance\n"
            f"({canvas_size}x{canvas_size}, {primitive_count} primitives, {iterations} iterations)",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

        # Plot async and sync times for each device
        for i, (device_label, device_results) in enumerate(by_device.items()):
            # Filter results for this primitive type
            prim_results = [
                r for r in device_results if r["primitive_type"] == primitive_type
            ]
            if not prim_results:
                continue

            # Sort by aa_width
            prim_results = sorted(prim_results, key=lambda r: r["aa_width"])

            aa_vals = [r["aa_width"] for r in prim_results]
            times_async = [r["time_async"] for r in prim_results]
            times_sync = [r["time_sync"] for r in prim_results]

            # Plot async times (solid line)
            ax.plot(
                aa_vals,
                times_async,
                linewidth=2,
                marker="o",
                markersize=5,
                linestyle="-",
                color=colors[i],
                label=f"{device_label} (async)",
                zorder=2,
            )

            # Plot sync times (dashed line)
            ax.plot(
                aa_vals,
                times_sync,
                linewidth=2,
                marker="s",
                markersize=4,
                linestyle="--",
                color=colors[i],
                alpha=0.6,
                label=f"{device_label} (sync)",
                zorder=2,
            )

        ax.set_ylabel("Execution Time (seconds)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Anti-Aliasing Width", fontsize=9, fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(aa_widths)

    # Add overall async speedup comparison plot
    fig2, ax_speedup = plt.subplots(figsize=(14, 6))

    ax_speedup.set_title(
        f"Async vs Sync Performance Gain Across Primitives\n"
        f"({canvas_size}x{canvas_size}, {primitive_count} primitives, {iterations} iterations)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Create grouped bar chart
    x = np.arange(len(aa_widths))
    width = 0.15  # Width of each bar

    for i, (device_label, device_results) in enumerate(by_device.items()):
        for j, primitive_type in enumerate(primitive_types):
            prim_results = [
                r for r in device_results if r["primitive_type"] == primitive_type
            ]
            prim_results = sorted(prim_results, key=lambda r: r["aa_width"])

            if not prim_results:
                continue

            speedups = [r["async_speedup"] for r in prim_results]
            offset = (
                i * len(primitive_types) + j - len(by_device) * len(primitive_types) / 2
            ) * width

            ax_speedup.bar(
                x + offset,
                speedups,
                width,
                label=f"{device_label} - {primitive_type}",
                color=colors[i],
                alpha=0.5 + 0.1 * j,
            )

    ax_speedup.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_speedup.set_ylabel("Async Speedup", fontsize=11, fontweight="bold")
    ax_speedup.set_xlabel("Anti-Aliasing Width", fontsize=11, fontweight="bold")
    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels([f"{w:.2f}" for w in aa_widths])
    ax_speedup.legend(loc="best", framealpha=0.9, fontsize=8, ncol=2)
    ax_speedup.grid(True, alpha=0.3, axis="y")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )
    fig2.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.5,
    )

    plt.tight_layout()

    # Save plots
    output_path1 = get_output_dir() / "graphics_aa_performance_by_primitive.png"
    output_path2 = get_output_dir() / "graphics_aa_performance_speedup.png"

    fig.savefig(output_path1, dpi=150, bbox_inches="tight")
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")

    print("\nPlots saved to:")
    print(f"  {output_path1}")
    print(f"  {output_path2}")


# ============================================================================
# Main Entry Point
# ============================================================================

AVAILABLE_TESTS = {
    "anisodiff": ("Anisotropic diffusion (Perona-Malik filtering)", test_anisodiff),
    "gaussian": ("Separable Gaussian convolution", test_gaussian),
    "vgg16": ("VGG16 CNN inference", test_vgg16),
    "resnet50": ("ResNet50 CNN inference", test_resnet50),
    "graphics4": ("KernelDraw4 display list rendering", test_graphics4),
    "graphics-aa": ("Graphics anti-aliasing performance comparison", test_graphics_aa),
}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-backend performance testing for yapocis library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "module", nargs="?", choices=list(AVAILABLE_TESTS.keys()), help="Module to test"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available test modules"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all available performance tests"
    )
    parser.add_argument(
        "--backends", type=str, help="Comma-separated list of backends (e.g., ocl,cuda)"
    )
    parser.add_argument(
        "--devices", type=str, help="Comma-separated list of device filters"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per test (default: 5)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Image size for tests (default: 512)"
    )

    args = parser.parse_args()

    if args.list:
        print("Available test modules:")
        print()
        for name, (desc, _) in AVAILABLE_TESTS.items():
            print(f"  {name:<15} - {desc}")
        print()
        print(f"Output directory: {get_output_dir()}")
        return 0

    if args.all:
        print("Running all performance tests")
        print(f"Output directory: {get_output_dir()}")
        print()

        failed_tests = []
        for module_name, (desc, test_func) in AVAILABLE_TESTS.items():
            print("=" * 70)
            print(f"Running: {module_name} - {desc}")
            print("=" * 70)
            try:
                result = test_func(args)
                if result != 0:
                    failed_tests.append(module_name)
                    print(f"\n⚠️  {module_name} test failed with code {result}")
            except Exception as e:
                failed_tests.append(module_name)
                print(f"\n❌ {module_name} test failed with exception: {e}")
                import traceback

                traceback.print_exc()
            print()

        # Summary
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Total tests: {len(AVAILABLE_TESTS)}")
        print(f"Passed: {len(AVAILABLE_TESTS) - len(failed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        if failed_tests:
            print(f"Failed tests: {', '.join(failed_tests)}")
        print("=" * 70)

        return 1 if failed_tests else 0

    if not args.module:
        parser.print_help()
        return 1

    print(f"Output directory: {get_output_dir()}")
    print()

    _, test_func = AVAILABLE_TESTS[args.module]
    return test_func(args)


if __name__ == "__main__":
    sys.exit(main())
