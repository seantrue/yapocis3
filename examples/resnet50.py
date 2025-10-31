#!/usr/bin/env python3
"""
ResNet50 image classification example.

This example demonstrates using ResNet50 for ImageNet classification with the
unified CNN kernels across different backends (OpenCL, Metal, Numbax).

ResNet50 achieves better accuracy than VGG16 with fewer parameters (25.6M vs 138M).

Usage:
    # Classify single image (auto-loads weights from model asset manager)
    python examples/resnet50.py dog.jpg

    # Classify with top-10 predictions
    python examples/resnet50.py dog.jpg --top-k 10

    # Extract embeddings
    python examples/resnet50.py dog.jpg --embeddings

    # Test mode: process all image assets
    python examples/resnet50.py --test

    # Batch processing with max limit
    python examples/resnet50.py --test --max-images 5

    # Use specific weights/labels
    python examples/resnet50.py dog.jpg --weights path/to/weights.npz --labels path/to/classes.txt

    # Try different backend
    YAPOCIS_BACKEND=ocl python examples/resnet50.py dog.jpg
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from yapocis.images import images as get_image_list, find_image

from yapocis.lib.resnet50 import ResNet50, ResNet50Resident, load_imagenet_labels
from yapocis.models import models, get_weights_path, get_classes_path


def load_image(image_path: str) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img).astype(np.float32) / 255.0


def resolve_model_paths(
        weights_arg: Optional[str] = None,
        labels_arg: Optional[str] = None,
        model_name: str = 'resnet50'
) -> Tuple[str, str]:
    """
    Resolve paths to weights and labels files.

    Args:
        weights_arg: Optional explicit path to weights
        labels_arg: Optional explicit path to labels
        model_name: Model name for asset manager lookup

    Returns:
        (weights_path, labels_path) tuple

    Raises:
        FileNotFoundError: If files not found
    """
    weights_path = weights_arg
    labels_path = labels_arg

    # Try model asset manager if not explicitly provided
    if weights_path is None:
        model_weights = get_weights_path(model_name)
        if model_weights:
            weights_path = str(model_weights)
            print(f"Using weights from model asset manager: {weights_path}")

    if labels_path is None:
        model_labels = get_classes_path(model_name)
        if model_labels:
            labels_path = str(model_labels)
            print(f"Using labels from model asset manager: {labels_path}")

    # Validate paths
    if weights_path is None or not Path(weights_path).exists():
        # Show helpful error with available models
        available_models = models()
        error_msg = f"ERROR: Weights file not found: {weights_path or 'not specified'}\n\n"
        error_msg += "ResNet50 weights are required to run this example.\n\n"

        if available_models:
            error_msg += "Available models:\n"
            for model in available_models:
                model_weights = get_weights_path(model)
                if model_weights:
                    error_msg += f"  - {model}: {model_weights}\n"
            error_msg += "\n"

        error_msg += "To use model asset manager:\n"
        error_msg += "  python examples/resnet50.py image.jpg\n"
        error_msg += "  (automatically uses yapocis/models/resnet50/weights.npz)\n\n"
        error_msg += "Or specify weights manually:\n"
        error_msg += "  python examples/resnet50.py image.jpg --weights path/to/weights.npz\n"

        raise FileNotFoundError(error_msg)

    if labels_path is None or not Path(labels_path).exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path or 'not specified'}\n"
            "Provide --labels argument or install model assets."
        )

    return weights_path, labels_path


def classify_image(
        resnet: ResNet50,
        image_path: str,
        labels: List[str],
        top_k: int = 5,
        show_embeddings: bool = False
) -> None:
    """Classify single image and print results."""
    print(f"\nClassifying: {image_path}")

    # Load and classify
    image = load_image(image_path)
    start_time = time.time()
    top_indices, top_probs = resnet.predict(image, top_k=top_k)
    elapsed = (time.time() - start_time) * 1000

    print(f"Inference time: {elapsed:.2f} ms\n")

    # Print top predictions
    print(f"Top {top_k} predictions:")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        print(f"  {i}. {labels[idx]:25s} {prob * 100:6.2f}%")

    # Show embeddings if requested
    if show_embeddings:
        embeddings = resnet.extract_embeddings(image)
        print(f"\nEmbeddings (2048-dimensional):")
        print(f"  Mean: {np.mean(embeddings):.4f}")
        print(f"  Std:  {np.std(embeddings):.4f}")
        print(f"  L2 norm: {np.linalg.norm(embeddings):.4f}")
        print(f"  Min: {np.min(embeddings):.4f}, Max: {np.max(embeddings):.4f}")


def test_mode(
        resnet: ResNet50,
        labels: List[str],
        max_images: Optional[int] = None
) -> None:
    """
    Test VGG16 on all available image assets.

    Args:
        vgg: VGG16 model instance
        labels: ImageNet class labels
        show_embeddings: If True, show embeddings instead of classification
        max_images: Maximum number of images to process (None for all)
    """
    image_names = get_image_list()

    if not image_names:
        print("No image assets found!")
        print("Image assets should be in src/yapocis/images/")
        return

    if max_images:
        image_names = image_names[:max_images]

    total_time = 0.0
    successful = 0

    for i, img_name in enumerate(image_names, 1):
        try:
            img_path = find_image(img_name)
            print(f"[{i}/{len(image_names)}] {img_name}")

            img = load_image(str(img_path))
            start_time = time.time()
            top_indices, top_probs = resnet.predict(img, top_k=3)
            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed

            # Print top prediction
            top_idx = top_indices[0]
            top_prob = top_probs[0]
            print(f"  → {labels[top_idx]:30s} {top_prob * 100:6.2f}% ({elapsed:.1f} ms)\n")

            successful += 1

        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

    # Summary
    if successful > 0:
        avg_time = total_time / successful
        throughput = 1000.0 / avg_time if avg_time > 0 else 0
        print("=" * 60)
        print(f"Processed {successful}/{len(image_names)} images successfully")
        print(f"Average time per image: {avg_time:.2f} ms")
        print(f"Throughput: {throughput:.2f} images/sec")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ResNet50 image classification example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image classification
  python examples/resnet50.py dog.jpg

  # Top-10 predictions
  python examples/resnet50.py dog.jpg --top-k 10

  # Extract embeddings
  python examples/resnet50.py dog.jpg --embeddings

  # Test on all image assets
  python examples/resnet50.py --test

  # Use different backend
  YAPOCIS_BACKEND=ocl python examples/resnet50.py dog.jpg
        """
    )

    parser.add_argument('image', nargs='?', help='Path to input image')
    parser.add_argument('--weights', help='Path to weights.npz (optional, auto-detected)')
    parser.add_argument('--labels', help='Path to classes.txt (optional, auto-detected)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions (default: 5)')
    parser.add_argument('--embeddings', action='store_true', help='Show feature embeddings')
    parser.add_argument('--resident', action='store_true', help='Use GPU-resident buffers for better performance')
    parser.add_argument('--test', action='store_true', help='Test mode: process all image assets')
    parser.add_argument('--max-images', type=int, help='Max images to process in test mode')
    parser.add_argument('--model', default='resnet50', help='Model name for asset manager (default: resnet50)')

    args = parser.parse_args()

    # Validate arguments
    if not args.test and not args.image:
        parser.error("Either provide an image path or use --test flag")

    # Show backend info
    backend = os.environ.get('YAPOCIS_BACKEND', 'auto')
    print(f"Using backend: {backend}")

    # Resolve model paths
    try:
        weights_path, labels_path = resolve_model_paths(
            args.weights, args.labels, args.model
        )
    except FileNotFoundError as e:
        print(str(e))
        return 1

    # Load model and labels
    if args.resident:
        print("Loading ResNet50Resident (GPU-resident buffers)...")
        resnet = ResNet50Resident(weights_path)
    else:
        print("Loading ResNet50...")
        resnet = ResNet50(weights_path)
    labels = load_imagenet_labels(labels_path)
    print(f"Loaded {len(labels)} ImageNet class labels\n")

    # Run classification or test mode
    try:
        if args.test:
            test_mode(resnet, labels, args.max_images)
        else:
            classify_image(resnet, args.image, labels, args.top_k, args.embeddings)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
