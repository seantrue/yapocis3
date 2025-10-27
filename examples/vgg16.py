#!/usr/bin/env python3
"""
VGG16 classification and embedding extraction example.

Usage:
    # Classify a single image
    python examples/vgg16.py path/to/image.jpg --weights vgg16_weights.npz

    # Test on all image assets
    python examples/vgg16.py --test --weights vgg16_weights.npz

    # Extract embeddings instead of classifications
    python examples/vgg16.py path/to/image.jpg --weights vgg16_weights.npz --embeddings

    # Use specific backend
    YAPOCIS_BACKEND=metal python examples/vgg16.py --test --weights vgg16_weights.npz
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yapocis.lib.vgg16 import VGG16, VGG16Resident, load_imagenet_labels
from yapocis.images import images as get_image_list, find_image
from yapocis.models import get_weights_path, get_classes_path, models as list_models


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (H, W, 3) in range [0, 255]
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    except ImportError:
        print("ERROR: PIL (Pillow) required for image loading")
        print("Install with: pip install Pillow")
        sys.exit(1)


def classify_image(
    vgg: VGG16,
    image_path: str,
    labels: list[str],
    show_embeddings: bool = False
) -> None:
    """
    Classify a single image and print results.

    Args:
        vgg: VGG16 model instance
        image_path: Path to image
        labels: ImageNet class labels
        show_embeddings: If True, show FC6 embeddings instead of classification
    """
    print(f"\nProcessing: {image_path}")
    print("-" * 60)

    # Load image
    try:
        image = load_image(image_path)
        print(f"Image size: {image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Time preprocessing
    start = time.time()
    preprocessed = vgg.preprocess_image(image)
    preprocess_time = time.time() - start
    print(f"Preprocessing time: {preprocess_time*1000:.2f} ms")

    if show_embeddings:
        # Extract embeddings
        start = time.time()
        features = vgg.extract_features(image, layer='fc6')
        inference_time = time.time() - start

        print(f"Inference time: {inference_time*1000:.2f} ms")
        print(f"\nFC6 Embeddings: shape={features.shape}")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std:  {features.std():.4f}")
        print(f"  Min:  {features.min():.4f}")
        print(f"  Max:  {features.max():.4f}")
        print(f"  L2 norm: {np.linalg.norm(features):.4f}")

        # Show top 10 embedding dimensions by magnitude
        top_indices = np.argsort(np.abs(features))[::-1][:10]
        print(f"\nTop 10 embedding dimensions:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Dim {idx:4d}: {features[idx]:+.4f}")
    else:
        # Classification
        start = time.time()
        top_indices, top_probs = vgg.predict(image, top_k=5)
        inference_time = time.time() - start

        print(f"Inference time: {inference_time*1000:.2f} ms")
        print(f"Total time: {(preprocess_time + inference_time)*1000:.2f} ms")

        print(f"\nTop 5 predictions:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
            class_name = labels[idx] if idx < len(labels) else f"class_{idx}"
            print(f"  {i}. {class_name:30s}  {prob*100:6.2f}%")


def test_on_assets(
    vgg: VGG16,
    labels: list[str],
    show_embeddings: bool = False,
    max_images: int = None
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

    print(f"\nFound {len(image_names)} image assets")
    print("=" * 60)

    if max_images:
        image_names = image_names[:max_images]

    total_time = 0.0
    successful = 0

    for i, name in enumerate(image_names, 1):
        try:
            # Remove extension for display
            display_name = name.replace('.jpg', '').replace('.png', '')
            print(f"\n[{i}/{len(image_names)}] {display_name}")

            # Get full path
            path = find_image(display_name)

            start = time.time()
            classify_image(vgg, path, labels, show_embeddings)
            elapsed = time.time() - start
            total_time += elapsed
            successful += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Processed {successful}/{len(image_names)} images successfully")
    if successful > 0:
        avg_time = total_time / successful
        print(f"Average time per image: {avg_time*1000:.2f} ms")
        print(f"Throughput: {1.0/avg_time:.2f} images/sec")


def resolve_model_paths(weights_arg: str = None, labels_arg: str = None, model_name: str = 'vgg16'):
    """
    Resolve weights and labels paths from arguments or model asset manager.

    Args:
        weights_arg: Weights path from command line (or None to use model manager)
        labels_arg: Labels path from command line (or None to use model manager)
        model_name: Model name to load from asset manager

    Returns:
        Tuple of (weights_path, labels_path)
    """
    weights_path = weights_arg
    labels_path = labels_arg

    # Try to get from model asset manager if not provided
    if weights_path is None:
        model_weights = get_weights_path(model_name)
        if model_weights:
            weights_path = str(model_weights)
            print(f"Using weights from model asset manager: {weights_path}")

    if labels_path is None:
        model_classes = get_classes_path(model_name)
        if model_classes:
            labels_path = str(model_classes)
            print(f"Using labels from model asset manager: {labels_path}")

    return weights_path, labels_path


def check_weights_file(weights_path: str, show_help: bool = True) -> bool:
    """
    Check if weights file exists and provide helpful error message.

    Args:
        weights_path: Path to weights file
        show_help: Show detailed help message

    Returns:
        True if file exists, False otherwise
    """
    if weights_path and os.path.exists(weights_path):
        return True

    if not show_help:
        return False

    print(f"ERROR: Weights file not found: {weights_path if weights_path else 'not specified'}")
    print()
    print("VGG16 weights are required to run this example.")
    print()
    print("Available models:")
    available_models = list_models()
    if available_models:
        for model in available_models:
            weights = get_weights_path(model)
            if weights:
                print(f"  - {model}: {weights}")
    else:
        print("  (no models found in yapocis/models/)")
    print()
    print("To use model asset manager:")
    print("  python examples/vgg16.py image.jpg")
    print("  (automatically uses yapocis/models/vgg16/weights.npz)")
    print()
    print("Or specify weights manually:")
    print("  python examples/vgg16.py image.jpg --weights path/to/weights.npz")
    print()
    print("You can also download pre-trained VGG16 weights from:")
    print("  - Keras: https://github.com/fchollet/deep-learning-models/releases")
    print("  - PyTorch: https://download.pytorch.org/models/vgg16-397923af.pth")
    print()
    return False


def main():
    parser = argparse.ArgumentParser(
        description='VGG16 classification and embedding extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    parser.add_argument(
        'image',
        nargs='?',
        help='Path to input image'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test on all image assets instead of single image'
    )

    # Model options
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to VGG16 weights file (.npz format). If not specified, uses model asset manager.'
    )
    parser.add_argument(
        '--labels',
        type=str,
        help='Path to ImageNet labels file. If not specified, uses model asset manager.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vgg16',
        help='Model name to load from asset manager (default: vgg16)'
    )

    # Output options
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Extract FC6 embeddings instead of classification'
    )
    parser.add_argument(
        '--resident',
        action='store_true',
        help='Use GPU-resident buffers for better performance'
    )

    # Processing options
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to process in --test mode'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['ocl', 'metal', 'numbax'],
        help='Backend to use (can also set YAPOCIS_BACKEND env var)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.test and not args.image:
        parser.error("Either provide an image path or use --test flag")

    # Set backend if specified
    if args.backend:
        os.environ['YAPOCIS_BACKEND'] = args.backend

    # Show backend info
    backend = os.environ.get('YAPOCIS_BACKEND', 'auto')
    print(f"Using backend: {backend}")

    # Resolve model paths from asset manager or command line
    weights_path, labels_path = resolve_model_paths(args.weights, args.labels, args.model)

    # Check weights file
    if not check_weights_file(weights_path):
        return 1

    # Load labels
    labels = load_imagenet_labels(labels_path, args.model)
    print(f"Loaded {len(labels)} ImageNet class labels")

    # Create VGG16 model
    print(f"\nInitializing VGG16...")
    try:
        if args.resident:
            vgg = VGG16Resident(weights_path)
            print("Using GPU-resident buffers")
        else:
            vgg = VGG16(weights_path)
    except Exception as e:
        print(f"Error initializing VGG16: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Process images
    try:
        if args.test:
            test_on_assets(vgg, labels, args.embeddings, args.max_images)
        else:
            classify_image(vgg, args.image, labels, args.embeddings)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
