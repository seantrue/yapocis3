#!/usr/bin/env python3
"""
RetinaNet Object Detection Example

Demonstrates how to use RetinaNet for detecting objects in images.

Usage:
    # Detect objects in a single image
    python examples/retinanet.py image.jpg

    # Test mode: process all image assets
    python examples/retinanet.py --test

    # Limit number of images in test mode
    python examples/retinanet.py --test --max-images 5

    # Adjust detection parameters
    python examples/retinanet.py image.jpg --threshold 0.5 --nms 0.4
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

# Add src to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from yapocis.lib.retinanet import RetinaNet, load_coco_labels
from yapocis.lib.graphics4 import Graphics4
from yapocis.images import images as get_image_list, find_image
from yapocis.models import get_weights_path, get_classes_path, models as list_models
from yapocis.utils.basic import show_array


def draw_detection_boxes(image: np.ndarray, detections: np.ndarray, class_names: Optional[List[str]]) -> np.ndarray:
    """Draw bounding boxes and labels on image.

    Args:
        image: Input image (H, W, 3) in range [0, 1] or [0, 255]
        detections: Array of detections (N, 6) with format (class_id, score, x1, y1, x2, y2)
        class_names: Optional list of class names

    Returns:
        Image with drawn bounding boxes (H, W, 4) in range [0, 1]
    """
    if len(detections) == 0:
        # Convert to RGBA if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        if image.shape[-1] == 3:
            h, w = image.shape[:2]
            rgba = np.ones((h, w, 4), dtype=np.float32)
            rgba[:, :, :3] = image
            return rgba
        return image

    # Create graphics context
    g = Graphics4()

    # Convert image to float32 RGBA
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    if image.shape[-1] == 3:
        h, w = image.shape[:2]
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = image
        image = rgba

    # Prepare drawing data
    all_lines = []
    text_locations = []
    text_labels = []
    text_colors = []

    # Draw each detection
    for class_id, score, y1, x1, y2, x2 in detections:
        # Get class name
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        # Choose color based on class (deterministic but distinct)
        color = [
            (class_id * 67) % 255 / 255.0,
            (class_id * 139) % 255 / 255.0,
            (class_id * 211) % 255 / 255.0,
            1.0
        ]

        # Draw bounding box
        lines = [
            [x1, y1, x2, y1],  # Top
            [x2, y1, x2, y2],  # Right
            [x2, y2, x1, y2],  # Bottom
            [x1, y2, x1, y1]   # Left
        ]
        all_lines.extend(lines)

        # Prepare label with score
        label = f"{class_name} {score:.2f}"
        # Position label above box, or below if near top edge
        text_y = y2 - 5 if y2 > 20 else y1 + 15
        text_locations.append((x1, text_y))
        text_labels.append(label)
        text_colors.append(color)

    # Draw all lines at once
    if all_lines:
        g.lines(image, all_lines, color=[1.0, 1.0, 1.0, 1.0], distance=2.0)

    # Draw all text labels
    if text_locations:
        g.draw_strings(image, text_locations, text_labels, colors=text_colors, size=18)

    return g.read(image)


def load_image(image_path: str) -> np.ndarray:
    """Load image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (H, W, 3) in range [0, 1]
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        return np.array(img).astype(np.float32) / 255.0
    except ImportError:
        print("ERROR: PIL (Pillow) required for image loading")
        print("Install with: pip install Pillow")
        sys.exit(1)


def run_torchvision_detection(
        image: np.ndarray,
        score_threshold: float = 0.3,
) -> List:
    """Run detection using torchvision RetinaNet for comparison.

    Args:
        image: Input image (H, W, 3) in range [0, 1]
        score_threshold: Detection confidence threshold

    Returns:
        List of detections in same format as yapocis: [(class_id, score, x1, y1, x2, y2), ...]
    """
    try:
        import torch
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.transforms import functional as F
    except ImportError:
        raise ImportError(
            "Torchvision required for --torchvision mode\n"
            "Install with: pip install torch torchvision"
        )

    # Load model
    model = retinanet_resnet50_fpn(pretrained=True, score_thresh=score_threshold)
    model.eval()

    # Convert to torch tensor (C, H, W)
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Convert to yapocis format: (class_id, score, x1, y1, x2, y2)
    detections = []
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        detections.append((int(label), float(score), float(x1), float(y1), float(x2), float(y2)))

    return detections


def detect_image(
        retinanet,
        image_path: str,
        class_names: Optional[List[str]],
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        save_to_queue: bool = False,
        use_torchvision: bool = False
) -> None:
    """Detect objects in a single image and print results.

    Args:
        retinanet: RetinaNet model instance (or None if use_torchvision=True)
        image_path: Path to image
        class_names: COCO class labels
        score_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold
        max_detections: Maximum detections to return
        save_to_queue: If True, save visualization to queue/raw
        use_torchvision: If True, use torchvision instead of yapocis
    """
    impl_name = "torchvision" if use_torchvision else "yapocis"
    print(f"\nProcessing: {image_path} [{impl_name}]")
    print("-" * 60)

    # Load image
    try:
        image = load_image(image_path)
        print(f"Image size: {image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Run detection
    start = time.time()
    if use_torchvision:
        detections = run_torchvision_detection(image, score_threshold)
    else:
        detections = retinanet.predict(
            image,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections
        )
    elapsed = time.time() - start

    print(f"Detection time: {elapsed*1000:.2f} ms")
    print(f"Found {len(detections)} objects")

    # Display detections
    if len(detections) > 0:
        print(f"\nDetections:")
        print(f"  {'Class':<30} {'Score':<8} {'BBox (x1,y1,x2,y2)'}")
        print(f"  {'-'*30} {'-'*8} {'-'*30}")

        for class_id, score, x1, y1, x2, y2 in detections:
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            print(f"  {class_name:<30} {score:.3f}    "
                  f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

        # Draw and save visualization
        if save_to_queue:
            vis_image = draw_detection_boxes(image, detections, class_names)
            image_name = Path(image_path).stem
            suffix = f" {impl_name}" if use_torchvision else ""
            show_array(vis_image, f"retinanet {image_name}{suffix}")
            print(f"\n✓ Visualization saved to queue/raw")
    else:
        print("\nNo objects detected")
        print(f"Try lowering the threshold (current: {score_threshold})")


def test_on_assets(
        retinanet: Optional[RetinaNet],
        class_names: Optional[List[str]],
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        max_images: Optional[int] = None,
        use_torchvision: bool = False
) -> None:
    """Test RetinaNet on all available image assets.

    Args:
        retinanet: RetinaNet model instance (or None if use_torchvision=True)
        class_names: COCO class labels
        score_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold
        max_detections: Maximum detections to return per image
        max_images: Maximum number of images to process (None for all)
        use_torchvision: If True, use torchvision instead of yapocis
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
    total_detections = 0
    successful = 0

    for i, name in enumerate(image_names, 1):
        try:
            # Remove extension for display
            display_name = name.replace('.jpg', '').replace('.png', '')
            print(f"\n[{i}/{len(image_names)}] {display_name}")

            # Get full path
            path = find_image(display_name)

            # Load and process
            image = load_image(str(path))
            start = time.time()
            if use_torchvision:
                detections = run_torchvision_detection(image, score_threshold)
            else:
                detections = retinanet.predict(
                    image,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    max_detections=max_detections
                )
            elapsed = time.time() - start
            total_time += elapsed

            print(f"  → {len(detections)} objects detected ({elapsed*1000:.1f} ms)")

            # Show top detections
            if len(detections) > 0:
                for class_id, score, x1, y1, x2, y2 in detections[:3]:
                    if class_names and class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    print(f"     {class_name:25s} {score*100:5.1f}%")

                # Draw and save visualization
                vis_image = draw_detection_boxes(image, detections, class_names)
                impl_name = "torchvision" if use_torchvision else ""
                suffix = f" {impl_name}" if impl_name else ""
                show_array(vis_image, f"retinanet {display_name}{suffix}")

            total_detections += len(detections)
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
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/successful:.1f}")
    print(f"\n✓ All visualizations saved to queue/raw")


def resolve_model_paths(
        weights_arg: Optional[str] = None,
        labels_arg: Optional[str] = None,
        model_name: str = 'retinanet'
) -> tuple[str, str]:
    """Resolve paths to weights and labels files.

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
        available_models = list_models()
        error_msg = f"ERROR: Weights file not found: {weights_path or 'not specified'}\n\n"
        error_msg += "RetinaNet weights are required to run this example.\n\n"

        if available_models:
            error_msg += "Available models:\n"
            for model in available_models:
                model_weights = get_weights_path(model)
                if model_weights:
                    error_msg += f"  - {model}: {model_weights}\n"
            error_msg += "\n"

        error_msg += "To use model asset manager:\n"
        error_msg += "  python examples/retinanet.py image.jpg\n"
        error_msg += "  (automatically uses yapocis/models/retinanet/weights.npz)\n\n"
        error_msg += "Or specify weights manually:\n"
        error_msg += "  python examples/retinanet.py image.jpg --weights path/to/weights.npz\n\n"
        error_msg += "Download RetinaNet weights:\n"
        error_msg += "  yapocis-tools models --download retinanet\n"

        raise FileNotFoundError(error_msg)

    if labels_path is None or not Path(labels_path).exists():
        print(f"WARNING: Labels file not found: {labels_path or 'not specified'}")
        print("Will use class IDs instead of names")
        labels_path = None

    return weights_path, labels_path


def main():
    parser = argparse.ArgumentParser(
        description='RetinaNet Object Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image detection
  python examples/retinanet.py dog.jpg

  # Test on all image assets
  python examples/retinanet.py --test

  # Test with max limit
  python examples/retinanet.py --test --max-images 5

  # Adjust thresholds
  python examples/retinanet.py dog.jpg --threshold 0.5 --nms 0.4

  # Use torchvision for comparison
  python examples/retinanet.py dog.jpg --torchvision

  # Use different backend (yapocis only)
  YAPOCIS_BACKEND=ocl python examples/retinanet.py dog.jpg
        """
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
        default=None,
        help='Path to RetinaNet weights file (.npz). If not specified, uses model asset manager.'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default = None,
        help='Path to COCO labels file. If not specified, uses model asset manager.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='retinanet',
        help='Model name to load from asset manager (default: retinanet)'
    )

    # Detection options
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--nms',
        type=float,
        default=0.5,
        help='NMS IoU threshold (default: 0.5)'
    )
    parser.add_argument(
        '--max-detections',
        type=int,
        default=100,
        help='Maximum detections per image (default: 100)'
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
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save visualization to queue/raw (automatic in --test mode)'
    )
    parser.add_argument(
        '--torchvision',
        action='store_true',
        help='Use torchvision RetinaNet instead of yapocis (for comparison)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.test and not args.image:
        parser.error("Either provide an image path or use --test flag")

    # Set backend if specified
    if args.backend:
        os.environ['YAPOCIS_BACKEND'] = args.backend

    # Show implementation
    labels_path = None
    if args.torchvision:
        print("Using implementation: torchvision")
        retinanet = None  # Not needed for torchvision mode

        # Get labels for torchvision
        labels_path = get_classes_path('retinanet')
        if labels_path:
            labels_path = str(labels_path)
    else:
        # Show backend info
        backend = os.environ.get('YAPOCIS_BACKEND', 'auto')
        print(f"Using implementation: yapocis")
        print(f"Backend: {backend}")

        # Resolve model paths from asset manager or command line
        try:
            weights_path, labels_path = resolve_model_paths(
                args.weights, args.labels, args.model
            )
        except FileNotFoundError as e:
            print(str(e))
            return 1

        # Create RetinaNet model
        print(f"\nInitializing RetinaNet...")
        try:
            retinanet = RetinaNet(weights_path)
            print(f"✓ Model loaded ({retinanet.num_classes} classes, {retinanet.num_anchors} anchors/location)")
        except Exception as e:
            print(f"Error initializing RetinaNet: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Load labels (for both implementations)
    class_names = None
    if labels_path:
        try:
            class_names = load_coco_labels(labels_path)
            print(f"Loaded {len(class_names)} COCO class labels")
        except Exception as e:
            print(f"Warning: Could not load labels: {e}")
            print("Will use class IDs instead")

    # Process images
    try:
        if args.test:
            test_on_assets(
                retinanet,
                class_names,
                args.threshold,
                args.nms,
                args.max_detections,
                args.max_images,
                use_torchvision=args.torchvision
            )
        else:
            detect_image(
                retinanet,
                args.image,
                class_names,
                args.threshold,
                args.nms,
                args.max_detections,
                save_to_queue=args.save or args.test,
                use_torchvision=args.torchvision
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
