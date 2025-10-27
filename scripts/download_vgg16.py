#!/usr/bin/env python3
"""
Download and convert VGG16 pre-trained weights to yapocis format.

This script downloads VGG16 weights from PyTorch's torchvision and converts
them to the yapocis naming convention and NPZ format.

Usage:
    python scripts/download_vgg16.py

    # Custom output directory
    python scripts/download_vgg16.py --output my_models/vgg16

Requirements:
    pip install torch torchvision
"""
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np

# Check for PyTorch
try:
    import torch
    import torchvision.models as models
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)


def download_imagenet_labels(output_path: Path) -> None:
    """Download ImageNet class labels."""
    import urllib.request
    import ssl

    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_file = output_path / "classes.txt"

    print(f"Downloading ImageNet labels from {labels_url}...")
    try:
        # Create SSL context that doesn't verify certificates
        # (needed on some systems where Python doesn't have access to system certs)
        ssl_context = ssl._create_unverified_context()

        # Use urlopen instead of urlretrieve for SSL context support
        with urllib.request.urlopen(labels_url, context=ssl_context) as response:
            content = response.read()

        # Write to file
        with open(labels_file, 'wb') as f:
            f.write(content)

        print(f"Saved labels to {labels_file}")
    except Exception as e:
        print(f"Warning: Could not download labels: {e}")
        print("You may need to provide classes.txt manually")


def convert_vgg16_weights(model, output_path: Path) -> None:
    """
    Convert VGG16 PyTorch weights to yapocis format.

    Args:
        model: PyTorch VGG16 model with pretrained weights
        output_path: Output directory for weights.npz
    """
    print("Converting VGG16 weights to yapocis format...")

    weights_dict = {}

    # Get model state dict
    state_dict = model.state_dict()

    # VGG16 architecture:
    # - 13 conv layers (conv0-12)
    # - 3 FC layers (fc0-2)

    # Map PyTorch layer names to yapocis names
    # PyTorch uses: features.0.weight, features.0.bias, etc.
    # yapocis uses: conv0_weights, conv0_bias, fc0_weights, fc0_bias

    # Convolutional layers
    conv_layer_idx = 0
    for name, param in state_dict.items():
        if 'features' in name:
            if 'weight' in name:
                # Conv layer weights: (out_channels, in_channels, kH, kW)
                weights = param.cpu().numpy()
                weights_dict[f'conv{conv_layer_idx}_weights'] = weights
                print(f"  conv{conv_layer_idx}_weights: {weights.shape}")
            elif 'bias' in name:
                # Conv layer bias: (out_channels,)
                bias = param.cpu().numpy()
                weights_dict[f'conv{conv_layer_idx}_bias'] = bias
                print(f"  conv{conv_layer_idx}_bias: {bias.shape}")
                conv_layer_idx += 1  # Increment after bias (marks end of layer)

    # Fully connected layers
    fc_layer_idx = 0
    for name, param in state_dict.items():
        if 'classifier' in name:
            if 'weight' in name:
                # FC layer weights: (out_features, in_features)
                weights = param.cpu().numpy()
                weights_dict[f'fc{fc_layer_idx}_weights'] = weights
                print(f"  fc{fc_layer_idx}_weights: {weights.shape}")
            elif 'bias' in name:
                # FC layer bias: (out_features,)
                bias = param.cpu().numpy()
                weights_dict[f'fc{fc_layer_idx}_bias'] = bias
                print(f"  fc{fc_layer_idx}_bias: {bias.shape}")
                fc_layer_idx += 1  # Increment after bias

    # Save to NPZ
    weights_file = output_path / "weights.npz"
    print(f"\nSaving weights to {weights_file}...")
    np.savez(str(weights_file), **weights_dict)

    # Print summary
    total_params = sum(v.size for v in weights_dict.values())
    file_size_mb = weights_file.stat().st_size / (1024 * 1024)
    print(f"\nSummary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Conv layers: {conv_layer_idx}")
    print(f"  FC layers: {fc_layer_idx}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and convert VGG16 weights to yapocis format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('src/yapocis/models/vgg16'),
        help='Output directory for weights and labels (default: src/yapocis/models/vgg16)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Skip downloading ImageNet labels'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}\n")

    # Download VGG16 model
    print("Downloading VGG16 pretrained model from PyTorch...")
    print("(This may take a few minutes on first run)")
    model = models.vgg16(pretrained=True)
    model.eval()  # Set to evaluation mode
    print("Download complete!\n")

    # Convert weights
    convert_vgg16_weights(model, args.output)

    # Download labels
    if not args.no_labels:
        print()
        download_imagenet_labels(args.output)

    print("\n" + "=" * 60)
    print("VGG16 weights successfully downloaded and converted!")
    print("=" * 60)
    print(f"\nFiles created in {args.output}:")
    print(f"  - weights.npz ({(args.output / 'weights.npz').stat().st_size / (1024*1024):.1f} MB)")
    if not args.no_labels and (args.output / 'classes.txt').exists():
        print(f"  - classes.txt")

    print("\nYou can now use VGG16 with:")
    print(f"  python examples/vgg16.py image.jpg")

    return 0


if __name__ == '__main__':
    sys.exit(main())
