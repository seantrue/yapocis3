#!/usr/bin/env python3
"""
Download and convert ResNet50 pre-trained weights to yapocis format.

This script downloads ResNet50 weights from PyTorch's torchvision and converts
them to the yapocis naming convention and NPZ format.

Usage:
    python scripts/download_resnet50.py

    # Custom output directory
    python scripts/download_resnet50.py --output my_models/resnet50

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


def fuse_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """
    Fuse batch normalization into convolution for inference.

    Args:
        conv_weight: Conv weights (out_channels, in_channels, kH, kW)
        bn_weight: BN gamma (out_channels,)
        bn_bias: BN beta (out_channels,)
        bn_mean: BN running mean (out_channels,)
        bn_var: BN running variance (out_channels,)
        eps: BN epsilon

    Returns:
        (fused_weight, fused_bias)
    """
    # Compute scale factor: gamma / sqrt(variance + epsilon)
    scale = bn_weight / np.sqrt(bn_var + eps)

    # Fuse into weights: weight * scale (broadcast over spatial dimensions)
    # conv_weight shape: (out_channels, in_channels, kH, kW)
    # scale shape: (out_channels,)
    fused_weight = conv_weight * scale.reshape(-1, 1, 1, 1)

    # Fuse into bias: beta - mean * scale
    fused_bias = bn_bias - bn_mean * scale

    return fused_weight, fused_bias


def convert_resnet50_weights(model, output_path: Path) -> None:
    """
    Convert ResNet50 PyTorch weights to yapocis format.

    ResNet50 architecture:
    - 1 stem: conv1 (7×7 conv, stride 2) + bn1
    - 4 stages with bottleneck blocks:
      - Stage 1: 3 blocks (block0-2)
      - Stage 2: 4 blocks (block3-6)
      - Stage 3: 6 blocks (block7-12)
      - Stage 4: 3 blocks (block13-15)
    - 1 FC layer: fc

    Each bottleneck block has:
    - conv1 + bn1: 1×1 conv (reduce channels)
    - conv2 + bn2: 3×3 conv
    - conv3 + bn3: 1×1 conv (expand channels)
    - downsample (conv + bn): 1×1 conv (projection shortcut, optional)

    Batch normalization is fused into convolution weights for inference.

    Args:
        model: PyTorch ResNet50 model with pretrained weights
        output_path: Output directory for weights.npz
    """
    print("Converting ResNet50 weights to yapocis format...")
    print("(Fusing batch normalization into convolution weights)")

    weights_dict = {}
    state_dict = model.state_dict()

    # Stem layer (conv1 + bn1)
    if 'conv1.weight' in state_dict:
        conv_w = state_dict['conv1.weight'].cpu().numpy()
        bn_w = state_dict['bn1.weight'].cpu().numpy()
        bn_b = state_dict['bn1.bias'].cpu().numpy()
        bn_mean = state_dict['bn1.running_mean'].cpu().numpy()
        bn_var = state_dict['bn1.running_var'].cpu().numpy()

        # Fuse batch norm into conv
        fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)

        weights_dict['stem_weights'] = fused_w
        weights_dict['stem_bias'] = fused_b
        print(f"  stem_weights: {fused_w.shape}")
        print(f"  stem_bias: {fused_b.shape}")

    # Bottleneck blocks
    # PyTorch structure: layer1, layer2, layer3, layer4
    # Each layer contains multiple Bottleneck blocks
    block_idx = 0

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        # Count blocks in this layer
        layer_idx = 0
        while True:
            block_prefix = f'{layer_name}.{layer_idx}'

            # Check if this block exists
            conv1_key = f'{block_prefix}.conv1.weight'
            if conv1_key not in state_dict:
                break  # No more blocks in this layer

            # conv1 + bn1 (1×1)
            if conv1_key in state_dict:
                conv_w = state_dict[f'{block_prefix}.conv1.weight'].cpu().numpy()
                bn_w = state_dict[f'{block_prefix}.bn1.weight'].cpu().numpy()
                bn_b = state_dict[f'{block_prefix}.bn1.bias'].cpu().numpy()
                bn_mean = state_dict[f'{block_prefix}.bn1.running_mean'].cpu().numpy()
                bn_var = state_dict[f'{block_prefix}.bn1.running_var'].cpu().numpy()

                fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)
                weights_dict[f'block{block_idx}_conv1_weights'] = fused_w
                weights_dict[f'block{block_idx}_conv1_bias'] = fused_b
                print(f"  block{block_idx}_conv1_weights: {fused_w.shape}")

            # conv2 + bn2 (3×3)
            conv2_key = f'{block_prefix}.conv2.weight'
            if conv2_key in state_dict:
                conv_w = state_dict[f'{block_prefix}.conv2.weight'].cpu().numpy()
                bn_w = state_dict[f'{block_prefix}.bn2.weight'].cpu().numpy()
                bn_b = state_dict[f'{block_prefix}.bn2.bias'].cpu().numpy()
                bn_mean = state_dict[f'{block_prefix}.bn2.running_mean'].cpu().numpy()
                bn_var = state_dict[f'{block_prefix}.bn2.running_var'].cpu().numpy()

                fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)
                weights_dict[f'block{block_idx}_conv2_weights'] = fused_w
                weights_dict[f'block{block_idx}_conv2_bias'] = fused_b
                print(f"  block{block_idx}_conv2_weights: {fused_w.shape}")

            # conv3 + bn3 (1×1)
            conv3_key = f'{block_prefix}.conv3.weight'
            if conv3_key in state_dict:
                conv_w = state_dict[f'{block_prefix}.conv3.weight'].cpu().numpy()
                bn_w = state_dict[f'{block_prefix}.bn3.weight'].cpu().numpy()
                bn_b = state_dict[f'{block_prefix}.bn3.bias'].cpu().numpy()
                bn_mean = state_dict[f'{block_prefix}.bn3.running_mean'].cpu().numpy()
                bn_var = state_dict[f'{block_prefix}.bn3.running_var'].cpu().numpy()

                fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)
                weights_dict[f'block{block_idx}_conv3_weights'] = fused_w
                weights_dict[f'block{block_idx}_conv3_bias'] = fused_b
                print(f"  block{block_idx}_conv3_weights: {fused_w.shape}")

            # Downsample (conv + bn) - optional projection shortcut
            downsample_conv_key = f'{block_prefix}.downsample.0.weight'
            if downsample_conv_key in state_dict:
                conv_w = state_dict[f'{block_prefix}.downsample.0.weight'].cpu().numpy()
                bn_w = state_dict[f'{block_prefix}.downsample.1.weight'].cpu().numpy()
                bn_b = state_dict[f'{block_prefix}.downsample.1.bias'].cpu().numpy()
                bn_mean = state_dict[f'{block_prefix}.downsample.1.running_mean'].cpu().numpy()
                bn_var = state_dict[f'{block_prefix}.downsample.1.running_var'].cpu().numpy()

                fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)
                weights_dict[f'block{block_idx}_downsample_weights'] = fused_w
                weights_dict[f'block{block_idx}_downsample_bias'] = fused_b
                print(f"  block{block_idx}_downsample_weights: {fused_w.shape}")

            block_idx += 1
            layer_idx += 1

    # Fully connected layer
    if 'fc.weight' in state_dict:
        weights = state_dict['fc.weight'].cpu().numpy()
        weights_dict['fc_weights'] = weights
        print(f"  fc_weights: {weights.shape}")

    if 'fc.bias' in state_dict:
        bias = state_dict['fc.bias'].cpu().numpy()
        weights_dict['fc_bias'] = bias
        print(f"  fc_bias: {bias.shape}")

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
    print(f"  Bottleneck blocks: {block_idx}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and convert ResNet50 weights to yapocis format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('src/yapocis/models/resnet50'),
        help='Output directory for weights and labels (default: src/yapocis/models/resnet50)'
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

    # Download ResNet50 model
    print("Downloading ResNet50 pretrained model from PyTorch...")
    print("(This may take a few minutes on first run)")
    model = models.resnet50(pretrained=True)
    model.eval()  # Set to evaluation mode
    print("Download complete!\n")

    # Convert weights
    convert_resnet50_weights(model, args.output)

    # Download labels
    if not args.no_labels:
        print()
        download_imagenet_labels(args.output)

    print("\n" + "=" * 60)
    print("ResNet50 weights successfully downloaded and converted!")
    print("=" * 60)
    print(f"\nFiles created in {args.output}:")
    print(f"  - weights.npz ({(args.output / 'weights.npz').stat().st_size / (1024*1024):.1f} MB)")
    if not args.no_labels and (args.output / 'classes.txt').exists():
        print(f"  - classes.txt")

    print("\nYou can now use ResNet50 with:")
    print(f"  python examples/resnet50.py image.jpg")

    return 0


if __name__ == '__main__':
    sys.exit(main())
