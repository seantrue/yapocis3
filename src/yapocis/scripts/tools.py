#!/usr/bin/env python3
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Unified yapocis development tools CLI.

Provides utilities for inspecting assets, backends, devices, and interfaces.
"""
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter


def cmd_assets(args):
    """Asset management - find and list yapocis assets."""
    from yapocis.fonts import font as find_font
    from yapocis.fonts import fonts
    from yapocis.icons import find_icon, icons
    from yapocis.images import find_image, images
    from yapocis.interfaces import find_interface, interfaces
    from yapocis.kernels import find_kernel, kernels

    # Map asset type names to lister functions
    listers = {
        "interfaces": interfaces,
        "kernels": kernels,
        "images": images,
        "icons": icons,
        "fonts": fonts,
    }

    finders = {
        "interface": find_interface,
        "kernel": find_kernel,
        "image": find_image,
        "icon": find_icon,
        "font": find_font,
    }

    if args.list:
        if args.all:
            asset_types = ["interfaces", "kernels", "images", "icons", "fonts"]
        else:
            asset_types = (
                args.assets
                if args.assets
                else ["interfaces", "kernels", "images", "icons", "fonts"]
            )

        for asset_type in asset_types:
            lister = listers.get(asset_type)
            if lister is None:
                print(f"Can't find asset manager for '{asset_type}'")
                sys.exit(1)
            print(f"## {asset_type.upper()}")
            for asset in lister():
                print(asset)
            print()
        return 0

    # Find specific assets
    if not args.assets:
        print("Specify asset to find or use --list")
        sys.exit(1)

    finder_keys = []
    if args.all or (
        not args.images
        and not args.interfaces
        and not args.kernels
        and not args.icons
        and not args.fonts
    ):
        finder_keys = ["interface", "kernel", "image", "icon", "font"]
    else:
        if args.interfaces:
            finder_keys.append("interface")
        if args.kernels:
            finder_keys.append("kernel")
        if args.images:
            finder_keys.append("image")
        if args.icons:
            finder_keys.append("icon")
        if args.fonts:
            finder_keys.append("font")

    for finder_key in finder_keys:
        finder = finders.get(finder_key)
        if finder is None:
            print(f"Can't find asset finder for '{finder_key}'")
            sys.exit(1)
        for asset in args.assets:
            try:
                print(f"{finder_key:10s} {asset:10s} {finder(asset)}")
            except ValueError:
                print(f"{finder_key:10s} {asset:10s} not found")
    return 0


def cmd_devices(args):
    """List OpenCL/Metal/CUDA devices and their configurations."""
    from yapocis.ocl.device import get_available_device_names
    from yapocis.utils.config import print_config

    device_names = get_available_device_names()

    if args.list:
        for device_name in device_names:
            print(device_name)
        return 0

    if args.devices:
        device_names = list(set(args.devices) & set(device_names))

    for device_name in device_names:
        print(device_name)
        print_config(device_name)
    return 0


def cmd_backends(args):
    """List all available backends and their devices."""
    from yapocis.core import get_backend, get_backend_device, get_backend_infos

    try:
        backend_infos = get_backend_infos()

        if args.json:
            import json

            print(json.dumps(backend_infos, indent=2))
            return 0

        # Print formatted output
        print("=" * 80)
        print("Available Backends and Devices")
        print("=" * 80)
        print()

        current_backend = get_backend()
        current_device_index, current_device_name = get_backend_device()

        for backend, info in backend_infos.items():
            status = "✓" if info["available"] else "✗"
            is_current_backend = backend == current_backend
            backend_marker = " ←" if is_current_backend else ""
            print(f"{status} {backend.upper()}{backend_marker}")

            if info["available"] and info["devices"]:
                for i, device in enumerate(info["devices"]):
                    device_marker = ""
                    if is_current_backend:
                        if current_device_index == i or current_device_name == device:
                            device_marker = " ←"
                    print(f"  [{i}] {device}{device_marker}")
            elif info["error"]:
                print(f"  Error: {info['error']}")
            else:
                print("  No devices found")
            print()

        print("Legend:")
        print("  ✓  = Backend available")
        print("  ✗  = Backend not available")
        print("  ← = Currently selected (backend or device)")
        print()

        if args.verbose:
            print("Environment Variables:")
            print("  YAPOCIS_BACKEND - Select backend (ocl, metal, cuda, numba)")
            print("  YAPOCIS_DEVICE  - Select device by name or index")
            print()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_models(args):
    """Download and convert pre-trained model weights."""
    import ssl
    import urllib.request
    from pathlib import Path

    import numpy as np

    # Check for PyTorch
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch torchvision")
        return 1

    def download_imagenet_labels(output_path: Path) -> None:
        """Download ImageNet class labels."""
        labels_url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        labels_file = output_path / "classes.txt"

        print(f"Downloading ImageNet labels from {labels_url}...")
        try:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl._create_unverified_context()

            # Use urlopen instead of urlretrieve for SSL context support
            with urllib.request.urlopen(labels_url, context=ssl_context) as response:
                content = response.read()

            # Write to file
            with open(labels_file, "wb") as f:
                f.write(content)

            print(f"Saved labels to {labels_file}")
        except Exception as e:
            print(f"Warning: Could not download labels: {e}")
            print("You may need to provide classes.txt manually")

    def fuse_bn_into_conv(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
        """Fuse batch normalization into convolution for inference."""
        # Compute scale factor: gamma / sqrt(variance + epsilon)
        scale = bn_weight / np.sqrt(bn_var + eps)

        # Fuse into weights: weight * scale (broadcast over spatial dimensions)
        fused_weight = conv_weight * scale.reshape(-1, 1, 1, 1)

        # Fuse into bias: beta - mean * scale
        fused_bias = bn_bias - bn_mean * scale

        return fused_weight, fused_bias

    def convert_vgg16_weights(model, output_path: Path) -> None:
        """Convert VGG16 PyTorch weights to yapocis format."""
        print("Converting VGG16 weights to yapocis format...")

        weights_dict = {}
        state_dict = model.state_dict()

        # Convolutional layers
        conv_layer_idx = 0
        for name, param in state_dict.items():
            if "features" in name:
                if "weight" in name:
                    weights = param.cpu().numpy()
                    weights_dict[f"conv{conv_layer_idx}_weights"] = weights
                    print(f"  conv{conv_layer_idx}_weights: {weights.shape}")
                elif "bias" in name:
                    bias = param.cpu().numpy()
                    weights_dict[f"conv{conv_layer_idx}_bias"] = bias
                    print(f"  conv{conv_layer_idx}_bias: {bias.shape}")
                    conv_layer_idx += 1

        # Fully connected layers
        fc_layer_idx = 0
        for name, param in state_dict.items():
            if "classifier" in name:
                if "weight" in name:
                    weights = param.cpu().numpy()
                    weights_dict[f"fc{fc_layer_idx}_weights"] = weights
                    print(f"  fc{fc_layer_idx}_weights: {weights.shape}")
                elif "bias" in name:
                    bias = param.cpu().numpy()
                    weights_dict[f"fc{fc_layer_idx}_bias"] = bias
                    print(f"  fc{fc_layer_idx}_bias: {bias.shape}")
                    fc_layer_idx += 1

        # Save to NPZ
        weights_file = output_path / "weights.npz"
        print(f"\nSaving weights to {weights_file}...")
        np.savez(str(weights_file), **weights_dict)

        # Print summary
        total_params = sum(v.size for v in weights_dict.values())
        file_size_mb = weights_file.stat().st_size / (1024 * 1024)
        print("\nSummary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Conv layers: {conv_layer_idx}")
        print(f"  FC layers: {fc_layer_idx}")

    def convert_resnet50_weights(model, output_path: Path) -> None:
        """Convert ResNet50 PyTorch weights to yapocis format."""
        print("Converting ResNet50 weights to yapocis format...")
        print("(Fusing batch normalization into convolution weights)")

        weights_dict = {}
        state_dict = model.state_dict()

        # Stem layer (conv1 + bn1)
        if "conv1.weight" in state_dict:
            conv_w = state_dict["conv1.weight"].cpu().numpy()
            bn_w = state_dict["bn1.weight"].cpu().numpy()
            bn_b = state_dict["bn1.bias"].cpu().numpy()
            bn_mean = state_dict["bn1.running_mean"].cpu().numpy()
            bn_var = state_dict["bn1.running_var"].cpu().numpy()

            # Fuse batch norm into conv
            fused_w, fused_b = fuse_bn_into_conv(conv_w, bn_w, bn_b, bn_mean, bn_var)

            weights_dict["stem_weights"] = fused_w
            weights_dict["stem_bias"] = fused_b
            print(f"  stem_weights: {fused_w.shape}")
            print(f"  stem_bias: {fused_b.shape}")

        # Bottleneck blocks
        block_idx = 0

        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_idx = 0
            while True:
                block_prefix = f"{layer_name}.{layer_idx}"

                # Check if this block exists
                conv1_key = f"{block_prefix}.conv1.weight"
                if conv1_key not in state_dict:
                    break

                # conv1 + bn1 (1×1)
                if conv1_key in state_dict:
                    conv_w = state_dict[f"{block_prefix}.conv1.weight"].cpu().numpy()
                    bn_w = state_dict[f"{block_prefix}.bn1.weight"].cpu().numpy()
                    bn_b = state_dict[f"{block_prefix}.bn1.bias"].cpu().numpy()
                    bn_mean = (
                        state_dict[f"{block_prefix}.bn1.running_mean"].cpu().numpy()
                    )
                    bn_var = state_dict[f"{block_prefix}.bn1.running_var"].cpu().numpy()

                    fused_w, fused_b = fuse_bn_into_conv(
                        conv_w, bn_w, bn_b, bn_mean, bn_var
                    )
                    weights_dict[f"block{block_idx}_conv1_weights"] = fused_w
                    weights_dict[f"block{block_idx}_conv1_bias"] = fused_b
                    print(f"  block{block_idx}_conv1_weights: {fused_w.shape}")

                # conv2 + bn2 (3×3)
                conv2_key = f"{block_prefix}.conv2.weight"
                if conv2_key in state_dict:
                    conv_w = state_dict[f"{block_prefix}.conv2.weight"].cpu().numpy()
                    bn_w = state_dict[f"{block_prefix}.bn2.weight"].cpu().numpy()
                    bn_b = state_dict[f"{block_prefix}.bn2.bias"].cpu().numpy()
                    bn_mean = (
                        state_dict[f"{block_prefix}.bn2.running_mean"].cpu().numpy()
                    )
                    bn_var = state_dict[f"{block_prefix}.bn2.running_var"].cpu().numpy()

                    fused_w, fused_b = fuse_bn_into_conv(
                        conv_w, bn_w, bn_b, bn_mean, bn_var
                    )
                    weights_dict[f"block{block_idx}_conv2_weights"] = fused_w
                    weights_dict[f"block{block_idx}_conv2_bias"] = fused_b
                    print(f"  block{block_idx}_conv2_weights: {fused_w.shape}")

                # conv3 + bn3 (1×1)
                conv3_key = f"{block_prefix}.conv3.weight"
                if conv3_key in state_dict:
                    conv_w = state_dict[f"{block_prefix}.conv3.weight"].cpu().numpy()
                    bn_w = state_dict[f"{block_prefix}.bn3.weight"].cpu().numpy()
                    bn_b = state_dict[f"{block_prefix}.bn3.bias"].cpu().numpy()
                    bn_mean = (
                        state_dict[f"{block_prefix}.bn3.running_mean"].cpu().numpy()
                    )
                    bn_var = state_dict[f"{block_prefix}.bn3.running_var"].cpu().numpy()

                    fused_w, fused_b = fuse_bn_into_conv(
                        conv_w, bn_w, bn_b, bn_mean, bn_var
                    )
                    weights_dict[f"block{block_idx}_conv3_weights"] = fused_w
                    weights_dict[f"block{block_idx}_conv3_bias"] = fused_b
                    print(f"  block{block_idx}_conv3_weights: {fused_w.shape}")

                # Downsample (conv + bn) - optional projection shortcut
                downsample_conv_key = f"{block_prefix}.downsample.0.weight"
                if downsample_conv_key in state_dict:
                    conv_w = (
                        state_dict[f"{block_prefix}.downsample.0.weight"].cpu().numpy()
                    )
                    bn_w = (
                        state_dict[f"{block_prefix}.downsample.1.weight"].cpu().numpy()
                    )
                    bn_b = state_dict[f"{block_prefix}.downsample.1.bias"].cpu().numpy()
                    bn_mean = (
                        state_dict[f"{block_prefix}.downsample.1.running_mean"]
                        .cpu()
                        .numpy()
                    )
                    bn_var = (
                        state_dict[f"{block_prefix}.downsample.1.running_var"]
                        .cpu()
                        .numpy()
                    )

                    fused_w, fused_b = fuse_bn_into_conv(
                        conv_w, bn_w, bn_b, bn_mean, bn_var
                    )
                    weights_dict[f"block{block_idx}_downsample_weights"] = fused_w
                    weights_dict[f"block{block_idx}_downsample_bias"] = fused_b
                    print(f"  block{block_idx}_downsample_weights: {fused_w.shape}")

                block_idx += 1
                layer_idx += 1

        # Fully connected layer
        if "fc.weight" in state_dict:
            weights = state_dict["fc.weight"].cpu().numpy()
            weights_dict["fc_weights"] = weights
            print(f"  fc_weights: {weights.shape}")

        if "fc.bias" in state_dict:
            bias = state_dict["fc.bias"].cpu().numpy()
            weights_dict["fc_bias"] = bias
            print(f"  fc_bias: {bias.shape}")

        # Save to NPZ
        weights_file = output_path / "weights.npz"
        print(f"\nSaving weights to {weights_file}...")
        np.savez(str(weights_file), **weights_dict)

        # Print summary
        total_params = sum(v.size for v in weights_dict.values())
        file_size_mb = weights_file.stat().st_size / (1024 * 1024)
        print("\nSummary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Bottleneck blocks: {block_idx}")

    # Handle model download
    model_name = args.model.lower()

    if model_name not in ["vgg16", "resnet50"]:
        print(f"ERROR: Unknown model '{model_name}'")
        print("Supported models: vgg16, resnet50")
        return 1

    # Determine output directory
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("src/yapocis/models") / model_name

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}\n")

    # Download model
    print(f"Downloading {model_name.upper()} pretrained model from PyTorch...")
    print("(This may take a few minutes on first run)")

    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)

    model.eval()  # Set to evaluation mode
    print("Download complete!\n")

    # Convert weights
    if model_name == "vgg16":
        convert_vgg16_weights(model, output_path)
    elif model_name == "resnet50":
        convert_resnet50_weights(model, output_path)

    # Download labels
    if not args.no_labels:
        print()
        download_imagenet_labels(output_path)

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} weights successfully downloaded and converted!")
    print("=" * 60)
    print(f"\nFiles created in {output_path}:")
    print(
        f"  - weights.npz ({(output_path / 'weights.npz').stat().st_size / (1024*1024):.1f} MB)"
    )
    if not args.no_labels and (output_path / "classes.txt").exists():
        print("  - classes.txt")

    return 0


def cmd_scaffold(args):
    """Create scaffolding for a yapocis extension project."""
    from pathlib import Path
    from textwrap import dedent

    project_name = args.name
    project_dir = Path(args.output) if args.output else Path(project_name)

    # Validate project name (must be valid Python identifier)
    if not project_name.isidentifier():
        print(f"ERROR: '{project_name}' is not a valid project name")
        print("Project name must be a valid Python identifier (letters, numbers, underscores)")
        return 1

    # Define the structure
    structure = {
        "directories": [
            "",  # root
            "interfaces",
            "kernels",
            "kernels/ocl",
            "kernels/metal",
            "kernels/cuda",
            "kernels/numbax",
            "lib",
            "tests",
            "images",
            "icons",
        ],
        "files": {}
    }

    # README.md template
    structure["files"]["README.md"] = dedent(f'''\
        # {project_name}

        A yapocis extension project.

        ## Installation

        ```bash
        # Create virtual environment with uv
        uv venv
        source .venv/bin/activate

        # Install dependencies
        uv pip install -e .

        # Or install with dev dependencies
        uv pip install -e ".[dev]"
        ```

        ## Usage

        Set the environment variable to use this extension:

        ```bash
        export YAPOCIS_PROJECT=$(pwd)
        ```

        Or in Python:

        ```python
        import os
        os.environ['YAPOCIS_PROJECT'] = '/path/to/{project_name}'

        from {project_name}.lib.{project_name} import {project_name.title().replace("_", "")}

        proc = {project_name.title().replace("_", "")}()
        result = proc.fill(image, 0.5)
        ```

        ## Testing

        ```bash
        # Run tests
        pytest tests/

        # Run with specific backend
        YAPOCIS_BACKEND=metal pytest tests/
        ```

        ## Project Structure

        ```
        {project_name}/
        ├── interfaces/
        │   └── {project_name}.interface    # Kernel interface definition
        ├── kernels/
        │   ├── ocl/
        │   │   └── {project_name}.mako     # OpenCL kernel
        │   ├── metal/
        │   │   └── {project_name}.mako     # Metal kernel
        │   ├── cuda/
        │   │   └── {project_name}.mako     # CUDA kernel
        │   └── numbax/
        │       └── {project_name}.py       # Numba kernel
        ├── lib/
        │   └── {project_name}.py           # Python wrapper
        ├── tests/
        │   └── test_{project_name}.py      # Tests
        ├── images/                         # Test images
        ├── icons/                          # Icons
        ├── pyproject.toml
        └── README.md
        ```

        ## Adding New Kernels

        1. Define the interface in `interfaces/{project_name}.interface`
        2. Implement kernels for each backend in `kernels/`
        3. Create Python wrapper in `lib/`
        4. Add tests in `tests/`

        ## License

        MIT
    ''')

    # pyproject.toml template
    structure["files"]["pyproject.toml"] = dedent(f'''\
        [project]
        name = "{project_name}"
        version = "0.1.0"
        description = "A yapocis extension project"
        readme = "README.md"
        requires-python = ">=3.10"
        dependencies = [
            "yapocis>=0.3.0",
            "numpy>=1.20",
        ]

        [project.optional-dependencies]
        dev = [
            "pytest>=7.0",
            "pytest-cov",
        ]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["lib", "interfaces", "kernels"]

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        python_files = ["test_*.py"]

        [tool.ruff]
        line-length = 100
        target-version = "py310"

        [tool.ruff.lint]
        select = ["E", "F", "I", "W"]
    ''')

    # .gitignore
    structure["files"][".gitignore"] = dedent('''\
        # Python
        __pycache__/
        *.py[cod]
        *$py.class
        *.so
        .Python
        build/
        develop-eggs/
        dist/
        downloads/
        eggs/
        .eggs/
        lib64/
        parts/
        sdist/
        var/
        wheels/
        *.egg-info/
        .installed.cfg
        *.egg

        # Virtual environments
        .venv/
        venv/
        ENV/

        # IDE
        .idea/
        .vscode/
        *.swp
        *.swo

        # Testing
        .pytest_cache/
        .coverage
        htmlcov/

        # OS
        .DS_Store
        Thumbs.db

        # Yapocis
        mako_modules/
        *.duckdb
        *.duckdb.wal
    ''')

    # Interface template
    class_name = project_name.title().replace("_", "")
    structure["files"][f"interfaces/{project_name}.interface"] = dedent(f'''\
        // {project_name} interface
        // A sample yapocis kernel interface

        interface {project_name} {{
            // Fill an array with a constant value
            kernel fill(
                sizeof a,
                in float a,
                float value
            );

            // Fill using resident buffer (in-place)
            extension res as fill(
                sizeof a,
                resident float a,
                float value
            );
        }};
    ''')

    # OpenCL kernel template
    structure["files"][f"kernels/ocl/{project_name}.mako"] = dedent(f'''\
        // {project_name} OpenCL kernel

        __kernel void fill(
            int size,
            __global float *a,
            float value
        ) {{
            int idx = get_global_id(0);
            if (idx >= size) return;
            a[idx] = value;
        }}
    ''')

    # Metal kernel template
    structure["files"][f"kernels/metal/{project_name}.mako"] = dedent(f'''\
        // {project_name} Metal kernel
        <%include file="constants.mako"/>

        kernel void fill(
            constant int &size [[buffer(0)]],
            device float *a [[buffer(1)]],
            constant float &value [[buffer(2)]],
            uint idx [[thread_position_in_grid]]
        ) {{
            if (idx >= (uint)size) return;
            a[idx] = value;
        }}
    ''')

    # CUDA kernel template
    structure["files"][f"kernels/cuda/{project_name}.mako"] = dedent(f'''\
        // {project_name} CUDA kernel

        extern "C" __global__ void fill(
            int size,
            float *a,
            float value
        ) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;
            a[idx] = value;
        }}
    ''')

    # Numba kernel template
    structure["files"][f"kernels/numbax/{project_name}.py"] = dedent(f'''\
        """
        {project_name} Numba kernel implementation.
        """
        import numpy as np
        from numba import njit, prange


        @njit(parallel=True, cache=True)
        def fill(size: int, a: np.ndarray, value: float) -> np.ndarray:
            """Fill array with a constant value."""
            flat = a.ravel()
            for idx in prange(size):
                flat[idx] = value
            return a
    ''')

    # Python library wrapper
    structure["files"][f"lib/{project_name}.py"] = dedent(f'''\
        """
        {project_name} - Python wrapper for {project_name} kernels.
        """
        import numpy as np
        from yapocis.lib.bound_kernels import BoundKernels


        class {class_name}(BoundKernels):
            """
            {project_name.title()} kernel wrapper.

            Provides GPU-accelerated array operations.
            """

            INTERFACE = "{project_name}"

            def fill(self, array: np.ndarray, value: float = 0.0) -> None:
                """
                Fill an array with a constant value on the device.

                The array is initialized to the value on the GPU. Use kmgr.read(array)
                to retrieve the result if needed.

                Args:
                    array: Array to fill (will be uploaded to device)
                    value: Value to fill the array with
                """
                array = np.ascontiguousarray(array, dtype=np.float32)
                self.kmgr.fill(array, value)

            def fill_res(self, array: np.ndarray, value: float = 0.0) -> None:
                """
                Fill a resident array with a constant value (in-place on device).

                Uses a buffer that is already on the device.

                Args:
                    array: Resident array to fill
                    value: Value to fill the array with
                """
                self.kmgr.fill_res(array, value)
    ''')

    # lib/__init__.py
    structure["files"]["lib/__init__.py"] = dedent(f'''\
        """
        {project_name} library.
        """
        from .{project_name} import {class_name}

        __all__ = ["{class_name}"]
    ''')

    # Test file
    structure["files"][f"tests/test_{project_name}.py"] = dedent(f'''\
        """
        Tests for {project_name} extension.
        """
        import os
        import sys
        from pathlib import Path

        import numpy as np
        import pytest

        # Set up YAPOCIS_PROJECT to find our extension
        PROJECT_ROOT = Path(__file__).parent.parent
        os.environ["YAPOCIS_PROJECT"] = str(PROJECT_ROOT)

        # Add lib to path for imports
        sys.path.insert(0, str(PROJECT_ROOT))

        from lib.{project_name} import {class_name}


        @pytest.fixture
        def processor():
            """Create a {class_name} instance."""
            return {class_name}()


        class Test{class_name}:
            """Tests for {class_name} class."""

            def test_fill_initializes_on_device(self, processor):
                """Test that fill initializes array on device."""
                array = np.empty((100, 100), dtype=np.float32)
                processor.fill(array, value=1.0)

                # Read back from device to verify
                result = processor.kmgr.read(array)
                assert result.shape == (100, 100)
                assert result.dtype == np.float32
                assert np.allclose(result, 1.0)

            def test_fill_with_value(self, processor):
                """Test that fill sets correct value."""
                array = np.empty((50, 50), dtype=np.float32)
                value = 3.14

                processor.fill(array, value=value)
                result = processor.kmgr.read(array)

                assert np.allclose(result, value)

            def test_fill_with_zero(self, processor):
                """Test filling with zero."""
                array = np.empty((64, 64), dtype=np.float32)

                processor.fill(array, value=0.0)
                result = processor.kmgr.read(array)

                assert np.allclose(result, 0.0)

            def test_fill_res_on_resident_buffer(self, processor):
                """Test fill_res on a resident buffer."""
                array = np.ones((32, 32), dtype=np.float32) * 5.0
                value = 2.0

                # First upload to device
                processor.kmgr.write(array)
                # Then fill in place
                processor.fill_res(array, value=value)
                # Read back
                result = processor.kmgr.read(array)

                assert np.allclose(result, value)

            def test_different_shapes(self, processor):
                """Test various array shapes."""
                shapes = [(10, 10), (100, 50), (256, 256), (512, 128)]

                for shape in shapes:
                    array = np.empty(shape, dtype=np.float32)
                    processor.fill(array, value=1.0)
                    result = processor.kmgr.read(array)
                    assert result.shape == shape
                    assert np.allclose(result, 1.0)


        def test_import():
            """Test that the module can be imported."""
            from lib.{project_name} import {class_name}
            assert {class_name} is not None


        if __name__ == "__main__":
            pytest.main([__file__, "-v"])
    ''')

    # tests/__init__.py
    structure["files"]["tests/__init__.py"] = '"""Test suite for {project_name}."""\n'.format(project_name=project_name)

    # Dry run mode
    if args.dry_run:
        print(f"[DRY RUN] Would create extension project: {project_name}")
        print(f"[DRY RUN] Project directory: {project_dir.absolute()}")
        print()
        print("Directories:")
        for dir_path in structure["directories"]:
            full_path = project_dir / dir_path if dir_path else project_dir
            exists = full_path.exists()
            status = "[EXISTS]" if exists else "[CREATE]"
            print(f"  {status} {full_path}")
        print()
        print("Files:")
        for file_path in structure["files"]:
            full_path = project_dir / file_path
            exists = full_path.exists()
            status = "[EXISTS]" if exists else "[CREATE]"
            if exists and args.force:
                status = "[OVERWRITE]"
            elif exists and args.complete:
                status = "[SKIP]"
            print(f"  {status} {full_path}")
        return 0

    # Check if directory exists
    if project_dir.exists():
        if not args.force and not args.complete:
            print(f"ERROR: Directory '{project_dir}' already exists")
            print("Use --force to overwrite or --complete to add missing files")
            return 1

    # Create directories
    created_dirs = 0
    for dir_path in structure["directories"]:
        full_path = project_dir / dir_path if dir_path else project_dir
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {full_path}")
            created_dirs += 1
        elif args.complete:
            pass  # Directory exists, that's fine in complete mode
        else:
            print(f"Directory exists: {full_path}")

    # Create files
    created_files = 0
    skipped_files = 0
    overwritten_files = 0

    for file_path, content in structure["files"].items():
        full_path = project_dir / file_path

        if full_path.exists():
            if args.force:
                full_path.write_text(content)
                print(f"Overwrote file: {full_path}")
                overwritten_files += 1
            elif args.complete:
                print(f"Skipped existing file: {full_path}")
                skipped_files += 1
            else:
                print(f"File exists: {full_path}")
                skipped_files += 1
        else:
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            print(f"Created file: {full_path}")
            created_files += 1

    # Summary
    print()
    print("=" * 60)
    print(f"Extension project '{project_name}' scaffolding complete!")
    print("=" * 60)
    print(f"  Location: {project_dir.absolute()}")
    print(f"  Directories created: {created_dirs}")
    print(f"  Files created: {created_files}")
    if overwritten_files:
        print(f"  Files overwritten: {overwritten_files}")
    if skipped_files:
        print(f"  Files skipped: {skipped_files}")
    print()
    print("Next steps:")
    print(f"  cd {project_dir}")
    print("  uv venv && source .venv/bin/activate")
    print("  uv pip install -e '.[dev]'")
    print(f"  export YAPOCIS_PROJECT=$(pwd)")
    print("  pytest tests/")
    print()

    return 0


def cmd_interfaces(args):
    """Generate documentation and stubs from interface files."""
    from mako.template import Template

    from yapocis.core.interface import (
        get_interface,
        load_interface,
        scan_kernel_parameters,
    )
    from yapocis.core.interface_context import load_interface_context
    from yapocis.interfaces import interfaces

    def render_interface_template(interface_name: str) -> str:
        """Load and render interface template with YAML context if needed."""
        template_text = load_interface(f"{interface_name}.interface")

        # Check if template contains Mako constructs
        if "%" not in template_text and "${" not in template_text:
            return template_text

        # Load YAML context
        context = load_interface_context(interface_name)

        if not context:
            # No context available, return unrendered
            return template_text

        # Apply context generator if specified in YAML
        from yapocis.core.interface_context import apply_context_generator

        context = apply_context_generator(context)

        # Render template
        template = Template(template_text)
        return template.render(**context)

    def explain_interface(interface, file=None):
        """Print human-readable interface documentation."""
        print("Interface:", interface.name, file=file)
        for kernel in interface.kernels:
            tiling = "tiled" if kernel.tiled else "untiled"
            if kernel.alias:
                print(
                    f"Kernel: {kernel.name} {tiling} alias for '{kernel.alias}'",
                    file=file,
                )
            else:
                print(f"Kernel: {kernel.name} {tiling}", file=file)

            iparam = 0
            ireturn = 0

            def motion(param):
                to_gpu = param.hint in ("in", "inout", "in_out")
                from_gpu = param.hint in ("out", "inout", "out_like", "outlike")
                on_gpu = param.hint in ("resident", "working")
                if to_gpu:
                    print("  Before call, copy buffer from array to GPU", file=file)
                if from_gpu:
                    print("  After call, copy buffer from GPU into array", file=file)
                if on_gpu:
                    print("  Buffer already on GPU", file=file)
                if not from_gpu:
                    pointer = "*" if param.hint else ""
                    print(
                        f"  Pass as parameter of type {pointer}{param.np_dtype}",
                        file=file,
                    )

            def returned(param, ireturn):
                print(
                    f"  Return as {param.np_dtype} (tuple position {ireturn})",
                    file=file,
                )
                return ireturn + 1

            def format_param(param):
                return f"{param.data_type} {param.pointer + ' ' if param.pointer else ''}{param.name}"

            for param in kernel.parameters:
                visible_param = not param.derived
                if visible_param:
                    print(f"  Param {iparam}: {format_param(param)}", file=file)
                    iparam += 1
                else:
                    print(f"  Param derived: {format_param(param)}", file=file)

                if param.hint:
                    if param.hint == "resident":
                        print("  Buffer is resident on GPU", file=file)
                        motion(param)
                    elif param.hint in ("out_like", "outlike"):
                        param_name = param.name.removeprefix("out_like_")
                        param_name = param_name.removeprefix("outlike_")
                        print(
                            f"  Allocate an array like '{param_name}' (opencl arg position={iparam}, dtype={param.np_dtype})",
                            file=file,
                        )
                        motion(param)
                        ireturn = returned(param, ireturn)
                    elif param.hint == "out":
                        print(f"  Allocate an array for {param.name}", file=file)
                        motion(param)
                        ireturn = returned(param, ireturn)
                    else:
                        print(f"  Coerce '{param.name}' to {param.np_dtype}", file=file)
                        motion(param)

                if param.dim:
                    print(f"  Pass in as {param.dim} `{param.name}`", file=file)
                    motion(param)
                print(file=file)
            print(file=file)

    def opencl_stubs(interface, file=None):
        """Generate OpenCL kernel stubs."""
        print("// kernels", interface.name, file=file)
        for kernel in interface.kernels:
            if kernel.alias:
                continue
            parameter_definitions = scan_kernel_parameters(kernel)
            kernel_parameters = []
            for parameter in kernel.parameters:
                if parameter.hint == "out_like":
                    kernel_parameters.append(
                        f"__global {parameter.data_type} *out_like_{parameter.name}"
                    )
                elif parameter.hint:
                    kernel_parameters.append(
                        f"__global {parameter.data_type} *{parameter.name}"
                    )
                elif parameter.dim:
                    kernel_parameters.append(f"{parameter.data_type} {parameter.name}")
                else:
                    kernel_parameters.append(
                        f"const {parameter.data_type} {parameter.name}"
                    )
            print(
                f"__kernel void {kernel.name}({', '.join(kernel_parameters)});",
                file=file,
            )
        print(file=file)

    def np_dtype_to_type_hint(np_dtype: str, is_buffer: bool) -> str:
        """Convert numpy dtype to Python type hint string."""
        if is_buffer:
            if np_dtype in ("float32", "float64", "float16"):
                return "NDArray[np.floating]"
            elif np_dtype in ("int32", "int64", "int16"):
                return "NDArray[np.signedinteger]"
            elif np_dtype in ("complex64", "complex128"):
                return "NDArray[np.complexfloating]"
            return f"NDArray  # {np_dtype}"
        else:
            # Scalars
            if np_dtype in ("float32", "float64", "float16"):
                return "float"
            elif np_dtype in ("int32", "int64", "int16"):
                return "int"
            return np_dtype

    def python_stubs(interface, file=None):
        """Generate Python stubs with proper type hints and docstrings."""
        print("from numpy.typing import NDArray", file=file)
        print("import numpy as np", file=file)
        print(file=file)

        class_name = interface.name.replace("_", " ").title().replace(" ", "")
        print(f"class {class_name}:", file=file)
        print(f'    """Interface: {interface.name}"""', file=file)
        print(file=file)

        for kernel in interface.kernels:
            # Build parameter list and return types
            params = []  # (name, type_hint, description)
            returns = []  # (type_hint, description)

            for param in kernel.parameters:
                is_buffer = bool(param.pointer)
                type_hint = np_dtype_to_type_hint(param.np_dtype, is_buffer)

                # Handle out_like first - these are derived but should be returns
                if param.hint == "out_like":
                    # Auto-allocated output - becomes return value
                    source = param.name.removeprefix("out_like_")
                    returns.append((type_hint, f"{param.np_dtype} array (like {source})"))
                    continue

                if param.derived:
                    # Skip derived params (width_of, height_of, etc.)
                    continue

                if param.hint == "out":
                    # Allocated output - becomes return value
                    returns.append((type_hint, f"{param.np_dtype} array"))
                elif param.hint in ("in", "inout", "resident"):
                    # User-provided parameter
                    role = {"in": "input", "inout": "input/output", "resident": "resident"}[
                        param.hint
                    ]
                    params.append((param.name, type_hint, f"{role} {param.np_dtype}"))
                elif not param.pointer:
                    # Scalar parameter
                    params.append((param.name, type_hint, f"{param.np_dtype} scalar"))
                else:
                    # Buffer without explicit hint
                    params.append((param.name, type_hint, f"{param.np_dtype} array"))

            # Build return type
            if not returns:
                return_type = "None"
            elif len(returns) == 1:
                return_type = returns[0][0]
            else:
                return_type = f"tuple[{', '.join(r[0] for r in returns)}]"

            # Generate method - kernel.name is the method name
            # kernel.alias is the base kernel it's derived from (if any)
            method_name = kernel.name

            # Signature
            if params:
                print(f"    def {method_name}(", file=file)
                print("        self,", file=file)
                for name, hint, _ in params:
                    print(f"        {name}: {hint},", file=file)
                print(f"    ) -> {return_type}:", file=file)
            else:
                print(f"    def {method_name}(self) -> {return_type}:", file=file)

            # Docstring
            print('        """', file=file)
            if params:
                print("        Args:", file=file)
                for name, _, desc in params:
                    print(f"            {name}: {desc}", file=file)
            if returns:
                print(file=file)
                print("        Returns:", file=file)
                for _, desc in returns:
                    print(f"            {desc}", file=file)
            print('        """', file=file)
            print("        ...", file=file)
            print(file=file)

    def python_pyi_stubs(interface, file=None):
        """Generate compact .pyi stub file format."""
        print("from numpy.typing import NDArray", file=file)
        print("import numpy as np", file=file)
        print(file=file)

        class_name = interface.name.replace("_", " ").title().replace(" ", "")
        print(f"class {class_name}:", file=file)

        for kernel in interface.kernels:
            params = []
            returns = []

            for param in kernel.parameters:
                is_buffer = bool(param.pointer)
                type_hint = np_dtype_to_type_hint(param.np_dtype, is_buffer)

                # Handle out_like first - these are derived but should be returns
                if param.hint == "out_like":
                    returns.append(type_hint)
                    continue

                if param.derived:
                    continue

                if param.hint == "out":
                    returns.append(type_hint)
                elif param.hint in ("in", "inout", "resident"):
                    params.append(f"{param.name}: {type_hint}")
                elif not param.pointer:
                    params.append(f"{param.name}: {type_hint}")
                else:
                    params.append(f"{param.name}: {type_hint}")

            # Return type
            if not returns:
                return_type = "None"
            elif len(returns) == 1:
                return_type = returns[0]
            else:
                return_type = f"tuple[{', '.join(returns)}]"

            # Method name - kernel.name is the actual method name
            method_name = kernel.name

            # Compact single-line format
            params_str = ", ".join(params)
            print(f"    def {method_name}(self, {params_str}) -> {return_type}: ...", file=file)

    requested_interfaces = args.interfaces if args.interfaces else []

    if args.list:
        print("# Available Interfaces")
        for interface_name in interfaces():
            print(f"  - {interface_name}")
        return 0

    if not requested_interfaces:
        # Get all interface names without .interface extension
        requested_interfaces = [name.replace(".interface", "") for name in interfaces()]

    for interface_name in requested_interfaces:
        # Render template with YAML context if needed
        rendered_template = render_interface_template(interface_name)

        # Skip if template still has unprocessed Mako constructs
        if "%" in rendered_template or "${" in rendered_template:
            print(f"\n## {interface_name}")
            print(
                f"Note: {interface_name} has unprocessed Mako template markup (no YAML context available)"
            )
            continue

        try:
            # Get interface object from rendered template
            # get_interface() expects the source text, not a name
            interface_obj = get_interface(rendered_template)
        except RuntimeError as e:
            print(f"\n## {interface_name}")
            print(f"Error parsing interface: {e}")
            continue

        if args.all or args.explain:
            print(f"\n## {interface_name}")
            explain_interface(interface_obj)

        if args.all or args.opencl:
            print(f"\n### OpenCL Stubs: {interface_name}")
            opencl_stubs(interface_obj)

        if args.all or args.python:
            print(f"\n### Python Stubs: {interface_name}")
            python_stubs(interface_obj)

        if args.all or args.stubs:
            print(f"\n### .pyi Stubs: {interface_name}")
            python_pyi_stubs(interface_obj)

        if args.write_lib:
            # Write .pyi stub to yapocis/lib/ directory
            from pathlib import Path

            import yapocis.lib

            lib_dir = Path(yapocis.lib.__file__).parent
            pyi_path = lib_dir / f"{interface_name}.pyi"

            with open(pyi_path, "w") as f:
                python_pyi_stubs(interface_obj, file=f)

            print(f"Wrote {pyi_path}")

    return 0


def main():
    """Main entry point for yapocis-tools."""
    parser = ArgumentParser(
        description="Yapocis development tools - inspect assets, backends, and interfaces",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available backends
  yapocis-tools backends

  # List all interfaces
  yapocis-tools interfaces --list

  # Show interface documentation
  yapocis-tools interfaces mandelbrot --explain

  # List all assets
  yapocis-tools assets --list --all

  # Find specific asset
  yapocis-tools assets test.jpg --images

  # List devices
  yapocis-tools devices --list

  # Download and convert pre-trained models
  yapocis-tools models vgg16
  yapocis-tools models resnet50 --output my_models/

  # Create extension project scaffolding
  yapocis-tools scaffold my_extension
  yapocis-tools scaffold my_extension --dry-run
  yapocis-tools scaffold my_extension --output /path/to/projects
  yapocis-tools scaffold my_extension --complete  # Add missing files
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Assets subcommand
    assets_parser = subparsers.add_parser("assets", help="List and find yapocis assets")
    assets_parser.add_argument("--images", action="store_true", help="Search in images")
    assets_parser.add_argument("--icons", action="store_true", help="Search in icons")
    assets_parser.add_argument("--fonts", action="store_true", help="Search in fonts")
    assets_parser.add_argument(
        "--kernels", action="store_true", help="Search in kernels"
    )
    assets_parser.add_argument(
        "--interfaces", action="store_true", help="Search in interfaces"
    )
    assets_parser.add_argument(
        "--all", action="store_true", help="Search all asset types"
    )
    assets_parser.add_argument(
        "--list", action="store_true", help="List available assets"
    )
    assets_parser.add_argument("assets", nargs="*", help="Asset names to find")

    # Devices subcommand
    devices_parser = subparsers.add_parser(
        "devices", help="List OpenCL/Metal/CUDA devices"
    )
    devices_parser.add_argument(
        "--list", action="store_true", help="List device names only"
    )
    devices_parser.add_argument("devices", nargs="*", help="Specific devices to query")

    # Backends subcommand
    backends_parser = subparsers.add_parser(
        "backends", help="List all available backends"
    )
    backends_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show environment variables"
    )
    backends_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Interfaces subcommand
    interfaces_parser = subparsers.add_parser(
        "interfaces", help="Generate interface documentation and stubs"
    )
    interfaces_parser.add_argument(
        "--list", action="store_true", help="List available interfaces"
    )
    interfaces_parser.add_argument(
        "--all", action="store_true", help="Generate all output types"
    )
    interfaces_parser.add_argument(
        "--explain", action="store_true", help="Generate human-readable docs"
    )
    interfaces_parser.add_argument(
        "--opencl", action="store_true", help="Generate OpenCL stubs"
    )
    interfaces_parser.add_argument(
        "--python", action="store_true", help="Generate Python stubs with docstrings"
    )
    interfaces_parser.add_argument(
        "--stubs", action="store_true", help="Generate compact .pyi stub format"
    )
    interfaces_parser.add_argument(
        "--write-lib",
        action="store_true",
        help="Write .pyi stubs to yapocis/lib/ directory",
    )
    interfaces_parser.add_argument(
        "interfaces", nargs="*", help="Interface names to process"
    )

    # Models subcommand
    models_parser = subparsers.add_parser(
        "models", help="Download and convert pre-trained model weights"
    )
    models_parser.add_argument(
        "model", choices=["vgg16", "resnet50"], help="Model to download"
    )
    models_parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: src/yapocis/models/<model>)",
    )
    models_parser.add_argument(
        "--no-labels", action="store_true", help="Skip downloading ImageNet labels"
    )

    # Scaffold subcommand
    scaffold_parser = subparsers.add_parser(
        "scaffold", help="Create scaffolding for a yapocis extension project"
    )
    scaffold_parser.add_argument(
        "name", help="Name of the extension project (must be valid Python identifier)"
    )
    scaffold_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory (default: ./<name>)",
    )
    scaffold_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating files",
    )
    scaffold_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files",
    )
    scaffold_parser.add_argument(
        "--complete",
        "-c",
        action="store_true",
        help="Add missing files/directories without overwriting existing ones",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    if args.command == "assets":
        return cmd_assets(args)
    elif args.command == "devices":
        return cmd_devices(args)
    elif args.command == "backends":
        return cmd_backends(args)
    elif args.command == "interfaces":
        return cmd_interfaces(args)
    elif args.command == "models":
        return cmd_models(args)
    elif args.command == "scaffold":
        return cmd_scaffold(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
