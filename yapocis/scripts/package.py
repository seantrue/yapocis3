#!/usr/bin/env python3
"""
Generate compiled kernel packages from interface specifications.

Kernel packages eliminate runtime parameter interpretation overhead by
pre-compiling all buffer management logic into typed Python methods.

Usage:
    yapocis-package <interface_name> --backend <backend>
    yapocis-package all --backend ocl
    yapocis-package convolvesep --backend metal --output yapocis/kernelpack/metal/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mako.template import Template

from yapocis.interfaces import find_interface
from yapocis.core.interface_context import load_interface_context


# Backend-specific imports
BACKEND_IMPORTS = {
    "ocl": "from yapocis.ocl.device_manager import DeviceManager",
    "metal": "from yapocis.metallic.device_manager import DeviceManager",
    "cuda": "from yapocis.cu.device_manager import DeviceManager",
    "numbax": "from yapocis.numbax.device_manager import DeviceManager",
}

BACKEND_NAMES = {
    "ocl": "OpenCL",
    "metal": "Metal",
    "cuda": "CUDA",
    "numbax": "Numba",
}

# All available interfaces
ALL_INTERFACES = [
    "convolvesep",
    "operators",
    "gradient",
    "median3x3",
    "colormath",
    "hsi",
    "zcs",
    "anisodiff",
    "graphics4",
    "truetrace",
]


def render_interface_with_context(interface_text: str, interface_name: str) -> str:
    """
    Render interface template with YAML context if available.

    For interfaces that use Mako templates (operators, convolvesep),
    this loads the YAML context and renders the template.

    For static interfaces, returns the text unchanged.
    """
    # Check if interface uses Mako templates
    if '%' not in interface_text and '${' not in interface_text:
        return interface_text

    # Load YAML context
    context = load_interface_context(interface_name)

    if not context:
        print(f"Warning: Interface '{interface_name}' uses Mako templates but no YAML context found")
        print(f"Expected: yapocis/interfaces/{interface_name}.yml")
        return interface_text

    # Special handling for convolvesep: generate 'convs' from 'gaussian_params'
    if interface_name == 'convolvesep' and 'gaussian_params' in context:
        import numpy as np
        from yapocis.lib.gaussian import GaussianWeights

        print("Generating 'convs' from gaussian_params for convolvesep template")
        gaussian_params = context['gaussian_params']

        # Generate Gaussian weights using the same logic as Gaussian.__init__
        gw = GaussianWeights(**gaussian_params)
        gws = gw.gaussian_weights()

        # Create convs list: [(name, array), ...]
        convs = [("gauss%s" % len(a), a) for a in gws]

        # Add to context
        context['convs'] = convs
        print(f"Generated {len(convs)} Gaussian kernels for package generation")

    # Render template with context
    try:
        template = Template(interface_text)
        rendered = template.render(**context)
        return rendered
    except Exception as e:
        print(f"Error rendering interface template: {e}")
        print(f"Context keys: {list(context.keys())}")
        raise


def parse_interface_simple(interface_text: str) -> dict[str, Any]:
    """
    Simple interface parser for kernel package generation.

    This is a simplified parser that extracts kernel signatures without
    full IDL parsing. It handles the common patterns in yapocis interfaces.

    Returns dict with:
        - name: interface name
        - kernels: list of (kernel_name, params_str, is_extension)
    """
    import re

    kernels = []
    interface_name = None

    # Extract interface name
    match = re.search(r'interface\s+(\w+)\s*{', interface_text)
    if match:
        interface_name = match.group(1)

    # Find all kernel and extension declarations
    # Pattern: kernel name(...); or extension name of base(...);
    kernel_pattern = r'^\s*(kernel|extension)\s+(\w+)(?:\s+of\s+(\w+))?\s*\((.*?)\)\s*;'

    for line in interface_text.split('\n'):
        # Skip Mako template lines
        if '%' in line or '${' in line:
            continue

        match = re.match(kernel_pattern, line)
        if match:
            decl_type = match.group(1)  # 'kernel' or 'extension'
            name = match.group(2)
            base = match.group(3)  # For extensions: the base kernel name
            params = match.group(4)

            is_extension = (decl_type == 'extension')

            kernels.append({
                'name': name,
                'base': base,
                'params': params,
                'is_extension': is_extension,
            })

    return {
        'name': interface_name,
        'kernels': kernels,
    }


def parse_parameters(params_str: str) -> list[dict[str, str]]:
    """
    Parse parameter string into structured parameters.

    Handles patterns like:
        in float a
        out_like a
        width_of input
        resident float buffer
        size_of int a
    """
    import re

    params = []

    # Split by commas, but be careful with types that might have commas
    param_parts = [p.strip() for p in params_str.split(',')]

    for part in param_parts:
        if not part:
            continue

        tokens = part.split()
        if len(tokens) < 1:
            continue

        param = {
            'hint': None,
            'dtype': None,
            'name': None,
            'derived_from': None,
        }

        # Parse hint keywords
        if tokens[0] in ('in', 'out', 'resident', 'inout'):
            param['hint'] = tokens[0]
            tokens = tokens[1:]
        elif tokens[0] in ('out_like', 'width_of', 'height_of', 'size_of', 'depth_of'):
            param['hint'] = tokens[0]
            # Pattern: "size_of int a" or "out_like a"
            # Skip dtype if present (e.g., "size_of int a" -> derived_from="a")
            # For "out_like a" -> derived_from="a"
            if len(tokens) >= 3:
                # e.g., "size_of int a"
                param['dtype'] = tokens[1]
                param['derived_from'] = tokens[2]
            elif len(tokens) == 2:
                # e.g., "out_like a"
                param['derived_from'] = tokens[1]
            param['name'] = f"__{tokens[0]}_{param['derived_from']}"
            params.append(param)
            continue

        # Parse dtype and name
        if len(tokens) >= 2:
            param['dtype'] = tokens[0]
            param['name'] = tokens[1]
        elif len(tokens) == 1:
            # Scalar type without hint (e.g., "int vertical")
            param['dtype'] = None
            param['name'] = tokens[0]

        params.append(param)

    return params


def generate_method_signature(kernel: dict[str, Any], params: list[dict[str, str]]) -> str:
    """Generate Python method signature from kernel parameters."""
    # Filter out derived parameters (width_of, height_of, etc.)
    user_params = [p for p in params if not p['hint'] in ('width_of', 'height_of', 'size_of', 'depth_of')]

    # Build argument list
    args = []
    for p in user_params:
        if p['hint'] in ('in', 'resident', 'inout'):
            args.append(f"{p['name']}: np.ndarray")
        elif p['hint'] == 'out' and not kernel['is_extension']:
            # Regular kernels allocate output internally
            pass
        elif p['hint'] == 'out' or (kernel['is_extension'] and p['hint'] == 'resident'):
            # Extension methods take output as parameter
            args.append(f"{p['name']}: np.ndarray")
        elif p['hint'] == 'out_like':
            # Allocates output based on input shape
            pass
        elif p['dtype'] in ('int', 'float', 'short', 'double'):
            # Scalar parameters
            args.append(f"{p['name']}: np.{p['dtype']}32" if p['dtype'] != 'double' else f"{p['name']}: np.float64")

    return ', '.join(args) if args else ''


def generate_package_class(interface_name: str, spec: dict[str, Any], backend: str) -> str:
    """Generate complete Python package class for an interface."""

    class_name = f"{interface_name.capitalize()}Package"
    backend_import = BACKEND_IMPORTS[backend]
    backend_name = BACKEND_NAMES[backend]

    # Header
    lines = []
    lines.append('"""')
    lines.append(f'Auto-generated {backend_name} kernel package for {interface_name} interface.')
    lines.append('')
    lines.append(f'Generated from: yapocis/interfaces/{interface_name}.interface')
    lines.append(f'Backend: {backend_name}')
    lines.append(f'Generation date: {datetime.now().strftime("%Y-%m-%d")}')
    lines.append('')
    lines.append('This package provides zero-overhead calls to GPU kernels.')
    lines.append('"""')
    lines.append('')
    lines.append('import numpy as np')
    lines.append(backend_import)

    # Check if any kernel has multiple outputs (need Tuple import)
    has_multi_output = any(
        len([p for p in kernel.get('params', '').split(',') if 'out_like' in p or 'out ' in p]) > 1
        for kernel in spec['kernels']
        if not kernel.get('is_extension', False)
    )
    if has_multi_output:
        lines.append('from typing import Tuple')

    lines.append('')
    lines.append('')

    # Class definition
    lines.append(f'class {class_name}:')
    lines.append('    """')
    lines.append(f'    Compiled kernel package for {interface_name} operations.')
    lines.append('    """')
    lines.append('')

    # __init__ method
    lines.append('    def __init__(self, dmgr: DeviceManager, program):')
    lines.append('        self.dmgr = dmgr')
    lines.append('        self.queue = dmgr.queue')
    lines.append('        self.autoflatten = dmgr.autoflatten')
    lines.append('        self.program = program')

    # Determine which kernels to track
    base_kernels = [k for k in spec['kernels'] if not k['is_extension'] and not k['name'].endswith('_h') and not k['name'].endswith('_v')]

    if len(base_kernels) == 1:
        lines.append('        self._kernel = None')
    else:
        lines.append('        self._kernels: dict[str, any] = {}')

    lines.append('')

    # load() method
    if len(base_kernels) == 1:
        kernel_name = base_kernels[0]['name']
        lines.append('    def load(self) -> None:')
        lines.append(f'        """Load {kernel_name} kernel from the program."""')
        lines.append(f'        self._kernel = self.program.{kernel_name}')
    else:
        lines.append('    def load(self, kernel_names: list[str] | None = None) -> None:')
        lines.append('        """Load specified kernels from the program."""')
        lines.append('        if kernel_names is None:')
        lines.append('            kernel_names = [')
        for k in base_kernels:
            lines.append(f'                "{k["name"]}",')
        lines.append('            ]')
        lines.append('        ')
        lines.append('        for name in kernel_names:')
        lines.append('            self._kernels[name] = getattr(self.program, name)')

    lines.append('')

    # Generate methods for each kernel
    for kernel in spec['kernels']:
        lines.extend(generate_kernel_method(kernel, interface_name, len(base_kernels) > 1))
        lines.append('')

    return '\n'.join(lines)


def generate_kernel_method(kernel: dict[str, Any], interface_name: str, multi_kernel: bool) -> list[str]:
    """Generate a single kernel method."""
    lines = []

    # For extensions, the method name should be base_extension
    # e.g., "extension res of median3x3" -> method name "median3x3_res"
    if kernel['is_extension'] and kernel['base']:
        method_name = f"{kernel['base']}_{kernel['name']}"
    else:
        method_name = kernel['name']

    params = parse_parameters(kernel['params'])

    # Method signature
    sig_params = generate_method_signature(kernel, params)

    # Determine return type
    output_params = [p for p in params if p['hint'] in ('out', 'out_like')]
    non_resident_outputs = [p for p in output_params if not (kernel['is_extension'] and p['hint'] == 'resident')]
    num_outputs = len(non_resident_outputs)

    # Special handling for truetrace multi-output or any multi-output kernel
    if num_outputs > 1 and not kernel['is_extension']:
        return_type = f' -> Tuple[{", ".join(["np.ndarray"] * num_outputs)}]'
    elif num_outputs == 1 and not kernel['is_extension']:
        return_type = ' -> np.ndarray'
    else:
        return_type = ' -> None'

    lines.append(f'    def {method_name}(self, {sig_params}){return_type}:')
    lines.append('        """')

    # Docstring
    if kernel['is_extension']:
        lines.append(f'        Execute {method_name} with resident buffers.')
    else:
        lines.append(f'        Execute {method_name} kernel.')
    lines.append('        """')

    # Method body - parameter handling
    kernel_args = []
    output_arrays = []
    output_counter = {}  # Track multiple outputs of the same derived_from

    for i, p in enumerate(params):
        if p['hint'] == 'width_of':
            lines.append(f'        # Param {i}: width_of {p["derived_from"]}')
            lines.append(f'        width = np.int32({p["derived_from"]}.shape[1] if len({p["derived_from"]}.shape) > 1 else {p["derived_from"]}.shape[0])')
            kernel_args.append('width')
        elif p['hint'] == 'height_of':
            lines.append(f'        # Param {i}: height_of {p["derived_from"]}')
            lines.append(f'        height = np.int32({p["derived_from"]}.shape[0] if len({p["derived_from"]}.shape) > 1 else 1)')
            kernel_args.append('height')
        elif p['hint'] == 'size_of':
            lines.append(f'        # Param {i}: size_of {p["dtype"]} {p["derived_from"]}')
            lines.append(f'        size = np.int32({p["derived_from"]}.size)')
            kernel_args.append('size')
        elif p['hint'] == 'in':
            lines.append(f'        # Param {i}: in {p["dtype"]} {p["name"]}')
            dtype_map = {'float': 'float32', 'int': 'int32', 'short': 'int16', 'double': 'float64'}
            np_dtype = dtype_map.get(p['dtype'], p['dtype'])
            lines.append(f'        {p["name"]} = {p["name"]}.astype(np.{np_dtype}, copy=False)')
            lines.append(f'        buf_{p["name"]} = self.dmgr.write({p["name"]}, wait=True)')
            kernel_args.append(f'buf_{p["name"]}')
        elif p['hint'] == 'resident':
            lines.append(f'        # Param {i}: resident {p["dtype"]} {p["name"]}')
            lines.append(f'        buf_{p["name"]} = self.dmgr.buffers.get(id({p["name"]}))')
            lines.append(f'        if buf_{p["name"]} is None:')
            lines.append(f'            raise ValueError("{p["name"]} array not resident on GPU.")')
            kernel_args.append(f'buf_{p["name"]}')
        elif p['hint'] == 'out_like':
            lines.append(f'        # Param {i}: out_like {p["derived_from"]}')
            # Find the dtype of the derived_from parameter
            derived_param = next((pp for pp in params if pp['name'] == p["derived_from"]), None)
            if derived_param and derived_param['dtype']:
                dtype_map = {'float': 'float32', 'int': 'int32', 'short': 'int16', 'double': 'float64'}
                np_dtype = dtype_map.get(derived_param['dtype'], derived_param['dtype'])
            else:
                np_dtype = 'float32'  # default

            # Handle multiple out_like parameters with the same derived_from
            base_name = p["derived_from"]
            if base_name not in output_counter:
                output_counter[base_name] = 0
            else:
                output_counter[base_name] += 1

            # Generate unique output names for multiple outputs
            if output_counter[base_name] == 0:
                output_name = 'output' if num_outputs == 1 else 'output_0'
            else:
                output_name = f'output_{output_counter[base_name]}'

            lines.append(f'        {output_name} = np.empty_like({p["derived_from"]}, dtype=np.{np_dtype})')
            lines.append(f'        buf_{output_name} = self.dmgr.ensure({output_name})')
            kernel_args.append(f'buf_{output_name}')
            output_arrays.append(output_name)
        elif p['hint'] == 'out' and kernel['is_extension']:
            lines.append(f'        # Param {i}: out {p["dtype"]} {p["name"]}')
            lines.append(f'        buf_{p["name"]} = self.dmgr.buffers.get(id({p["name"]}))')
            lines.append(f'        if buf_{p["name"]} is None:')
            lines.append(f'            raise ValueError("{p["name"]} array not resident on GPU.")')
            kernel_args.append(f'buf_{p["name"]}')
        elif p['hint'] == 'out':
            # This is handled implicitly for non-extension kernels
            pass
        elif p['dtype'] in ('int', 'float', 'short', 'double'):
            lines.append(f'        # Param {i}: {p["dtype"]} {p["name"]} (scalar)')
            dtype_map = {'int': 'int32', 'float': 'float32', 'short': 'int16', 'double': 'float64'}
            np_dtype = dtype_map.get(p['dtype'], p['dtype'])
            lines.append(f'        {p["name"]}_val = np.{np_dtype}({p["name"]})')
            kernel_args.append(f'{p["name"]}_val')

    lines.append('')

    # Determine global size
    first_array = next((p['name'] for p in params if p['hint'] in ('in', 'resident')), None)
    if first_array:
        lines.append('        # Execute kernel')
        lines.append(f'        global_size = ({first_array}.size,) if self.autoflatten else {first_array}.shape')
    else:
        lines.append('        # Execute kernel')
        lines.append('        global_size = (1,)')

    # Kernel invocation
    kernel_ref = 'self._kernel' if not multi_kernel else f'self._kernels["{kernel["name"]}"]'
    lines.append(f'        {kernel_ref}(')
    lines.append('            self.queue,')
    lines.append('            global_size,')
    lines.append('            None,  # local_size')
    for arg in kernel_args:
        lines.append(f'            {arg},')
    lines.append('        )')

    # Return handling
    if output_arrays:
        lines.append('')
        lines.append('        # Read results back to host')
        for arr in output_arrays:
            lines.append(f'        self.dmgr.read({arr})')

        if interface_name == 'truetrace' and len(output_arrays) > 1:
            lines.append(f'        return ({", ".join(output_arrays)})')
        elif len(output_arrays) == 1:
            lines.append(f'        return {output_arrays[0]}')

    return lines


def generate_package(interface_name: str, backend: str, output_path: Path | None = None) -> Path:
    """Generate a kernel package for the specified interface and backend."""

    # Find interface file
    try:
        interface_path = Path(find_interface(interface_name))
    except FileNotFoundError:
        print(f"Error: Interface '{interface_name}' not found")
        sys.exit(1)

    # Read interface text
    interface_text = interface_path.read_text()

    # Render interface with YAML context if it uses Mako templates
    print(f"Processing interface: {interface_name}")
    rendered_text = render_interface_with_context(interface_text, interface_name)

    # Parse interface (simple parsing for code generation)
    spec = parse_interface_simple(rendered_text)

    if not spec['kernels']:
        print(f"Error: No kernels found in {interface_name}.interface after rendering")
        print("This may indicate an issue with the interface specification or YAML context.")
        sys.exit(1)

    print(f"Found {len(spec['kernels'])} kernels in {interface_name}")

    # Generate package code
    package_code = generate_package_class(interface_name, spec, backend)

    # Determine output path
    if output_path is None:
        output_path = Path(f'yapocis/kernelpack/{backend}/{interface_name}.py')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(package_code)

    print(f"Generated: {output_path}")
    return output_path


def main():
    """Main entry point for package generator."""
    parser = argparse.ArgumentParser(
        description='Generate compiled kernel packages from interface specifications'
    )
    parser.add_argument(
        'interface',
        help='Interface name (e.g., median3x3) or "all" for all interfaces'
    )
    parser.add_argument(
        '--backend',
        '-b',
        required=True,
        choices=['ocl', 'metal', 'cuda', 'numbax'],
        help='Backend to generate for'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        help='Output directory (default: yapocis/kernelpack/<backend>/)'
    )

    args = parser.parse_args()

    # Generate for one or all interfaces
    if args.interface == 'all':
        for interface_name in ALL_INTERFACES:
            try:
                output_path = args.output / f'{interface_name}.py' if args.output else None
                generate_package(interface_name, args.backend, output_path)
            except Exception as e:
                print(f"Error generating {interface_name}: {e}")
    else:
        generate_package(args.interface, args.backend, args.output)


if __name__ == '__main__':
    main()
