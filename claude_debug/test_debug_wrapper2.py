#!/usr/bin/env python3
"""Debug the argument processing in detail."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp
from yapocis.cu.kernel_manager import CUDAKernelManager

print("Creating CUDAKernelManager...")
mgr = CUDAKernelManager()

print("Loading imageops program...")
prog = mgr.load_program('imageops')

# Get kernel spec
kspec = next((k for k in prog.iface.kernels if k.name == 'clamp_pixels'), None)
print(f"\nKernel spec: {kspec.name}")
print(f"Parameters ({len(kspec.parameters)}):")
for i, param in enumerate(kspec.parameters):
    print(f"  [{i}] {param.name}")
    print(f"      hint={param.hint}, derived={param.derived}, derived_from={param.derived_from}")
    print(f"      type={param.data_type}, pointer={param.pointer}")

# Create test data
a = np.array([0.05, 0.5, 0.95], dtype=np.float32)
low = 0.1
high = 0.9

print(f"\nInput array: {a}")
print(f"low={low}, high={high}")

# Manually simulate the argument processing
print("\nSimulating argument processing...")
aspecs = kspec.parameters
args = (a, low, high)
dev_args = []
out_idxs = []

arg_idx = 0
for i, spec in enumerate(aspecs):
    hint = getattr(spec, "hint", "")
    is_derived = getattr(spec, "derived", False)

    print(f"\nProcessing param[{i}]: {spec.name}")
    print(f"  hint={hint}, derived={is_derived}, derived_from={spec.derived_from}")

    if is_derived and hint.startswith("out_like"):
        print("  -> Auto-allocating output")
        derived_from = getattr(spec, "derived_from", None)
        if derived_from:
            ref_idx = next((j for j, s in enumerate(aspecs) if s.name == derived_from and not s.derived), None)
            print(f"  -> Looking for ref '{derived_from}', found at idx {ref_idx}")
            if ref_idx is not None and ref_idx < len(dev_args):
                ref = dev_args[ref_idx]
                print(f"  -> Ref shape: {ref.shape}, dtype: {ref.dtype}")
                dev = cp.empty_like(ref)
                print(f"  -> Created output: shape={dev.shape}, dtype={dev.dtype}")
                dev_args.append(dev)
                out_idxs.append(i)
                continue

    # Use provided argument
    if arg_idx < len(args):
        aval = args[arg_idx]
        print(f"  -> Using arg[{arg_idx}]: {aval}")
        arg_idx += 1

        # Convert to GPU
        if isinstance(aval, np.ndarray):
            dev = cp.asarray(aval)
            print(f"  -> Converted to GPU: shape={dev.shape}, dtype={dev.dtype}")
        else:
            dev = aval
            print(f"  -> Scalar: {dev}")
        dev_args.append(dev)

        if hint in ("out", "inout", "resident") or hint.startswith("out_"):
            out_idxs.append(i)

print(f"\nFinal dev_args: {len(dev_args)} arguments")
for i, arg in enumerate(dev_args):
    if hasattr(arg, 'shape'):
        print(f"  [{i}] GPU array: shape={arg.shape}, dtype={arg.dtype}")
    else:
        print(f"  [{i}] Scalar: {arg}")

print(f"\nOutput indices: {out_idxs}")
