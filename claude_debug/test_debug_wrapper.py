#!/usr/bin/env python3
"""Debug the wrapper call step by step."""
import os
os.environ['YAPOCIS_BACKEND'] = 'cuda'

import numpy as np
import cupy as cp
from yapocis.cu.kernel_manager import CUDAKernelManager
from yapocis.core.devbuf import ensure_device, empty_like_device
from yapocis.core.autopad import pad_sizes
from yapocis.core.tunecache import get_local_size

print("Creating CUDAKernelManager...")
mgr = CUDAKernelManager()

print("Loading imageops program...")
prog = mgr.load_program('imageops')

# Get kernel spec
kspec = next((k for k in prog.iface.kernels if k.name == 'clamp_pixels'), None)
print(f"Kernel spec: {kspec}")
print(f"  name: {kspec.name}")
print(f"  has args: {hasattr(kspec, 'args')}")
aspecs = getattr(kspec, "args", None)
print(f"  aspecs: {aspecs}")

# Create test data
a = np.random.random(100).astype(np.float32)
low = 0.1
high = 0.9

print(f"\nInput args:")
print(f"  a: type={type(a)}, shape={a.shape}, dtype={a.dtype}")
print(f"  low: {low} (type={type(low)})")
print(f"  high: {high} (type={type(high)})")

# Simulate what the wrapper does
args = (a, low, high)
print(f"\nargs tuple: {len(args)} elements")

print("\nProcessing arguments...")
if aspecs is not None and len(aspecs) == len(args):
    print("Using aspecs path")
else:
    print("Using fallback path (no aspecs)")
    dev_args = [ensure_device(arg) for arg in args]
    for i, da in enumerate(dev_args):
        print(f"  dev_args[{i}]: type={type(da)}, value/shape={da if isinstance(da, (int, float)) else da.shape}")

# Check pad_sizes
print("\nCalling pad_sizes...")
global_size = None
local_size = None
default_local = (16, 16)
cached_ls = get_local_size(prog.name, 'clamp_pixels', backend='cuda')
preferred = cached_ls or default_local
print(f"  global_size: {global_size}")
print(f"  local_size: {local_size}")
print(f"  preferred: {preferred}")

pg, pl = pad_sizes(global_size, local_size, tuple(args), preferred)
print(f"  pad_sizes returned: pg={pg}, pl={pl}")

gs = pg if pg is not None else global_size
ls = pl if pl is not None else local_size
print(f"  final gs={gs}, ls={ls}")

# Check _grid_block
print("\nCalculating grid/block...")
def _grid_block(gs, ls):
    def cd(a,b): return (a + b - 1)//b
    if isinstance(gs, int):
        tx = (ls or (256,))[0]
        result = (cd(gs,tx),1,1),(tx,1,1)
        print(f"  1D mode: gs={gs}, tx={tx} -> {result}")
        return result
    dims = list(gs or (1,)) + [1,1]; dims=dims[:3]
    lsz  = list(ls or (16,16,1)) + [1,1]; lsz=lsz[:3]
    grid = tuple(cd(dims[i], lsz[i]) for i in range(3))
    block= tuple(lsz)
    result = grid, block
    print(f"  ND mode: dims={dims}, lsz={lsz} -> {result}")
    return result

grid, block = _grid_block(gs, ls)
print(f"Final grid={grid}, block={block}")

print("\nExpected for 100 elements:")
print(f"  Should be something like: grid=(1,), block=(256,) or similar")
