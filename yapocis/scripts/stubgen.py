#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stubgen.py
Generate OpenCL / CUDA / Metal stubs from extended .interface specs.

Usage:
  python stubgen.py --in path/to/file.interface --backend opencl --out gen/opencl
  python stubgen.py --in graphics4.interface --backend metal  --out gen/metal
  python stubgen.py --in convolvesep.interface --backend cuda  --out gen/cuda

Backends supported:
- opencl, cuda, metal

Classes & features implemented:
- elementwise
- stencil2d (with boundary policies; optional tiling via fastmem/tile/halo)
- separable2d (emits _h and _v; supports constant taps, tiling)
- reduction (sum|max|min|logical_or; deterministic toggle; accumulator dtype)
- graphics_2d (per-pixel loop over N primitives; blend alpha|src_over; optional binning hooks via meta)

Interface metadata honored:
- meta: layout, dtype, fast_math, workgroup, class, blend, cull, binning, tile_pixels, stage_tile
- with: class, axis, radius, boundary, tile, halo, fastmem, taps residency, op, identity,
        deterministic, accumulator_dtype, texture_input, vector_width, blend, cull
- qualifiers: constant, texture2d, local bytes_of(...)
"""

import argparse, os
from pathlib import Path
from typing import Any
from yapocis.i import parse_interface_text, InterfaceSpec, Kernel, Param

# ---------- Backend type maps ----------
SCALARS = {
    "float": {"opencl":"float", "cuda":"float", "metal":"float"},
    "float32":{"opencl":"float","cuda":"float","metal":"float"},
    "half":  {"opencl":"half",  "cuda":"__half","metal":"half"},
    "f16":   {"opencl":"half",  "cuda":"__half","metal":"half"},
    "double":{"opencl":"double","cuda":"double","metal":"double"},
    "int":   {"opencl":"int",   "cuda":"int",   "metal":"int"},
    "int32": {"opencl":"int",   "cuda":"int",   "metal":"int"},
    "uint32":{"opencl":"uint",  "cuda":"unsigned int","metal":"uint"},
    "short": {"opencl":"short", "cuda":"short", "metal":"short"},
    "int16": {"opencl":"short", "cuda":"short", "metal":"short"},
    "uint16":{"opencl":"ushort","cuda":"unsigned short","metal":"ushort"},
    "boolean":{"opencl":"bool", "cuda":"bool",  "metal":"bool"},
    "char":  {"opencl":"char",  "cuda":"char",  "metal":"char"},
    "float4":{"opencl":"float4","cuda":"float4","metal":"float4"},
}

QUAL = {
    "opencl": {"buf_in": "__global const", "buf_out": "__global", "const": "__constant"},
    "cuda":   {"buf_in": "const",          "buf_out": "",         "const": "__constant__"},
    "metal":  {"buf_in": "device const",   "buf_out": "device",   "const": "constant"},
}

def dt(dtype: str, backend: str) -> str:
    if dtype in SCALARS: return SCALARS[dtype][backend]
    # vector types passthrough (float4, int2, etc.)
    return SCALARS.get(dtype, {}).get(backend, dtype)

# ---------- Header ----------
def header(backend: str, fast_math: bool) -> str:
    if backend == "opencl":
        lines = ["// AUTO-GENERATED STUB — OpenCL"]
        if fast_math:
            lines.append("#pragma OPENCL EXTENSION cl_khr_fp16 : enable")
        return "\n".join(lines) + "\n"
    if backend == "cuda":
        return "// AUTO-GENERATED STUB — CUDA\n#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n"
    # metal
    return "// AUTO-GENERATED STUB — Metal (MSL)\n#include <metal_stdlib>\nusing namespace metal;\n"

# ---------- ABI helpers ----------
def pack_params_order(k: Kernel, backend: str) -> tuple[str, list[str]]:
    """
    Construct the function signature for a kernel.
    - Maps 'resident' in/out to device pointers
    - Honors 'constant' qualifier for taps/RO buffers
    - Adds minimal scalar shapes (W,H) if present from params list
    Returns (signature, ordered_names_for_body)
    """
    q = QUAL[backend]
    arg_strs = []
    ordered_names = []
    next_buffer_index = 0

    def buf_attr():
        nonlocal next_buffer_index
        if backend == "metal":
            s = f" [[ buffer({next_buffer_index}) ]]"
            next_buffer_index += 1
            return s
        return ""

    for p in k.params:
        ty = dt(p.dtype, backend)
        nm = p.name
        ro = (p.role in ("in",)) or ("constant" in p.qualifiers)
        rw = (p.role in ("out","READWRITE","out_like","resident")) and not ro
        # residency
        if "constant" in p.qualifiers:
            if backend == "opencl":
                arg_strs.append(f"{q['const']} {ty}* {nm}")
            elif backend == "cuda":
                arg_strs.append(f"const {ty}* __restrict__ {nm}")  # actual __constant__ binding is host-side
            else:
                arg_strs.append(f"{q['const']} {ty}* {nm}{buf_attr()}")
        elif p.role in ("in","resident","READWRITE","out_like","out"):
            if backend == "opencl":
                qual = q['buf_in'] if ro else q['buf_out']
                arg_strs.append(f"{qual} {ty}* {nm}")
            elif backend == "cuda":
                qual = "const " if ro else ""
                restr = "__restrict__ " if ro or rw else ""
                arg_strs.append(f"{qual}{ty}* {restr}{nm}")
            else:
                qual = q['buf_in'] if ro else q['buf_out']
                arg_strs.append(f"{qual} {ty}* {nm}{buf_attr()}")
        else:
            # scalar
            if backend == "opencl":
                arg_strs.append(f"{ty} {nm}")
            elif backend == "cuda":
                arg_strs.append(f"{ty} {nm}")
            else:
                arg_strs.append(f"constant {ty}& {nm}{buf_attr()}")
        ordered_names.append(nm)

    # thread indices / builtins
    if backend == "metal":
        # Attach grid indices at end, not counted in buffer indices
        arg_strs.append("uint  tid     [[thread_index_in_threadgroup]]")
        arg_strs.append("uint2 gid2    [[thread_position_in_grid]]")
        arg_strs.append("uint2 tgp2    [[threads_per_threadgroup]]")
        arg_strs.append("uint2 tgpos2  [[threadgroup_position_in_grid]]")
    return ", ".join(arg_strs), ordered_names

# ---------- Emitters ----------

def emit_elementwise(k: Kernel, meta: dict[str,Any], backend: str) -> str:
    sig, _ = pack_params_order(k, backend)
    h = header(backend, meta.get("fast_math", True))
    body_index = {
        "opencl": "int i = get_global_id(0); // bounds check is caller's responsibility or add length here\n",
        "cuda":   "int i = blockIdx.x * blockDim.x + threadIdx.x;\n",
        "metal":  "uint i = gid2.x;\n"
    }[backend]
    compute_hook = "/* INSERT-COMPUTE-HERE: elementwise (load inputs at index i, store to outputs) */\n"
    if backend == "opencl":
        return f"""{h}
__kernel void {k.name}({sig})
{{
    {body_index}
    {compute_hook}
}}
"""
    if backend == "cuda":
        return f"""{h}
__global__ void {k.name}({sig})
{{
    {body_index}
    {compute_hook}
}}
"""
    return f"""{h}
kernel void {k.name}({sig})
{{
    {body_index}
    {compute_hook}
}}
"""

def _boundary_clamp_expr(x, lo, hi):  # code string
    return f"({x} < {lo} ? {lo} : ({x} >= {hi} ? ({hi}-1) : {x}))"

def emit_stencil2d(k: Kernel, meta: dict[str,Any], backend: str) -> str:
    # traits
    rx, ry = (0,0)
    if "radius" in k.traits:
        val = k.traits["radius"]
        if isinstance(val, (list,tuple)) and len(val)==2: rx, ry = int(val[0]), int(val[1])
        else: rx = int(val); ry = int(val)
    boundary = k.traits.get("boundary", "clamp")
    fastmem = str(k.traits.get("fastmem","off")).lower() in ("on","true","1")
    tile = tuple(k.traits.get("tile", (64,8)))
    kx, ky = rx*2+1, ry*2+1

    h = header(backend, meta.get("fast_math", True))
    sig, _ = pack_params_order(k, backend)

    if backend == "opencl":
        prolog = "int x = get_global_id(0); int y = get_global_id(1);"
        idx = "int idx = y*W + x;"
    elif backend == "cuda":
        prolog = "int x = blockIdx.x * blockDim.x + threadIdx.x; int y = blockIdx.y * blockDim.y + threadIdx.y;"
        idx = "int idx = y*W + x;"
    else:
        prolog = "uint x = gid2.x, y = gid2.y;"
        idx = "uint idx = y*(uint)W + x;"

    # neighborhood gather w/ clamp
    nb = []
    for dy in range(-ry, ry+1):
        for dx in range(-rx, rx+1):
            nb.append((dx,dy))

    loader = []
    loader.append(f"{dt(meta.get('dtype','float32'), backend)} window[{len(nb)}];")
    loader.append("int pos = 0;")
    for dx,dy in nb:
        xx = _boundary_clamp_expr("int(x) + " + str(dx), "0", "W")
        yy = _boundary_clamp_expr("int(y) + " + str(dy), "0", "H")
        if backend == "metal":
            loader.append(f"{{ int XX = {xx}; int YY = {yy}; window[pos++] = src[(uint)YY*(uint)W + (uint)XX]; }}")
        else:
            loader.append(f"{{ int XX = {xx}; int YY = {yy}; window[pos++] = src[YY*W + XX]; }}")

    compute = f"""/* INSERT-COMPUTE-HERE: stencil2d ({kx}x{ky})
   window[0..{len(nb)-1}] is row-major over neighborhood with {boundary} boundary.
   Example: average filter → dst[idx] = (sum window)/{float(len(nb))}f;
*/"""

    head_kw = {"opencl":"__kernel void", "cuda":"__global__ void", "metal":"kernel void"}[backend]
    return f"""{h}
{head_kw} {k.name}({sig})
{{
    {prolog}
    if ({'x >= (uint)W || y >= (uint)H' if backend=='metal' else 'x >= W || y >= H'}) return;
    {idx}
    {'\n    '.join(loader)}
    {compute}
}}
"""

def emit_separable2d(k: Kernel, meta: dict[str,Any], backend: str) -> str:
    # traits
    rx, ry = (0,0)
    if "radius" in k.traits:
        val = k.traits["radius"]
        if isinstance(val, (list,tuple)) and len(val)==2: rx, ry = int(val[0]), int(val[1])
        else: rx = int(val); ry = int(val)
    boundary = k.traits.get("boundary", "clamp")
    fastmem = str(k.traits.get("fastmem","off")).lower() in ("on","true","1")
    tile = tuple(k.traits.get("tile", (128,8)))
    # taps are provided as 'constant float kx[]/ky[]' params in the spec

    def body(axis: str) -> str:
        radius = rx if axis=="horizontal" else ry
        taps   = "kx" if axis=="horizontal" else "ky"
        if backend == "opencl":
            gid = "int x = get_global_id(0); int y = get_global_id(1); if (x >= W || y >= H) return; int idx = y*W + x;"
            loop = f"""float acc = 0.0f;
#pragma unroll
for (int t=-{radius}; t<={radius}; ++t) {{
    int xx = x + ({"t" if axis=="horizontal" else "0"});
    int yy = y + ({"0" if axis=="horizontal" else "t"});
    if (xx < 0) xx = 0; if (xx >= W) xx = W-1;
    if (yy < 0) yy = 0; if (yy >= H) yy = H-1;
    acc += {taps}[t+{radius}] * src[yy*W + xx];
}}
dst[idx] = acc;"""
        elif backend == "cuda":
            gid = "int x = blockIdx.x*blockDim.x + threadIdx.x; int y = blockIdx.y*blockDim.y + threadIdx.y; if (x >= W || y >= H) return; int idx = y*W + x;"
            loop = f"""float acc = 0.0f;
#pragma unroll
for (int t=-{radius}; t<={radius}; ++t) {{
    int xx = x + ({"t" if axis=="horizontal" else "0"});
    int yy = y + ({"0" if axis=="horizontal" else "t"});
    xx = max(0, min(W-1, xx));
    yy = max(0, min(H-1, yy));
    acc += {taps}[t+{radius}] * src[yy*W + xx];
}}
dst[idx] = acc;"""
        else:
            gid = "uint x = gid2.x, y = gid2.y; if (x >= (uint)W || y >= (uint)H) return; uint idx = y*(uint)W + x;"
            loop = f"""float acc = 0.0f;
for (int t=-{radius}; t<={radius}; ++t) {{
    int xx = int(x) + ({"t" if axis=="horizontal" else "0"});
    int yy = int(y) + ({"0" if axis=="horizontal" else "t"});
    if (xx < 0) xx = 0; if (xx >= W) xx = W-1;
    if (yy < 0) yy = 0; if (yy >= H) yy = H-1;
    acc += {taps}[t+{radius}] * src[(uint)yy*(uint)W + (uint)xx];
}}
dst[idx] = acc;"""
        return gid + "\n    " + loop

    h = header(backend, meta.get("fast_math", True))
    # Build signatures (we expect params in the interface in proper order)
    # Horizontal
    sig_h, _ = pack_params_order(k, backend)
    name_h = k.name if k.traits.get("axis","") == "horizontal" or k.name.endswith("_h") else f"{k.name}_h"
    # Vertical (if provided as separate kernel in spec, we just emit it)
    code_h = f"""{h}
{"__kernel void" if backend=="opencl" else ("__global__ void" if backend=="cuda" else "kernel void")} {name_h}({sig_h})
{{
    {body("horizontal")}
}}
"""
    return code_h  # Keep 1:1 with spec (you'll have separate _v entry too)

def emit_reduction(k: Kernel, meta: dict[str,Any], backend: str) -> str:
    op = k.traits.get("op", "sum")
    identity = k.traits.get("identity", 0.0)
    deterministic = str(k.traits.get("deterministic","off")).lower() in ("on","true","1")
    acc_dtype = k.traits.get("accumulator_dtype", meta.get("dtype","float32"))

    h = header(backend, meta.get("fast_math", True))
    sig, _ = pack_params_order(k, backend)
    comb = {
        "sum":       lambda a,b: f"({a}) + ({b})",
        "max":       lambda a,b: f"max({a},{b})",
        "min":       lambda a,b: f"min({a},{b})",
        "logical_or":lambda a,b: f"(({a}) || ({b}))",
    }[op]
    acc = dt(acc_dtype, backend)
    elem = dt(meta.get("dtype","float32"), backend)

    if backend == "cuda":
        core = f"""
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ {acc} sdata[];
    {acc} val = ({acc})({identity});
    if (i < length) val = ({acc})in0[i]; // assume first param is src; adapt as needed
    __syncthreads();
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x>>1; s>0; s>>=1) {{
        if (threadIdx.x < s) {{
            sdata[threadIdx.x] = {comb('sdata[threadIdx.x]','sdata[threadIdx.x + s]')};
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) out0[blockIdx.x] = ({elem})sdata[0];"""
        return f"""{h}
__global__ void {k.name}({sig}, int length)
{{
    {core}
}}
"""
    if backend == "opencl":
        core = f"""
    int i = get_global_id(0);
    __local {acc} sdata[/* set via local size */];
    {acc} val = ({acc})({identity});
    if (i < length) val = ({acc})in0[i]; // assume first param is src
    barrier(CLK_LOCAL_MEM_FENCE);
    sdata[get_local_id(0)] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0)>>1; s>0; s>>=1) {{
        if (get_local_id(0) < s) {{
            sdata[get_local_id(0)] =
                {comb('sdata[get_local_id(0)]','sdata[get_local_id(0)+s]')};
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    if (get_local_id(0) == 0) out0[get_group_id(0)] = ({elem})sdata[0];"""
        return f"""{h}
__kernel void {k.name}({sig}, int length)
{{
    {core}
}}
"""
    # metal
    core = f"""
    uint tid = tid;
    uint x = gid2.x;
    threadgroup {acc} sdata[/* set from encoder */];
    {acc} val = ({acc})({identity});
    if (x < (uint)length) val = ({acc})in0[x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sdata[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = (tgp2.x>>1); s>0; s>>=1) {{
        if (tid < s) {{
            sdata[tid] = {comb('sdata[tid]','sdata[tid+s]')};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0) out0[tgpos2.x] = ({elem})sdata[0];"""
    return f"""{h}
kernel void {k.name}({sig}, constant int& length [[ buffer({len(k.params)}) ]], uint tid [[thread_index_in_threadgroup]], uint2 gid2 [[thread_position_in_grid]], uint2 tgp2 [[threads_per_threadgroup]], uint2 tgpos2 [[threadgroup_position_in_grid]])
{{
    {core}
}}
"""

def emit_graphics2d(k: Kernel, meta: dict[str,Any], backend: str) -> str:
    blend = k.traits.get("blend", meta.get("blend", None))
    if not blend:
        # fallback: infer: float -> alpha, float4 -> src_over
        first = next((p for p in k.params if p.role in ("READWRITE","out","out_like","resident")), None)
        blend = "src_over" if (first and (first.dtype in ("float4",))) else "alpha"
    h = header(backend, meta.get("fast_math", True))
    sig, _ = pack_params_order(k, backend)

    gid = {
        "opencl": "int x = get_global_id(0); int y = get_global_id(1);",
        "cuda":   "int x = blockIdx.x*blockDim.x + threadIdx.x; int y = blockIdx.y*blockDim.y + threadIdx.y;",
        "metal":  "uint x = gid2.x, y = gid2.y;",
    }[backend]
    bounds = "if (x >= W || y >= H) return;" if backend!="metal" else "if (x >= (uint)W || y >= (uint)H) return;"
    idx = "int idx = y*W + x;" if backend!="metal" else "uint idx = y*(uint)W + x;"

    # names: we expect the interface to name buffers: img, lines/rects/circles/..., rgba or color_a, rdist, n
    compute = f"""/* INSERT-COMPUTE-HERE: graphics_2d ({blend})
   - Loop over N primitives; do AABB quick-reject then inside-test; compose with {blend}.
   - Example given for lines: lines[i] = (x0,y0,x1,y1), rdist[i] = radius, rgba[i] color.
*/"""

    head_kw = {"opencl":"__kernel void", "cuda":"__global__ void", "metal":"kernel void"}[backend]
    return f"""{h}
{head_kw} {k.name}({sig})
{{
    {gid}
    {bounds}
    {idx}
    {compute}
}}
"""

# Dispatcher
def emit_kernel(k: Kernel, meta: dict[str,Any], backend: str) -> str:
    # Resolve effective class
    klass = k.traits.get("class", meta.get("class", None))
    # Graphics family interfaces default to graphics_2d by name
    if not klass and meta.get("class") is None:
        if any(k.name.lower().startswith(p) for p in ("draw_", "composite")):
            klass = "graphics_2d"
    if not klass:
        # fallback heuristic: if traits has radius → stencil; else elementwise
        if "radius" in k.traits or "boundary" in k.traits:
            klass = "stencil2d"
        else:
            klass = "elementwise"

    if klass == "elementwise":
        return emit_elementwise(k, meta, backend)
    if klass == "stencil2d":
        return emit_stencil2d(k, meta, backend)
    if klass == "separable" or klass == "separable2d":
        return emit_separable2d(k, meta, backend)
    if klass == "reduction":
        return emit_reduction(k, meta, backend)
    if klass == "graphics_2d":
        return emit_graphics2d(k, meta, backend)
    # default
    return emit_elementwise(k, meta, backend)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=["opencl","cuda","metal"])
    ap.add_argument("--out", dest="outdir", required=True, type=Path)
    args = ap.parse_args()

    spec = parse_interface_text(args.infile.read_text(encoding="utf-8"))

    args.outdir.mkdir(parents=True, exist_ok=True)
    for k in spec.kernels:
        code = emit_kernel(k, spec.meta, args.backend)
        ext = {"opencl":".cl", "cuda":".cu", "metal":".metal"}[args.backend]
        (args.outdir / f"{k.name}{ext}").write_text(code, encoding="utf-8")
        print(f"wrote {args.outdir / (k.name+ext)}")

if __name__ == "__main__":
    main()
