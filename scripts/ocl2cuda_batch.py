#!/usr/bin/env python3
import argparse, re
from pathlib import Path
SHIM = r"""/* Auto-translated from OpenCL to CUDA (Mako kept intact). */
#ifndef YAPOCIS_CUDA_SHIMS_H
#define YAPOCIS_CUDA_SHIMS_H
#define __kernel extern "C" __global__
#define __local  __shared__
#define barrier(x) __syncthreads()
__device__ __forceinline__ int get_global_id_x(){return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ __forceinline__ int get_global_id_y(){return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ __forceinline__ int get_global_id_z(){return blockIdx.z*blockDim.z+threadIdx.z;}
#define get_global_id(N) ( (N)==0 ? get_global_id_x() : ( (N)==1 ? get_global_id_y() : get_global_id_z() ) )
__device__ __forceinline__ int clampi(int v,int lo,int hi){return max(lo,min(hi,v));}
__device__ __forceinline__ float clampf(float v,float lo,float hi){return fminf(hi,fmaxf(lo,v));}
#define clamp(V,L,H) _Generic((V), float:clampf, double:clampf, default:clampi)((V),(L),(H))
template<typename V> __device__ __forceinline__ V  vload(int off, const void* base){return reinterpret_cast<const V*>(base)[off];}
template<typename V> __device__ __forceinline__ void vstore(V v, int off, void* base){reinterpret_cast<V*>(base)[off]=v;}
#define vload4(OFF,PTR)       vload<float4>((OFF),(const void*)(PTR))
#define vstore4(VAL,OFF,PTR)  vstore<float4>((VAL),(OFF),(void*)(PTR))
#endif
"""
SUBS=[(r"\b__global\s+",""),(r"(\([^)]*)\bconstant\b\s+([^)]*\))",r"\1\2"),(r"(?m)^(\s*)constant\s+",r"\1__constant__ "),(r"\bget_global_id\s*\(\s*0\s*\)","get_global_id(0)"),(r"\bget_global_id\s*\(\s*1\s*\)","get_global_id(1)"),(r"\bget_global_id\s*\(\s*2\s*\)","get_global_id(2)"),(r"\bbarrier\s*\(\s*CLK_LOCAL_MEM_FENCE\s*\)","barrier(CLK_LOCAL_MEM_FENCE)")]
def translate_text(text:str)->str:
    out=text
    for pat,rep in SUBS: out=re.sub(pat,rep,out)
    if "YAPOCIS_CUDA_SHIMS_H" not in out: out=SHIM+"\n"+out
    return out
def main():
    import sys
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="yapocis/kernels")
    ap.add_argument("--dst",default="yapocis/cukernels")
    args=ap.parse_args()
    src_dir=Path(args.src); dst_dir=Path(args.dst)
    if not src_dir.exists(): print(f"Source not found: {src_dir}", file=sys.stderr); sys.exit(1)
    dst_dir.mkdir(parents=True, exist_ok=True)
    count=0
    for src in sorted(src_dir.glob("*.mako")):
        text=src.read_text(); cuda_text=translate_text(text)
        out_path=dst_dir/f"cuda_{src.name}"; out_path.write_text(cuda_text)
        print("✓", out_path); count+=1
    print(f"Done. Converted {count} file(s) -> {dst_dir}")
if __name__=="__main__": main()
