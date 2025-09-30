/* Separable convolution kernels (Metal), names and constants determined at compile time */

<%include file="constants.mako" />
<%
import os
_use_tiled = os.environ.get('YAPOCIS_METAL_TILED', '0').lower() in ('1','true','yes')
%>
% if _use_tiled:
#define USE_TILED 1
% else:
#define USE_TILED 0
% endif

// Ensure deterministic arithmetic (avoid FMA contraction) to better match OpenCL
#pragma clang fp contract(off)

% for name, conv in convs:
<% margin = len(conv)//2 %>
kernel void ${name}(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint2 gxy [[ thread_position_in_grid ]])
{
    if (gxy.x >= (uint)width || gxy.y >= (uint)height) return;
    size_t gx = (size_t)gxy.x;
    size_t gy = (size_t)gxy.y;
    size_t i = gy * (size_t)width + gx;  // row-major index like OpenCL
    float sum = 0.0f;
    if (vertical) {
       // Horizontal convolution: step by +/-1 within the row
       if (${margin} <= gx && gx < (size_t)(width - ${margin})) {
           % for j in range(len(conv)):
           sum = sum + a[i+(${j-margin})] * ${conv[j]};
           % endfor
       }
    } else {
       // Vertical convolution: step by +/-width across rows
       if (${margin} <= gy && gy < (size_t)(height - ${margin})) {
           % for j in range(len(conv)):
           sum = sum + a[i+(${j-margin}) * (size_t)width] * ${conv[j]};
           % endfor
       }
    }
    ret[i] = sum;
}

kernel void ${name}_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint2 gxy [[ thread_position_in_grid ]],
                      ushort2 lxy [[ thread_position_in_threadgroup ]],
                      ushort2 lsz [[ threads_per_threadgroup ]])
{
    if (gxy.x >= (uint)width || gxy.y >= (uint)height) return;
    const int x = (int)gxy.x;
    const int y = (int)gxy.y;
    size_t i = (size_t)y * (size_t)width + (size_t)x;  // row-major index
    float sum = 0.0f;
    // Horizontal convolution (within row): require margin on X
    int inside = (${margin} <= x && x < (width - ${margin}));
    % if margin <= 16:
#if USE_TILED
    // Tiled path with horizontal halo (matches OCL semantics)
    const ushort tx = lsz.x;
    const ushort ty = lsz.y;
    const ushort lx = lxy.x;
    const ushort ly = lxy.y;
    const ushort stride = (ushort)(tx + (2 * ${margin}));
    threadgroup float tile[((${margin}*2) + 64) * 64]; // row-major, 64x64 cap
    const uint trow = (uint)ly; // y within tile
    const uint tcol = (uint)(lx + ${margin});
    const uint tidx = trow * stride + tcol;
    // Center sample
    tile[tidx] = a[i];
    // Halos loaded by edge threads with bounds checks
    % for k in range(1, margin+1):
    if (lx == 0 && x >= ${k}) { tile[tidx - ${k}] = a[i - ${k}]; }
    if (lx == tx - 1 && (x + ${k}) < width) { tile[tidx + ${k}] = a[i + ${k}]; }
    % endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (inside) {
        % for j in range(len(conv)):
        sum += tile[tidx + (${j}-${margin})] * ${conv[j]};
        % endfor
    }
    #else
    // Direct-read path
    if (inside) {
        % for j in range(len(conv)):
        sum += a[i + (${j}-${margin})] * ${conv[j]};
        % endfor
    }
    #endif
    % else:
    // Direct-read path (no tiling for large kernels)
    if (inside) {
        % for j in range(len(conv)):
        sum += a[i + (${j}-${margin})] * ${conv[j]};
        % endfor
    }
    % endif
    ret[i] = sum;
}

kernel void ${name}_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint2 gxy [[ thread_position_in_grid ]],
                      ushort2 lxy [[ thread_position_in_threadgroup ]],
                      ushort2 lsz [[ threads_per_threadgroup ]])
{
    if (gxy.x >= (uint)width || gxy.y >= (uint)height) return;
    const int x = (int)gxy.x;
    const int y = (int)gxy.y;
    size_t i = (size_t)y * (size_t)width + (size_t)x;  // row-major index
    float sum = 0.0f;
    // Vertical convolution (across rows): require margin on Y
    int inside = (${margin} <= y && y < (height - ${margin}));
    % if margin <= 16:
#if USE_TILED
    // Tiled path with vertical halo
    const ushort tx = lsz.x;
    const ushort ty = lsz.y;
    const ushort lx = lxy.x;
    const ushort ly = lxy.y;
    const ushort stride = (ushort)(tx);
    threadgroup float tile[64 * ((${margin}*2) + 64)]; // height-major tile
    const uint trow = (uint)(ly + ${margin});
    const uint tcol = (uint)lx;
    const uint tidx = trow * stride + tcol;
    // Center
    tile[tidx] = a[i];
    // Halos loaded by edge threads with bounds checks
    % for k in range(1, margin+1):
    if (ly == 0 && y >= ${k}) { tile[(trow - ${k}) * stride + tcol] = a[i - ${k} * width]; }
    if (ly == ty - 1 && (y + ${k}) < height) { tile[(trow + ${k}) * stride + tcol] = a[i + ${k} * width]; }
    % endfor
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (inside) {
        % for j in range(len(conv)):
        sum += tile[(trow + (${j}-${margin})) * stride + tcol] * ${conv[j]};
        % endfor
    }
    #else
    // Direct-read path
    if (inside) {
        % for j in range(len(conv)):
        sum += a[i + (${j}-${margin}) * width] * ${conv[j]};
        % endfor
    }
    #endif
    % else:
    // Direct-read path (no tiling for large kernels)
    if (inside) {
        % for j in range(len(conv)):
        sum += a[i + (${j}-${margin}) * width] * ${conv[j]};
        % endfor
    }
    % endif
    ret[i] = sum;
}
% endfor
