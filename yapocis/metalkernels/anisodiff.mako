/* Perona-Malik filter (Metal) */

<%include file="constants.mako" />
<%include file="helpers.mako" />

// Tiled stencil settings for shared-memory caching
#ifndef TILE_MAX_X
#define TILE_MAX_X 32
#endif
#ifndef TILE_MAX_Y
#define TILE_MAX_Y 32
#endif

% for planes in ["",2,3,4]:
kernel void filterImage${planes}(constant int &width [[ buffer(0) ]],
                                 constant int &height [[ buffer(1) ]],
                                 const device FLOAT${planes} *input [[ buffer(2) ]],
                                 constant float &scale [[ buffer(3) ]],
                                 constant float &step_size [[ buffer(4) ]],
                                 device FLOAT${planes} *output [[ buffer(5) ]],
                                 uint2 gxy   [[ thread_position_in_grid ]])
{
    const int x = (int)gxy.x;
    const int y = (int)gxy.y;
    if (x >= width || y >= height) {
        return;
    }
    const size_t i = (size_t)x * (size_t)height + (size_t)y;
    FLOAT${planes} central = input[i];
    // Mirror OpenCL behavior: copy borders through unchanged
    if (x == 0 || y == 0 || x == (width - 1) || y == (height - 1)) {
        output[i] = central;
        return;
    }

    const FLOAT${planes} invscale = (FLOAT${planes})(1.0f / scale);
    FLOAT${planes} accumulator = (FLOAT${planes})(0.0f);
    // Direct neighbor fetches
    const FLOAT${planes} n = input[i + 1];
    const FLOAT${planes} s = input[i - 1];
    const FLOAT${planes} e = input[i - height];
    const FLOAT${planes} w = input[i + height];

    FLOAT${planes} di = n - central;
    FLOAT${planes} cf = di * invscale;
    cf = ONE${planes} / (ONE${planes} + (cf * cf));
    accumulator += di * cf;

    di = s - central;
    cf = di * invscale;
    cf = ONE${planes} / (ONE${planes} + (cf * cf));
    accumulator += di * cf;

    di = e - central;
    cf = di * invscale;
    cf = ONE${planes} / (ONE${planes} + (cf * cf));
    accumulator += di * cf;

    di = w - central;
    cf = di * invscale;
    cf = ONE${planes} / (ONE${planes} + (cf * cf));
    accumulator += di * cf;

    output[i] = central + accumulator * (FLOAT${planes})(step_size);
}
% endfor
