/* Metal kernels to compute direction and magnitude of gradient of an image. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Compute gradient for images with 1,2,3,4 channels per pixel. */

%for planes in ["",2,3,4]:
kernel void gradient${planes}(constant int &width [[ buffer(0) ]],
                              constant int &height [[ buffer(1) ]],
                              const device FLOAT${planes}* input [[ buffer(2) ]],
                              constant int &reach [[ buffer(3) ]],
                              device FLOAT${planes}* grad [[ buffer(4) ]],
                              device FLOAT${planes}* theta [[ buffer(5) ]],
                              uint2 gxy [[ thread_position_in_grid ]],
                              ushort2 lxy [[ thread_position_in_threadgroup ]],
                              ushort2 lsz [[ threads_per_threadgroup ]])
{
    const int x = (int)gxy.x;
    const int y = (int)gxy.y;
    if (x >= width || y >= height) {
        return;
    }
    const size_t i = (size_t)x * (size_t)height + (size_t)y;
    FLOAT${planes} gr = ZERO${planes};
    FLOAT${planes} th = ZERO${planes};
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        // For reach==1, use shared-memory tile with 1-pixel halo for better locality
        if (reach == 1) {
            const ushort tx = (ushort)min((int)lsz.x, 32);
            const ushort ty = (ushort)min((int)lsz.y, 32);
            const ushort lx = (ushort)min((int)lxy.x, 31);
            const ushort ly = (ushort)min((int)lxy.y, 31);
            const ushort stride = (ushort)(32 + 2);
            threadgroup FLOAT${planes} tile[(32 + 2) * (32 + 2)];
            const ushort tidx = (ushort)((lx + 1) * stride + (ly + 1));
            tile[tidx] = input[i];
            const size_t iN = i + 1;
            const size_t iS = i - 1;
            const size_t iE = i - height;
            const size_t iW = i + height;
            if (lx == 0)         tile[tidx - stride] = input[iE];
            if (lx == tx - 1)    tile[tidx + stride] = input[iW];
            if (ly == 0)         tile[tidx - 1]      = input[iS];
            if (ly == ty - 1)    tile[tidx + 1]      = input[iN];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const FLOAT${planes} dx = tile[tidx - stride] - tile[tidx + stride];
            const FLOAT${planes} dy = tile[tidx + 1] - tile[tidx - 1];
            gr = sqrt(dx*dx+dy*dy);
            th = atan2pi(dy, dx);
        } else {
            FLOAT${planes} dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
            FLOAT${planes} dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
            gr = sqrt(dx*dx+dy*dy);
            th = atan2pi(dy, dx);
        }
    }
    // Replace NaNs component-wise with zero using Metal's select(a,b,cond)
    // select(a, b, cond) returns cond ? b : a, works for scalar and vector types
    grad[i] = select(gr, ZERO${planes}, isnan(gr));
    theta[i] = select(th, ZERO${planes}, isnan(th));
}
%endfor
