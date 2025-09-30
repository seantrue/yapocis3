/* Metal kernels to find zero-crossings in an image. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

// Define a deterministic sign function to match OpenCL semantics exactly
static inline float fsign(float x) {
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

/* Find zero-crossings for pixels of 1-4 channels. */
%for planes in [1,2,3,4]:
#define ADDRESS${planes}(xx,yy) (ADDRESS(xx,yy))*${planes}

kernel void zcs${planes}(constant int &width [[ buffer(0) ]],
                         constant int &height [[ buffer(1) ]],
                         constant float &threshold [[ buffer(2) ]],
                         const device float* input [[ buffer(3) ]],
                         device float* zcs [[ buffer(4) ]],
                         uint2 gxy [[ thread_position_in_grid ]])
{
    // Original pixel position and base index for writes
    const int ox = (int)gxy.x;
    const int oy = (int)gxy.y;
    if (ox >= width || oy >= height) return;
    size_t i = ADDRESS${planes}(ox, oy);
    // Clamp sampling coordinates to interior to avoid OOB
    int x = ox < 1 ? 1 : (ox > (width - 2) ? (width - 2) : ox);
    int y = oy < 1 ? 1 : (oy > (height - 2) ? (height - 2) : oy);
    float zclt;
    // Slightly bias the compare to match OpenCL rounding on borderline values.
    // Treat near-threshold deltas as not crossing to avoid tiny rounding flips.
    const float eps = 1e-10f;
    const float thr = threshold + eps;
    float v, below, left, above, right;
    float delta;
    %for plane in range(planes):
    v = input[i];
    above = input[ADDRESS${planes}(x,y+1)];
    below = input[ADDRESS${planes}(x,y-1)];
    left = input[ADDRESS${planes}(x-1,y)];
    right = input[ADDRESS${planes}(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = fsign(above);
    below = fsign(below);
    left = fsign(left);
    right = fsign(right);
    zclt = (delta > thr) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[i] = zclt;
    i += 1;
    %endfor
}
%endfor

kernel void sub(device float* input1 [[ buffer(0) ]],
                device float* input2 [[ buffer(1) ]],
                device float *output [[ buffer(2) ]],
                uint gid [[ thread_position_in_grid ]])
{
    output[gid] = input1[gid] - input2[gid];
}
