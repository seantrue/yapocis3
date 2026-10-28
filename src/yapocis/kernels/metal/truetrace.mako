/* Metal versions of SDFPath kernels. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

inline float4 make_seed(float x, float y) {
    return float4(x, y, 0.0f, 0.0f);
}

// A function, not a lambda: Metal compiles at the language version of the SDK the host Python
// was built with, and the SDK 14.2 one (uv's CPython 3.13.0) rejects lambdas, while SDK 15.2 and
// 26.1 accept them; the same source compiled under one Python and failed under another.
inline float tt_lerp(float t, float a, float b) {
    return a + t*(b-a);
}

kernel void init_seed_from_mask(constant int &width  [[ buffer(0) ]],
                                constant int &height [[ buffer(1) ]],
                                const device int *mask [[ buffer(2) ]],
                                device float4 *seed [[ buffer(3) ]],
                                device float *dist [[ buffer(4) ]],
                                uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    int m = mask[i];
    seed[i] = (m != 0) ? make_seed((float)x, (float)y) : make_seed(-1.0f, -1.0f);
    dist[i] = (m != 0) ? 0.0f : 1e30f;
}

kernel void jfa_step(constant int &width  [[ buffer(0) ]],
                     constant int &height [[ buffer(1) ]],
                     constant int &jump   [[ buffer(2) ]],
                     const device float4 *seed_in [[ buffer(3) ]],
                     device float *dist_inout      [[ buffer(4) ]],
                     device float4 *seed_out       [[ buffer(5) ]],
                     uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);

    float4 best_seed = seed_in[i];
    float  best_dist = dist_inout[i];

    const int dirs[8][2] = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    for (int k=0;k<8;++k) {
        int nx = x + dirs[k][0]*jump;
        int ny = y + dirs[k][1]*jump;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        size_t nidx = ADDRESS(nx, ny);
        float4 nseed = seed_in[nidx];
        if (nseed.x < 0.0f) continue;
        float dx = nseed.x - (float)x;
        float dy = nseed.y - (float)y;
        float d2 = dx*dx + dy*dy;
        if (d2 < best_dist*best_dist) {
            best_dist = metal::sqrt(d2);
            best_seed = nseed;
        }
    }
    seed_out[i] = best_seed;
    dist_inout[i] = best_dist;
}

kernel void marching_squares(constant int &width  [[ buffer(0) ]],
                             constant int &height [[ buffer(1) ]],
                             const device float *dist [[ buffer(2) ]],
                             constant float &iso      [[ buffer(3) ]],
                             device atomic_uint &seg_counter [[ buffer(4) ]],
                             device float *segments_out [[ buffer(5) ]],
                             uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    if (x >= width-1 || y >= height-1) return;

    size_t idx00 = i;
    size_t idx10 = ADDRESS(x+1,y);
    size_t idx01 = ADDRESS(x,y+1);
    size_t idx11 = ADDRESS(x+1,y+1);

    float v00 = dist[idx00] - iso;
    float v10 = dist[idx10] - iso;
    float v11 = dist[idx11] - iso;
    float v01 = dist[idx01] - iso;

    int c = 0;
    if (v00 > 0) c |= 1;
    if (v10 > 0) c |= 2;
    if (v11 > 0) c |= 4;
    if (v01 > 0) c |= 8;
    if (c == 0 || c == 15) return;

    float2 p[4];
    float t0 = v00/(v00 - v10 + 1e-20f); p[0] = float2(tt_lerp(t0, (float)x, (float)(x+1)), (float)y);
    float t1 = v10/(v10 - v11 + 1e-20f); p[1] = float2((float)(x+1), tt_lerp(t1, (float)y, (float)(y+1)));
    float t2 = v01/(v01 - v11 + 1e-20f); p[2] = float2(tt_lerp(t2, (float)x, (float)(x+1)), (float)(y+1));
    float t3 = v00/(v00 - v01 + 1e-20f); p[3] = float2((float)x,      tt_lerp(t3, (float)y, (float)(y+1)));

    int pairs[2][2]; int nseg = 0;
    switch(c) {
        case 1: case 14: pairs[0][0]=3; pairs[0][1]=0; nseg=1; break;
        case 2: case 13: pairs[0][0]=0; pairs[0][1]=1; nseg=1; break;
        case 3: case 12: pairs[0][0]=3; pairs[0][1]=1; nseg=1; break;
        case 4: case 11: pairs[0][0]=1; pairs[0][1]=2; nseg=1; break;
        case 6: case 9:  pairs[0][0]=0; pairs[0][1]=2; nseg=1; break;
        case 7: case 8:  pairs[0][0]=3; pairs[0][1]=2; nseg=1; break;
        case 5: case 10: {
            float d = v00*v11 - v10*v01;
            if (c == 5) {
                if (d > 0) { pairs[0][0]=3; pairs[0][1]=0; pairs[1][0]=1; pairs[1][1]=2; }
                else       { pairs[0][0]=0; pairs[0][1]=1; pairs[1][0]=2; pairs[1][1]=3; }
            } else {
                if (d > 0) { pairs[0][0]=0; pairs[0][1]=1; pairs[1][0]=2; pairs[1][1]=3; }
                else       { pairs[0][0]=3; pairs[0][1]=0; pairs[1][0]=1; pairs[1][1]=2; }
            }
            nseg=2; break; }
        default: return;
    }

    for (int i=0;i<nseg;++i) {
        uint slot = atomic_fetch_add_explicit(&seg_counter, (uint)1, memory_order_relaxed);
        const uint base = slot * 4u;
        float2 a = p[pairs[i][0]];
        float2 b = p[pairs[i][1]];
        segments_out[base+0] = a.x; segments_out[base+1] = a.y;
        segments_out[base+2] = b.x; segments_out[base+3] = b.y;
    }
}
