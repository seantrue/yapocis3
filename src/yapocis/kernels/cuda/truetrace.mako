/* SDFPath kernels: seed init from mask, JFA distance, marching squares */
<%include file="constants.mako" />
<%include file="helpers.mako" />

/* truetrace's wrapper passes (W, H) arrays and reads x as the slow axis, as the
   OpenCL and Metal helpers address them; CUDA's shared helpers address x as the
   fast axis (row-major), which transposed every traced shape on CUDA. These
   kernels keep their own addressing. */
#define TT_ADDRESS(xx,yy) (((xx)*height)+(yy))
#define TT_X(addr) ((addr)/height)
#define TT_Y(addr) ((addr)%height)

inline __device__ float4 make_seed(float x, float y) {
    return make_float4(x, y, 0.0f, 0.0f);
}

extern "C" __global__ void init_seed_from_mask(
    const int width,
    const int height,
    const int *mask, /* 0 or 1 per pixel */
    float4 *seed,     /* store seed as (x,y,_,_) */
    float *dist       /* Euclidean distance (pixels) */
) {
    const int idx = get_global_id(0);
    const int x = TT_X(idx);
    const int y = TT_Y(idx);
    if (x >= width || y >= height) return;
    if (mask[idx] != 0) {
        seed[idx] = make_seed((float)x, (float)y);
        dist[idx] = 0.0f;
    } else {
        seed[idx] = make_seed(-1.0f, -1.0f);
        dist[idx] = 1e30f;
    }
}

extern "C" __global__ void jfa_step(
    const int width,
    const int height,
    const int jump,
    const float4 *seed_in,
    float *dist_inout,
    float4 *seed_out
) {
    const int idx = get_global_id(0);
    const int x = TT_X(idx);
    const int y = TT_Y(idx);
    if (x >= width || y >= height) return;

    float4 best_seed = seed_in[idx];
    float  best_dist = dist_inout[idx];

    const int dirs[8][2] = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    for (int k=0;k<8;++k) {
        int nx = x + dirs[k][0]*jump;
        int ny = y + dirs[k][1]*jump;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const int nidx = TT_ADDRESS(nx,ny);
        float4 nseed = seed_in[nidx];
        if (nseed.x < 0.0f) continue; // invalid seed
        float dx = nseed.x - (float)x;
        float dy = nseed.y - (float)y;
        float d2 = dx*dx + dy*dy;
        if (d2 < best_dist*best_dist) {
            best_dist = sqrt(d2);
            best_seed = nseed;
        }
    }
    seed_out[idx] = best_seed;
    dist_inout[idx] = best_dist;
}

extern "C" __global__ void marching_squares(
    const int width,
    const int height,
    const float *dist,
    const float iso,
    unsigned int *seg_counter,
    float *segments_out
) {
    const int idx = get_global_id(0);
    const int x = TT_X(idx);
    const int y = TT_Y(idx);
    if (x >= width-1 || y >= height-1) return;

    const int idx00 = idx;;
    const int idx10 = TT_ADDRESS(x+1, y);
    const int idx01 = TT_ADDRESS(x,y+1);
    const int idx11 = TT_ADDRESS(x+1,y+1);

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

    #define LERP(t, a, b) ((a) + (t)*((b)-(a)))
    float2 p[4];
    float t0 = v00/(v00 - v10 + 1e-20f); p[0].x = LERP(t0, (float)x, (float)(x+1)); p[0].y = (float)y;
    float t1 = v10/(v10 - v11 + 1e-20f); p[1].x = (float)(x+1); p[1].y = LERP(t1, (float)y, (float)(y+1));
    float t2 = v01/(v01 - v11 + 1e-20f); p[2].x = LERP(t2, (float)x, (float)(x+1)); p[2].y = (float)(y+1);
    float t3 = v00/(v00 - v01 + 1e-20f); p[3].x = (float)x;      p[3].y = LERP(t3, (float)y, (float)(y+1));

    int pairs[2][2]; int nseg = 0;
    switch(c) {
        case 1: case 14: pairs[0][0]=3; pairs[0][1]=0; nseg=1; break;
        case 2: case 13: pairs[0][0]=0; pairs[0][1]=1; nseg=1; break;
        case 3: case 12: pairs[0][0]=3; pairs[0][1]=1; nseg=1; break;
        case 4: case 11: pairs[0][0]=1; pairs[0][1]=2; nseg=1; break;
        case 6: case 9:  pairs[0][0]=0; pairs[0][1]=2; nseg=1; break;
        case 7: case 8:  pairs[0][0]=3; pairs[0][1]=2; nseg=1; break;
        case 5: case 10: {
            float d = v00*v11 - v10*v01; // asymptotic decider
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
        unsigned int slot = atomicAdd(seg_counter, 1);
        const unsigned int base = slot * 4u;
        float2 a = p[pairs[i][0]];
        float2 b = p[pairs[i][1]];
        segments_out[base+0] = a.x; segments_out[base+1] = a.y;
        segments_out[base+2] = b.x; segments_out[base+3] = b.y;
    }
}
