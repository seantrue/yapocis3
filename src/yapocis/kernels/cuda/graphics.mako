<%include file="constants.mako"/>
#define DXDY2
<%include file="helpers.mako"/>

/* Kernels that implement graphics primitives against a float array.
 * The float array is assumed to be a 2D array of floats, with the
 * first dimension being the width of the image, and the second
 * dimension being the height of the image.
 *
 * The kernels are:
 *   lines: draw lines
 *   rects: draw rectangles
 *   circles: draw circles
 *   triangles: draw triangles
 *   ellipses: draw ellipses
 *   composite: composite multiple sprites
 *
 * Each kernel takes a float array, and a list of primitives to draw.
 * The list of primitives is a list of coordinates, and a list of
 * colors.  The coordinates are in the same coordinate system as the
 * float array, and the colors are in the range [0,1].
 */

extern "C" __kernel
void lines(int width, int height,  READWRITE float *a, int nelements, READONLY float2 *coords, READONLY float *rdist, READONLY float *aa_widths, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float2 point = make_float2((float)x, (float)y);
    for (int ielement = 0; ielement < nelements; ielement++) {
        float threshold = ((*rdist++ * .5) + .5);
        float aa_width = *aa_widths++;
        float threshold2 = threshold * threshold;
        float2 linestart = *coords++;
        float2 lineend = *coords++;
        float2 value_alpha = *colors++;
        float value = value_alpha.x;
        float alpha = value_alpha.y;
        float linemag2 = dxdy2(lineend, linestart);
        float2 delta = lineend - linestart;
        float2 u2 = (point - linestart) * (delta);
        float u = (u2.x + u2.y) / linemag2;
        float2 intersection = linestart + (u * delta);
        float dist2 = dxdy2(point,intersection);

        if (u >= 0.0f && u <= 1.0f) {
            if (aa_width > 0.0f) {
                // Anti-aliased edge using smoothstep
                float dist = sqrt(dist2);
                float edge_outer = threshold + aa_width;
                float edge_inner = threshold - aa_width;
                float t = smoothstep(edge_outer, edge_inner, dist);
                float clr = (value * alpha) + (color * (1.0 - alpha));
                color = mix(color, clr, t * alpha);
            } else {
                // Hard edge (original behavior, fast path)
                float clr = (value * alpha) + (color * (1.0 - alpha));
                color = (dist2 < threshold2) ? clr : color;
            }
        }
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

extern "C" __kernel
void rects(int width, int height,  READWRITE float *a, int nlines, READONLY float4 *coords, READONLY float *aa_widths, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float4 region;
    for (int iline=0; iline < nlines; iline++) {
        region = *coords++;
        float aa_width = *aa_widths++;
        float2 rc = *colors++;
        float value = rc.x;
        float alpha = rc.y;

        float l = min(region.x,region.z);
        float b = min(region.y,region.w);
        float r = max(region.x,region.z);
        float t = max(region.y,region.w);

        bool inside = (l <= x && x <= r && b <= y && y <= t);

        if (aa_width > 0.0f && inside) {
            // Anti-aliased edges using distance to nearest edge
            float dist_l = (float)x - l;
            float dist_r = r - (float)x;
            float dist_b = (float)y - b;
            float dist_t = t - (float)y;
            float edge_dist = min(min(dist_l, dist_r), min(dist_b, dist_t));

            float blend_t = smoothstep(0.0f, aa_width, edge_dist);
            float clr = (value * alpha) + (color * (1.0f - alpha));
            color = mix(color, clr, blend_t * alpha);
        } else if (inside) {
            // Hard edge (original behavior, fast path)
            float clr = (value * alpha) + (color * (1.0f - alpha));
            color = clr;
        }
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

extern "C" __kernel
void circles(int width, int height,  READWRITE float *a, int nlines, READONLY float *coords, READONLY float *aa_widths, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float2 point = make_float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float cx = *coords++;
        float cy = *coords++;
        float radius = *coords++;
        float aa_width = *aa_widths++;
        float2 rc = *colors++;
        float value = rc.x;
        float alpha = rc.y;
        float2 center = make_float2(cx, cy);
        float dist = sqrt(dxdy2(center, point));

        if (aa_width > 0.0f) {
            // Anti-aliased edge using smoothstep
            float edge_outer = radius + aa_width;
            float edge_inner = radius - aa_width;
            float t = smoothstep(edge_outer, edge_inner, dist);
            float clr = (value * alpha) + (color * (1.0 - alpha));
            color = mix(color, clr, t * alpha);
        } else {
            // Hard edge (original behavior, fast path)
            float clr = (value * alpha) + (color * (1.0 - alpha));
            color = (dist <= radius) ? clr : color;
        }
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

#define signex(p1,p2,p3) ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

// Helper function to compute distance from point to line segment
__device__ static inline float dist_to_segment(float2 p, float2 a, float2 b) {
  float2 pa = p - a;
  float2 ba = b - a;
  float h = fmaxf(0.0f, fminf(1.0f, dot(pa, ba) / dot(ba, ba)));
  return length(pa - h * ba);
}

extern "C" __kernel
void triangles(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, READONLY float *aa_widths, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float2 point = make_float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float2 v1 = *coords++;
        float2 v2 = *coords++;
        float2 v3 = *coords++;
        float aa_width = *aa_widths++;
        float2 rc = *colors++;
        float value = rc.x;
        float alpha = rc.y;

        float d1 = signex(point, v1, v2);
        float d2 = signex(point, v2, v3);
        float d3 = signex(point, v3, v1);
        bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        bool inside = !(has_neg && has_pos);

        if (aa_width > 0.0f && inside) {
            // Anti-aliased edges using distance to nearest edge
            float dist1 = dist_to_segment(point, v1, v2);
            float dist2 = dist_to_segment(point, v2, v3);
            float dist3 = dist_to_segment(point, v3, v1);
            float edge_dist = min(min(dist1, dist2), dist3);

            float blend_t = smoothstep(0.0f, aa_width, edge_dist);
            float clr = (value * alpha) + (color * (1.0f - alpha));
            color = mix(color, clr, blend_t * alpha);
        } else if (inside) {
            // Hard edge (original behavior, fast path)
            float clr = (value * alpha) + (color * (1.0f - alpha));
            color = clr;
        }
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
extern "C" __kernel
void ellipses(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, READONLY float *aa_widths, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float2 point = make_float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float2 center = *coords++;
        float2 axes = *coords++;
        float aa_width = *aa_widths++;
        float2 rc = *colors++;
        float value = rc.x;
        float alpha = rc.y;

        float2 delta = (point - center) / axes;
        float d = length(delta);

        if (aa_width > 0.0f) {
            // Anti-aliased edge using smoothstep
            // d < 1: inside, d = 1: on boundary, d > 1: outside
            float avg_radius = (axes.x + axes.y) * 0.5f;
            float dist_to_edge = (d - 1.0f) * avg_radius;
            float edge_outer = aa_width;
            float edge_inner = -aa_width;
            float t = smoothstep(edge_outer, edge_inner, dist_to_edge);
            float clr = (value * alpha) + (color * (1.0f - alpha));
            color = mix(color, clr, t * alpha);
        } else {
            // Hard edge (original behavior, fast path)
            float clr = (value * alpha) + (color * (1.0f - alpha));
            color = (d <= 1.0f) ? clr : color;
        }
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}


//#define DEBUG
extern "C" __kernel
void composite(int width, int height,  READWRITE float *a, int n, READONLY int4 *coords,
                READWRITE float2 *pixels, READONLY int *ipixels, int npixels)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    int2 xy = make_int2(x,y);
    for (int j=0; j < n; j++) {
        // Get the x,y,w,h of the destination
        int4 coord = coords[j];
        // Get the first pixel location in the pixel buffer
        int ipixel = ipixels[j];
        int2 lb = make_int2(coord.x, coord.y);
        int2 yt = make_int2(coord.x, coord.y) + make_int2(coord.z, coord.w);
        // Is the current xy inside the destination area?
        int outside = ANYISLESS(xy,lb) + ANYISGTE(xy,yt);
        int2 source_location = xy-lb;
        // CUDA uses row-major order, so Y varies along outer dimension
        int source_offset = (source_location.y*coord.z)+source_location.x;
        int ipixel_source = ipixel + source_offset;
        int inside = !outside && 0<=ipixel_source && ipixel_source < npixels;
        float2 color_alpha = inside ? pixels[ipixel_source] : make_float2(0.0f,0.0f);
        float clr = (color_alpha.x*color_alpha.y)+(color*(1.0-color_alpha.y));
        color = inside ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
