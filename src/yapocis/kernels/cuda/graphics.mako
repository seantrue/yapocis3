<%include file="constants.mako"/>
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
#define DXDY2

extern "C" __kernel
void lines(int width, int height,  READWRITE float *a, int nelements, READONLY float2 *coords, READONLY float *rdist, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float2 point = make_float2((float)x, (float)y);
    for (int ielement = 0; ielement < nelements; ielement++) {
        float threshold2 = ((*rdist++ * .5) + .5);
        threshold2 = threshold2*threshold2;
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
        float clr = (value*alpha)+(color*(1.0-alpha));
        color = (u >= 0.0f && u <= 1.0 && dist2 < threshold2) ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

extern "C" __kernel
void rects(int width, int height,  READWRITE float *a, int nlines, READONLY float4 *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float4 region;
    for (int iline=0; iline < nlines; iline++) {
        region = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0-rc.y));

        float l = min(region.x,region.z);
        float b = min(region.y,region.w);
        float r = max(region.x,region.z);
        float t = max(region.y,region.w);

        color = (l <= x && x <= r && b <= y && y <= t) ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

extern "C" __kernel
void circles(int width, int height,  READWRITE float *a, int nlines, READONLY float *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float2 point = make_float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float x = *coords++;
        float y = *coords++;
        float radius = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0-rc.y));
        radius *= radius;
        float2 center = make_float2(x,y);
        color = dxdy2(center,point) <= radius ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

#define signex(p1,p2,p3) ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

extern "C" __kernel
void triangles(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, READONLY float2 *colors)
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
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0-rc.y));

        float d1 = signex(point, v1, v2);
        float d2 = signex(point, v2, v3);
        float d3 = signex(point, v3, v1);
        bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        color = !(has_neg && has_pos) ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
extern "C" __kernel
void ellipses(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = X(i);;
    int y = Y(i);
    float2 point = make_float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float2 center = *coords++;
        float2 axes = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0-rc.y));

        float2 dist = (point-center)*(point-center)/(axes*axes);
        color = (dist.x+dist.y) <= 1.0 ? clr : color;
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
