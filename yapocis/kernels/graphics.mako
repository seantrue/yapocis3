#define READONLY __constant
#define READWRITE __global
__kernel
void lines(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, float rdist, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = i/height;
    int y = i-(x*height);
    float threshold = (rdist * .5) + .5;
    float2 point = (float2)((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float2 linestart  = *coords++;
        float2 lineend = *coords++;
        float2 value_alpha = *colors++;
        float value = value_alpha.s0;
        float alpha = value_alpha.s1;
        float linemag = distance( lineend, linestart );
        float2 delta = lineend-linestart;
        float2 u2 = (point-linestart)*(delta);
        float u = (u2.x+u2.y)/(linemag*linemag);
        float2 intersection = linestart + (u * delta);
        float dist = distance(point, intersection);
        float clr = (value*alpha)+(color*(1.0-alpha));
        color = (u >= 0.0f && u <= 1.0 && dist < threshold) ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel
void rects(int width, int height,  READWRITE float *a, int nlines, READONLY float4 *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    float x = i/height;
    float y = i-(x*height);
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

__kernel
void circles(int width, int height,  READWRITE float *a, int nlines, READONLY float *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = i/height;
    int y = i-(x*height);
    float2 point = (float2)((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float x = *coords++;
        float y = *coords++;
        float radius = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0-rc.y));

        float2 center = (float2) (x,y);
        color = distance(center,point) <= radius ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

#define signex(p1,p2,p3) ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

__kernel
void triangles(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = i/height;
    int y = i-(x*height);
    float2 point = (float2)((float)x,(float)y);
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
__kernel
void ellipses(int width, int height,  READWRITE float *a, int nlines, READONLY float2 *coords, READONLY float2 *colors)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int x = i/height;
    int y = i-(x*height);
    float2 point = (float2)((float)x,(float)y);
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
__kernel
void composite(int width, int height,  READWRITE float *a, int n, READONLY int4 *coords,
                READWRITE float2 *pixels, READONLY int *ipixels, int npixels)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    int2 xy = (int2) (i/height,(i-((i/height)*height)));
    for (int j=0; j < n; j++) {
        // Get the x,y,w,h of the destination
        int4 coord = coords[j];
        // Get the first pixel location in the pixel buffer
        int ipixel = ipixels[j];
        int2 lb = coord.s10;
        int2 yt = coord.s10 + coord.s32;
        // Is the current xy inside the destination area?
        int outside = any(isless(xy,lb)) + any(isgreaterequal(xy,yt));
        int2 source_location = xy-lb;
        int source_offset = (source_location.x*coord.s2)+source_location.y;
        int ipixel_source = ipixel + source_offset;
        int inside = !outside && 0<=ipixel_source && ipixel_source < npixels;
        float2 color_alpha = inside ? pixels[ipixel_source] : (float2)(0.0f,0.0f);
        float clr = (color_alpha.s0*color_alpha.s1)+(color*(1.0-color_alpha.s1));
        color = inside ? clr : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
