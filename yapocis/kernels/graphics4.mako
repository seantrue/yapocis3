// Kernels for operating on HxWx4 arrays
// array[,,3] is the "alpha" channel
// The alpha channel is ignored (but must be present) for the input/output array (a)
// Colors are [N,4] where color[,3] is the alpha and is _not_ ignored

#define ONES (float4)(1.0f, 1.0f, 1.0f, 1.0f)
#define ZEROS (float4)(0.0f, 0.0f, 0.0f, 0.0f)

__kernel void lines(int width, int height, __global float4 *a, int nelements,
                    __global float2 *coords, float rdist,
                    __global float4 *colors) {
  size_t i = get_global_id(0);
  int x = i / height;
  int y = i - (x * height);
  float4 color = a[i];

  float threshold = (rdist * .5) + .5;
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float2 linestart = *coords++;
    float2 lineend = *coords++;
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float linemag = distance(lineend, linestart);
    float2 delta = lineend - linestart;
    float2 u2 = (point - linestart) * (delta);
    float u = (u2.x + u2.y) / (linemag * linemag);
    float2 intersection = linestart + (u * delta);
    float dist = distance(point, intersection);
    float4 clr = (rgba * alpha) + (color * (ONES - alpha));
    color = (u >= 0.0f && u <= 1.0f && dist < threshold) ? clr : color;
  }
  color.s3 = 0.0f;
  a[i] = color;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void rects(int width, int height, __global float4 *a, int nelements,
                    __global float4 *coords, __global float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  float x = i / height;
  float y = i - (x * height);

  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float4 region = *coords++;

    float l = min(region.x, region.z);
    float b = min(region.y, region.w);
    float r = max(region.x, region.z);
    float t = max(region.y, region.w);
    pixel = (l <= x && x <= r && b <= y && y <= t) ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void circles(int width, int height, __global float4 *a, int nelements,
                      __global float *coords, __global float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = i / height;
  int y = i - (x * height);

  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float x = *coords++;
    float y = *coords++;
    float radius = *coords++;

    float2 center = (float2)(x, y);
    pixel = distance(center, point) <= radius ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

#define signex(p1, p2, p3)                                                     \
  ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

__kernel void triangles(int width, int height, __global float4 *a, int nelements,
                        __global float2 *coords, __global float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = i / height;
  int y = i - (x * height);
  float2 center;
  float2 point = (float2)((float)x, (float)y);
  float radius;
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float2 v1 = *coords++;
    float2 v2 = *coords++;
    float2 v3 = *coords++;

    float d1 = signex(point, v1, v2);
    float d2 = signex(point, v2, v3);
    float d3 = signex(point, v3, v1);
    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    pixel = !(has_neg && has_pos) ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
__kernel void ellipses(int width, int height, __global float4 *a, int nelements,
                       __global float4 *coords, __global float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = i / height;
  int y = i - (x * height);
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float4 center_axes = *coords++;
    float2 center = (float2)(center_axes.s0,center_axes.s1);
    float2 axes = (float2)(center_axes.s2,center_axes.s3);
    float2 dist = (point - center) * (point - center) / (axes * axes);

    pixel = (dist.x + dist.y) <= 1.0 ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void composite(int width, int height, __global float4 *a, int nelements,
                        __global int4 *coords, __global float4 *pixels,
                        __global int *ipixels, int npixels) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int2 xy = (int2)(i / height, (i - ((i / height) * height)));
  for (int ielement = 0; ielement < nelements; ielement++) {
    // Get the x,y,w,h of the destination
    int4 coord = *coords++;
    int ipixel = *ipixels++;

    int2 lb = coord.s10;
    int2 yt = lb + coord.s32;
    // Is the current xy inside the destination area?
    int outside = any(isless(xy, lb)) + any(isgreaterequal(xy, yt));
    int2 source_location = xy - lb;
    int source_offset = (source_location.x * coord.s2) + source_location.y;
    int ipixel_source = ipixel + source_offset;
    int inside = !outside && 0 <= ipixel_source && ipixel_source < npixels;

    float4 rgba = inside ? pixels[ipixel_source] : ZEROS;
    float4 alpha = (float4)(rgba.s3, rgba.s3, rgba.s3, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));
    pixel = inside ? clr : pixel;
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}
