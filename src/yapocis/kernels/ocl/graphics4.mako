// Kernels for operating on HxWx4 arrays
// array[,,3] is the "alpha" channel
// The alpha channel is ignored (but must be present) for the input/output array (a)
// Colors are [N,4] where color[,3] is the alpha and is _not_ ignored

<%include file="constants.mako" />
#define DXDY2
<%include file="helpers.mako" />

#define ONES ONE4
#define ZEROS ZERO4

__kernel void lines(int width, int height, READWRITE float4 *a, int nelements,
                    READONLY float2 *coords, READONLY float *rdist,
                    READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  int x = X(i);
  int y = Y(i);
  float4 color = a[i];

  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float threshold = ((*rdist++ * .5) + .5);
    float aa_width = *aa_widths++;
    float threshold2 = threshold * threshold;
    float2 linestart = *coords++;
    float2 lineend = *coords++;
    float4 rgba = *colors++;
    float linemag2 = dxdy2(lineend, linestart);
    float2 delta = lineend - linestart;
    float2 u2 = (point - linestart) * (delta);
    // A zero-length line is a dot at its start (as numbax draws it); 0/0 would skip it
    float u = linemag2 > 0.0f ? (u2.x + u2.y) / linemag2 : 0.0f;
    float2 intersection = linestart + (u * delta);
    float dist2 = dxdy2(point,intersection);

    if (u >= 0.0f && u <= 1.0f) {
      if (aa_width > 0.0f) {
        // Anti-aliased edge using smoothstep
        float dist = sqrt(dist2);
        float edge_outer = threshold + aa_width;
        float edge_inner = threshold - aa_width;
        float t = smoothstep(edge_outer, edge_inner, dist);
        // Porter-Duff "over" compositing with edge blending
        float effective_alpha = t * rgba.s3;
        float3 rgb = rgba.s012 * effective_alpha + color.s012 * (1.0f - effective_alpha);
        float alpha = effective_alpha + color.s3 * (1.0f - effective_alpha);
        color = (float4)(rgb, alpha);
      } else {
        // Hard edge - Porter-Duff "over" compositing
        bool inside = (dist2 < threshold2);
        if (inside) {
          float3 rgb = rgba.s012 * rgba.s3 + color.s012 * (1.0f - rgba.s3);
          float alpha = rgba.s3 + color.s3 * (1.0f - rgba.s3);
          color = (float4)(rgb, alpha);
        }
      }
    }
  }
  a[i] = color;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void rects(int width, int height, READWRITE float4 *a, int nelements,
                    READONLY float4 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 region = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    float l = min(region.x, region.z);
    float b = min(region.y, region.w);
    float r = max(region.x, region.z);
    float t = max(region.y, region.w);

    bool inside = (l <= x && x <= r && b <= y && y <= t);

    if (aa_width > 0.0f && inside) {
      // Anti-aliased edges using distance to nearest edge
      float dist_l = (float)x - l;
      float dist_r = r - (float)x;
      float dist_b = (float)y - b;
      float dist_t = t - (float)y;
      float edge_dist = min(min(dist_l, dist_r), min(dist_b, dist_t));

      float blend_t = smoothstep(0.0f, aa_width, edge_dist);
      // Porter-Duff "over" compositing with edge blending
      float effective_alpha = blend_t * rgba.s3;
      float3 rgb = rgba.s012 * effective_alpha + pixel.s012 * (1.0f - effective_alpha);
      float alpha = effective_alpha + pixel.s3 * (1.0f - effective_alpha);
      pixel = (float4)(rgb, alpha);
    } else if (inside) {
      // Hard edge - Porter-Duff "over" compositing
      float3 rgb = rgba.s012 * rgba.s3 + pixel.s012 * (1.0f - rgba.s3);
      float alpha = rgba.s3 + pixel.s3 * (1.0f - rgba.s3);
      pixel = (float4)(rgb, alpha);
    }
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void circles(int width, int height, READWRITE float4 *a, int nelements,
                      READONLY float *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float cx = *coords++;
    float cy = *coords++;
    float radius = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;
    float2 center = (float2)(cx, cy);
    float dist = sqrt(dxdy2(center, point));

    if (aa_width > 0.0f) {
      // Anti-aliased edge using smoothstep
      float edge_outer = radius + aa_width;
      float edge_inner = radius - aa_width;
      float t = smoothstep(edge_outer, edge_inner, dist);
      // Porter-Duff "over" compositing with edge blending
      float effective_alpha = t * rgba.s3;
      float3 rgb = rgba.s012 * effective_alpha + pixel.s012 * (1.0f - effective_alpha);
      float alpha = effective_alpha + pixel.s3 * (1.0f - effective_alpha);
      pixel = (float4)(rgb, alpha);
    } else {
      // Hard edge - Porter-Duff "over" compositing
      bool inside = (dist <= radius);
      if (inside) {
        float3 rgb = rgba.s012 * rgba.s3 + pixel.s012 * (1.0f - rgba.s3);
        float alpha = rgba.s3 + pixel.s3 * (1.0f - rgba.s3);
        pixel = (float4)(rgb, alpha);
      }
    }
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

#define signex(p1, p2, p3)                                                     \
  ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

// Triangle fill rule. A pixel exactly on an edge shared by two triangles belongs to
// only one of them, so translucent fills are not blended twice along the seam.
// tri_edge is exactly antisymmetric in (a, b): it is computed with the endpoints in a
// canonical order, so the two triangles get exactly opposite values on a shared edge.
static inline float tri_edge(float ax, float ay, float bx, float by, float px, float py) {
  bool swap = (bx < ax) || (bx == ax && by < ay);
  float sx = swap ? bx : ax, sy = swap ? by : ay;
  float ex = swap ? ax : bx, ey = swap ? ay : by;
  float v = (ex - sx) * (py - sy) - (ey - sy) * (px - sx);
  return swap ? -v : v;
}

// With both triangles wound the same way, a shared edge runs in opposite directions,
// and exactly one direction owns pixels lying on it
static inline bool tri_owns(float dx, float dy) {
  return dy > 0.0f || (dy == 0.0f && dx > 0.0f);
}

static inline bool tri_inside(float px, float py, float x1, float y1, float x2, float y2,
                    float x3, float y3) {
  float area = tri_edge(x1, y1, x2, y2, x3, y3);
  if (area == 0.0f) return false;
  float s = area > 0.0f ? 1.0f : -1.0f;  // normalise the winding
  float e1 = s * tri_edge(x1, y1, x2, y2, px, py);
  float e2 = s * tri_edge(x2, y2, x3, y3, px, py);
  float e3 = s * tri_edge(x3, y3, x1, y1, px, py);
  return (e1 > 0.0f || (e1 == 0.0f && tri_owns(s * (x2 - x1), s * (y2 - y1)))) &&
         (e2 > 0.0f || (e2 == 0.0f && tri_owns(s * (x3 - x2), s * (y3 - y2)))) &&
         (e3 > 0.0f || (e3 == 0.0f && tri_owns(s * (x1 - x3), s * (y1 - y3))));
}

// Helper function to compute distance from point to line segment
static __inline float dist_to_segment(float2 p, float2 a, float2 b) {
  float2 pa = p - a;
  float2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
  return length(pa - h * ba);
}

__kernel void triangles(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY float2 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float2 v1 = *coords++;
    float2 v2 = *coords++;
    float2 v3 = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

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
      // Porter-Duff "over" compositing with edge blending
      float effective_alpha = blend_t * rgba.s3;
      float3 rgb = rgba.s012 * effective_alpha + pixel.s012 * (1.0f - effective_alpha);
      float alpha = effective_alpha + pixel.s3 * (1.0f - effective_alpha);
      pixel = (float4)(rgb, alpha);
    } else if (inside) {
      // Hard edge - Porter-Duff "over" compositing
      float3 rgb = rgba.s012 * rgba.s3 + pixel.s012 * (1.0f - rgba.s3);
      float alpha = rgba.s3 + pixel.s3 * (1.0f - rgba.s3);
      pixel = (float4)(rgb, alpha);
    }
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// mesh_triangles: triangles with a top-left fill rule, for meshes (a triangulated
// polygon). A pixel exactly on an edge shared by two triangles belongs to only one
// of them, so translucent fills are not blended twice along the seams. Unlike
// triangles, pixels on a triangle's non-owning edges are not drawn.
__kernel void mesh_triangles(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY float2 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float2 v1 = *coords++;
    float2 v2 = *coords++;
    float2 v3 = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    bool inside = tri_inside(point.x, point.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y);

    if (aa_width > 0.0f && inside) {
      // Anti-aliased edges using distance to nearest edge
      float dist1 = dist_to_segment(point, v1, v2);
      float dist2 = dist_to_segment(point, v2, v3);
      float dist3 = dist_to_segment(point, v3, v1);
      float edge_dist = min(min(dist1, dist2), dist3);

      float blend_t = smoothstep(0.0f, aa_width, edge_dist);
      // Porter-Duff "over" compositing with edge blending
      float effective_alpha = blend_t * rgba.s3;
      float3 rgb = rgba.s012 * effective_alpha + pixel.s012 * (1.0f - effective_alpha);
      float alpha = effective_alpha + pixel.s3 * (1.0f - effective_alpha);
      pixel = (float4)(rgb, alpha);
    } else if (inside) {
      // Hard edge - Porter-Duff "over" compositing
      float3 rgb = rgba.s012 * rgba.s3 + pixel.s012 * (1.0f - rgba.s3);
      float alpha = rgba.s3 + pixel.s3 * (1.0f - rgba.s3);
      pixel = (float4)(rgb, alpha);
    }
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
__kernel void ellipses(int width, int height, READWRITE float4 *a, int nelements,
                       READONLY float4 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = (float2)((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 center_axes = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    float2 center = (float2)(center_axes.s0,center_axes.s1);
    float2 axes = (float2)(center_axes.s2,center_axes.s3);
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
      // Porter-Duff "over" compositing with edge blending
      float effective_alpha = t * rgba.s3;
      float3 rgb = rgba.s012 * effective_alpha + pixel.s012 * (1.0f - effective_alpha);
      float alpha = effective_alpha + pixel.s3 * (1.0f - effective_alpha);
      pixel = (float4)(rgb, alpha);
    } else {
      // Hard edge - Porter-Duff "over" compositing
      bool inside = (d <= 1.0f);
      if (inside) {
        float3 rgb = rgba.s012 * rgba.s3 + pixel.s012 * (1.0f - rgba.s3);
        float alpha = rgba.s3 + pixel.s3 * (1.0f - rgba.s3);
        pixel = (float4)(rgb, alpha);
      }
    }
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void composite(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY int4 *coords, READWRITE float4 *pixels,
                        READONLY int *ipixels, int npixels) {
  size_t i = get_global_id(0);
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  int2 xy = (int2)(x,y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    // Get the x,y,w,h of the destination
    int4 coord = *coords++;
    int ipixel = *ipixels++;

    int2 lb = coord.s10;
    int2 yt = lb + coord.s32;
    // Is the current xy inside the destination area?
    int outside = ANYISLESS(xy,lb) + ANYISGTE(xy,yt);
    int2 source_location = xy - lb;
    int source_offset = (source_location.x * coord.s2) + source_location.y;
    int ipixel_source = ipixel + source_offset;
    int inside = !outside && 0 <= ipixel_source && ipixel_source < npixels;

    if (inside) {
      float4 rgba = pixels[ipixel_source];
      // Porter-Duff "over" compositing
      float3 rgb = rgba.s012 * rgba.s3 + pixel.s012 * (1.0f - rgba.s3);
      float alpha = rgba.s3 + pixel.s3 * (1.0f - rgba.s3);
      pixel = (float4)(rgb, alpha);
    }
  }
  a[i] = pixel;
  barrier(CLK_GLOBAL_MEM_FENCE);
}
