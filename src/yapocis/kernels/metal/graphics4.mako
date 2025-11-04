// Metal kernels for operating on HxWx4 arrays

<%include file="constants.mako" />
#define DXDY2
<%include file="helpers.mako" />

#define ONES ONE4
#define ZEROS ZERO4

kernel void lines(constant int &width [[ buffer(0) ]],
                  constant int &height [[ buffer(1) ]],
                  device float4 *a [[ buffer(2) ]],
                  constant int &nelements [[ buffer(3) ]],
                  const device float2 *coords [[ buffer(4) ]],
                  const device float *rdist [[ buffer(5) ]],
                  const device float *aa_widths [[ buffer(6) ]],
                  const device float4 *colors [[ buffer(7) ]],
                  uint i [[ thread_position_in_grid ]]) {
  int x = X(i);
  int y = Y(i);
  float4 color = a[i];

  float2 point = float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float threshold = ((*rdist++ * .5f) + .5f);
    float aa_width = *aa_widths++;
    float threshold2 = threshold * threshold;
    float2 linestart = *coords++;
    float2 lineend = *coords++;
    float4 rgba = *colors++;
    float4 alpha = float4(rgba.w, rgba.w, rgba.w, 0.0f);
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
        float4 clr = (rgba * alpha) + (color * (ONES - alpha));
        color = mix(color, clr, t * rgba.w);
      } else {
        // Hard edge (original behavior, fast path)
        float4 clr = (rgba * alpha) + (color * (ONES - alpha));
        color = (dist2 < threshold2) ? clr : color;
      }
    }
  }
  color.w = 0.0f;
  a[i] = color;
}

kernel void rects(constant int &width [[ buffer(0) ]],
                  constant int &height [[ buffer(1) ]],
                  device float4 *a [[ buffer(2) ]],
                  constant int &nelements [[ buffer(3) ]],
                  const device float4 *coords [[ buffer(4) ]],
                  const device float4 *colors [[ buffer(5) ]],
                  uint i [[ thread_position_in_grid ]]) {
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = float4(rgba.w, rgba.w, rgba.w, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float4 region = *coords++;

    float l = min(region.x, region.z);
    float b = min(region.y, region.w);
    float r = max(region.x, region.z);
    float t = max(region.y, region.w);
    pixel = (l <= x && x <= r && b <= y && y <= t) ? clr : pixel;
  }
  a[i] = pixel;
}

kernel void circles(constant int &width [[ buffer(0) ]],
                    constant int &height [[ buffer(1) ]],
                    device float4 *a [[ buffer(2) ]],
                    constant int &nelements [[ buffer(3) ]],
                    const device float *coords [[ buffer(4) ]],
                    const device float4 *colors [[ buffer(5) ]],
                    uint i [[ thread_position_in_grid ]]) {
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  float2 point = float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = float4(rgba.w, rgba.w, rgba.w, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float cx = *coords++;
    float cy = *coords++;
    float radius = *coords++;
    radius *= radius;

    float2 center = float2(cx, cy);
    pixel = dxdy2(center, point) <= radius ? clr : pixel;
  }
  a[i] = pixel;
}

#define signex(p1, p2, p3) ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

kernel void triangles(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float4 *a [[ buffer(2) ]],
                      constant int &nelements [[ buffer(3) ]],
                      const device float2 *coords [[ buffer(4) ]],
                      const device float4 *colors [[ buffer(5) ]],
                      uint i [[ thread_position_in_grid ]]) {
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = float4(rgba.w, rgba.w, rgba.w, 0.0f);
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
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
kernel void ellipses(constant int &width [[ buffer(0) ]],
                     constant int &height [[ buffer(1) ]],
                     device float4 *a [[ buffer(2) ]],
                     constant int &nelements [[ buffer(3) ]],
                     const device float4 *coords [[ buffer(4) ]],
                     const device float4 *colors [[ buffer(5) ]],
                     uint i [[ thread_position_in_grid ]]) {
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 rgba = *colors++;
    float4 alpha = float4(rgba.w, rgba.w, rgba.w, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));

    float4 center_axes = *coords++;
    float2 center = float2(center_axes.x,center_axes.y);
    float2 axes = float2(center_axes.z,center_axes.w);
    float2 dist = (point - center) * (point - center) / (axes * axes);

    pixel = (dist.x + dist.y) <= 1.0f ? clr : pixel;
  }
  a[i] = pixel;
}

kernel void composite(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float4 *a [[ buffer(2) ]],
                      constant int &nelements [[ buffer(3) ]],
                      const device int4 *coords [[ buffer(4) ]],
                      const device float4 *pixels [[ buffer(5) ]],
                      const device int *ipixels [[ buffer(6) ]],
                      constant int &npixels [[ buffer(7) ]],
                      uint i [[ thread_position_in_grid ]]) {
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  int2 xy = int2(x,y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    int4 coord = *coords++;
    int ipixel = *ipixels++;

    int2 lb = coord.yx;  // Swap for column-major, matching OCL's coord.s10
    int2 yt = lb + coord.wz;  // Swap dimensions, matching OCL's coord.s32
    int outside = ANYISLESS(xy,lb) + ANYISGTE(xy,yt);
    int2 source_location = xy - lb;
    // Blocks raveled row-major: offset = row*width + col = x*width + y (canvas x=rows, y=cols)
    int source_offset = (source_location.x * coord.z) + source_location.y;
    int ipixel_source = ipixel + source_offset;
    int inside = !outside && 0 <= ipixel_source && ipixel_source < npixels;

    float4 rgba = inside ? pixels[ipixel_source] : ZEROS;
    float4 alpha = float4(rgba.w, rgba.w, rgba.w, 0.0f);
    float4 clr = (rgba * alpha) + (pixel * (ONES - alpha));
    pixel = inside ? clr : pixel;
  }
  a[i] = pixel;
}
