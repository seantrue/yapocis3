/* Metal kernels implementing graphics primitives against a float array */

<%include file="constants.mako" />
#define DXDY2
<%include file="helpers.mako" />

kernel void lines(constant int &width [[ buffer(0) ]],
                  constant int &height [[ buffer(1) ]],
                  device float *a [[ buffer(2) ]],
                  constant int &nelements [[ buffer(3) ]],
                  const device float2 *coords [[ buffer(4) ]],
                  const device float *rdist [[ buffer(5) ]],
                  const device float2 *colors [[ buffer(6) ]],
                  uint i [[ thread_position_in_grid ]])
{
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float2 point = float2((float)x, (float)y);
    for (int ielement = 0; ielement < nelements; ielement++) {
        float threshold2 = ((*rdist++ * .5f) + .5f);
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
        float clr = (value*alpha)+(color*(1.0f-alpha));
        color = (u >= 0.0f && u <= 1.0f && dist2 < threshold2) ? clr : color;
    }
    a[i] = color;
}

kernel void rects(constant int &width [[ buffer(0) ]],
                  constant int &height [[ buffer(1) ]],
                  device float *a [[ buffer(2) ]],
                  constant int &nlines [[ buffer(3) ]],
                  const device float4 *coords [[ buffer(4) ]],
                  const device float2 *colors [[ buffer(5) ]],
                  uint i [[ thread_position_in_grid ]])
{
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float4 region;
    for (int iline=0; iline < nlines; iline++) {
        region = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0f-rc.y));

        float l = min(region.x,region.z);
        float b = min(region.y,region.w);
        float r = max(region.x,region.z);
        float t = max(region.y,region.w);

        color = (l <= x && x <= r && b <= y && y <= t) ? clr : color;
    }
    a[i] = color;
}

kernel void circles(constant int &width [[ buffer(0) ]],
                    constant int &height [[ buffer(1) ]],
                    device float *a [[ buffer(2) ]],
                    constant int &nlines [[ buffer(3) ]],
                    const device float *coords [[ buffer(4) ]],
                    const device float2 *colors [[ buffer(5) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float2 point = float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float cx = *coords++;
        float cy = *coords++;
        float radius = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0f-rc.y));
        radius *= radius;
        float2 center = float2(cx,cy);
        color = dxdy2(center,point) <= radius ? clr : color;
    }
    a[i] = color;
}

#define signex(p1,p2,p3) ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

kernel void triangles(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float *a [[ buffer(2) ]],
                      constant int &nlines [[ buffer(3) ]],
                      const device float2 *coords [[ buffer(4) ]],
                      const device float2 *colors [[ buffer(5) ]],
                      uint i [[ thread_position_in_grid ]])
{
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float2 point = float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float2 v1 = *coords++;
        float2 v2 = *coords++;
        float2 v3 = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0f-rc.y));

        float d1 = signex(point, v1, v2);
        float d2 = signex(point, v2, v3);
        float d3 = signex(point, v3, v1);
        bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        color = !(has_neg && has_pos) ? clr : color;
    }
    a[i] = color;
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
kernel void ellipses(constant int &width [[ buffer(0) ]],
                     constant int &height [[ buffer(1) ]],
                     device float *a [[ buffer(2) ]],
                     constant int &nlines [[ buffer(3) ]],
                     const device float2 *coords [[ buffer(4) ]],
                     const device float2 *colors [[ buffer(5) ]],
                     uint i [[ thread_position_in_grid ]])
{
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    float2 point = float2((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++) {
        float2 center = *coords++;
        float2 axes = *coords++;
        float2 rc = *colors++;
        float clr = (rc.x*rc.y)+(color*(1.0f-rc.y));

        float2 dist = (point-center)*(point-center)/(axes*axes);
        color = (dist.x+dist.y) <= 1.0f ? clr : color;
    }
    a[i] = color;
}

kernel void composite(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float *a [[ buffer(2) ]],
                      constant int &n [[ buffer(3) ]],
                      const device int4 *coords [[ buffer(4) ]],
                      const device float2 *pixels [[ buffer(5) ]],
                      const device int *ipixels [[ buffer(6) ]],
                      constant int &npixels [[ buffer(7) ]],
                      uint i [[ thread_position_in_grid ]])
{
    float color = a[i];
    int x = X(i);
    int y = Y(i);
    int2 xy = int2(x,y);
    for (int j=0; j < n; j++) {
        int4 coord = coords[j];
        int ipixel = ipixels[j];
        int2 lb = coord.yx;  // Swap for column-major, matching OCL's coord.s10
        int2 yt = lb + coord.wz;  // Swap dimensions, matching OCL's coord.s32
        int outside = ANYISLESS(xy,lb) + ANYISGTE(xy,yt);
        int2 source_location = xy - lb;
        // Blocks raveled row-major: offset = row*width + col = x*width + y (canvas x=rows, y=cols)
        int source_offset = (source_location.x * coord.z) + source_location.y;
        int ipixel_source = ipixel + source_offset;
        int inside = !outside && 0<=ipixel_source && ipixel_source < npixels;
        float2 color_alpha = inside ? pixels[ipixel_source] : float2(0.0f,0.0f);
        float clr = (color_alpha.x*color_alpha.y)+(color*(1.0f-color_alpha.y));
        color = inside ? clr : color;
    }
    a[i] = color;
}
