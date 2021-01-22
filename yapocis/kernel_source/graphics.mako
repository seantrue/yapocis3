
__kernel
void draw(int width, int height,  __global float *a, int nlines, __global float2 *lines, float rdist)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    // Todo: Do this with a table lookup
    int x = i/height;
    int y = i-(x*height);
    float line_color =1.0;
    float threshold = rdist * .5;
    float alias_threshold = threshold + 0.5;
    float2 point = (float2)((float)x,(float)y);
    for (int iline=0; iline < nlines; iline++, lines += 2) {
        float2 linestart  = lines[0];
        float2 lineend = lines[1];
        float linemag = distance( lineend, linestart );
        float2 delta = lineend-linestart;
        float2 u2 = (point-linestart)*(delta);
        float u = (u2.x+u2.y)/(linemag*linemag);
        //float u = (((point.x - linestart.x) * (lineend.x - linestart.x)) + (( point.y - linestart.y) * (lineend.y - linestart.y))) / (linemag * linemag);

        float2 intersection = linestart + (u * delta);
        float dist = distance( point, intersection );
        color = (u >= 0.0f && u <= 1.0f && dist < threshold) ? line_color : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel
void rects(int width, int height,  __global float *a, int nlines, __global float4 *lines, float rcolor)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    // Todo: Do this with a table lookup
    float x = i/height;
    float y = i-(x*height);
    float4 region;
    for (int iline=0; iline < nlines; iline++) {
        region = lines[iline];
        float l = min(region.x,region.z);
        float b = min(region.y,region.w);
        float r = max(region.x,region.z);
        float t = max(region.y,region.w);
        color = (l <= x && x <= r && b <= y && y <= t) ? rcolor : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel
void circles(int width, int height,  __global float *a, int nlines, __global float2 *lines, float rcolor)
{
    size_t i =  get_global_id(0);
    float color = a[i];
    // Todo: Do this with a table lookup
    int x = i/height;
    int y = i-(x*height);
    float2 center;
    float2 point = (float2)((float)x,(float)y);
    float radius;
    for (int iline=0; iline < nlines; iline++, lines += 2) {
        center = lines[0];
        radius = lines[1].x;
        color = distance(center,point) <= radius ? rcolor : color;
    }
    a[i] = color;
    barrier(CLK_GLOBAL_MEM_FENCE);
}

float signex (float2 p1, float2 p2, float2 p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}


bool pointInTriangle (float2 pt, float2 v1, float2 v2, float2 v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = signex(pt, v1, v2);
    d2 = signex(pt, v2, v3);
    d3 = signex(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}