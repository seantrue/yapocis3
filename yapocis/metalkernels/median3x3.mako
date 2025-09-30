/* Metal kernel to compute 3x3 median filter on a 2D image */

<%include file="constants.mako" />

static inline float median9(thread float *a, int n) {
    float l,h; bool lt;
    % for i in range(9//2+1):
        % for j in range(i+1, 9):
            lt = a[${j}] < a[${i}];
            l = lt ? a[${j}] : a[${i}];
            h = lt ? a[${i}] : a[${j}];
            a[${i}] = l;
            a[${j}] = h;
        % endfor
    % endfor
    return a[n/2];
}

kernel void median3x3(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint2 gxy [[ thread_position_in_grid ]])
{
    const int x = (int)gxy.x;
    const int y = (int)gxy.y;
    // In this interface, 'width' is total elements; derive logical width (rows)
    const int Wdim = width / max(1, height);
    if (x >= Wdim || y >= height) return;

    const size_t i = (size_t)x * (size_t)height + (size_t)y;
    float m = a[i];
    // Skip borders: copy through
    if (x == 0 || y == 0 || x == Wdim - 1 || y == height - 1) { ret[i] = m; return; }
    // Direct neighborhood from device memory
    float aa[9];
    const size_t h = (size_t)height;
    aa[0] = a[i - h - 1];
    aa[1] = a[i - h];
    aa[2] = a[i - h + 1];
    aa[3] = a[i - 1];
    aa[4] = a[i];
    aa[5] = a[i + 1];
    aa[6] = a[i + h - 1];
    aa[7] = a[i + h];
    aa[8] = a[i + h + 1];
    m = median9(&aa[0], 9);
    ret[i] = m;
}
