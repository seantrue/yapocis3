##  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
<%include file="constants.mako" />
<%include file="helpers.mako" />

// ============================================================================
// Simplex Noise Implementation (Ken Perlin, 2001) - Metal Backend
// ============================================================================

// Hash function for pseudo-random gradient generation
inline int hash_int(int x, int y, int z, int seed)
{
    int h = seed;
    h ^= x * 374761393;
    h ^= y * 668265263;
    h ^= z * 1264258069;
    h = (h ^ (h >> 13)) * 1274126177;
    return h ^ (h >> 16);
}

// 2D gradient vectors (12 directions)
constant float2 grad2[] = {
    float2( 1.0f,  1.0f), float2(-1.0f,  1.0f),
    float2( 1.0f, -1.0f), float2(-1.0f, -1.0f),
    float2( 1.0f,  0.0f), float2(-1.0f,  0.0f),
    float2( 0.0f,  1.0f), float2( 0.0f, -1.0f),
    float2( 1.0f,  1.0f), float2( 0.0f, -1.0f),
    float2(-1.0f,  1.0f), float2( 0.0f, -1.0f)
};

// 3D gradient vectors (12 directions)
constant float3 grad3[] = {
    float3( 1.0f,  1.0f,  0.0f), float3(-1.0f,  1.0f,  0.0f),
    float3( 1.0f, -1.0f,  0.0f), float3(-1.0f, -1.0f,  0.0f),
    float3( 1.0f,  0.0f,  1.0f), float3(-1.0f,  0.0f,  1.0f),
    float3( 1.0f,  0.0f, -1.0f), float3(-1.0f,  0.0f, -1.0f),
    float3( 0.0f,  1.0f,  1.0f), float3( 0.0f, -1.0f,  1.0f),
    float3( 0.0f,  1.0f, -1.0f), float3( 0.0f, -1.0f, -1.0f)
};

// Dot product helper for 2D
inline float dot2(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

// Dot product helper for 3D
inline float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 2D simplex noise (single octave)
inline float simplex2d_single(float x, float y, int seed, int tileable, int tile_w, int tile_h)
{
    const float F2 = 0.3660254037844386f;
    const float G2 = 0.21132486540518713f;

    float s = (x + y) * F2;
    int i = floor(x + s);
    int j = floor(y + s);

    float t = (i + j) * G2;
    float X0 = i - t;
    float Y0 = j - t;
    float x0 = x - X0;
    float y0 = y - Y0;

    int i1, j1;
    if (x0 > y0) {
        i1 = 1; j1 = 0;
    } else {
        i1 = 0; j1 = 1;
    }

    float x1 = x0 - i1 + G2;
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;

    int ii = i, jj = j;
    if (tileable) {
        ii = i % tile_w;
        jj = j % tile_h;
        if (ii < 0) ii += tile_w;
        if (jj < 0) jj += tile_h;
    }

    float n0, n1, n2;

    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 < 0.0f) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        int gi0 = hash_int(ii, jj, 0, seed) & 11;
        n0 = t0 * t0 * dot2(grad2[gi0], float2(x0, y0));
    }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 < 0.0f) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        int ii1 = tileable ? (ii + i1) % tile_w : (ii + i1);
        int jj1 = tileable ? (jj + j1) % tile_h : (jj + j1);
        if (tileable && ii1 < 0) ii1 += tile_w;
        if (tileable && jj1 < 0) jj1 += tile_h;
        int gi1 = hash_int(ii1, jj1, 0, seed) & 11;
        n1 = t1 * t1 * dot2(grad2[gi1], float2(x1, y1));
    }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 < 0.0f) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        int ii2 = tileable ? (ii + 1) % tile_w : (ii + 1);
        int jj2 = tileable ? (jj + 1) % tile_h : (jj + 1);
        if (tileable && ii2 < 0) ii2 += tile_w;
        if (tileable && jj2 < 0) jj2 += tile_h;
        int gi2 = hash_int(ii2, jj2, 0, seed) & 11;
        n2 = t2 * t2 * dot2(grad2[gi2], float2(x2, y2));
    }

    return 70.0f * (n0 + n1 + n2);
}

// 3D simplex noise (single octave)
inline float simplex3d_single(float x, float y, float z, int seed)
{
    const float F3 = 0.3333333333333333f;
    const float G3 = 0.16666666666666666f;

    float s = (x + y + z) * F3;
    int i = floor(x + s);
    int j = floor(y + s);
    int k = floor(z + s);

    float t = (i + j + k) * G3;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1=1; j1=0; k1=0; i2=1; j2=1; k2=0;
        } else if (x0 >= z0) {
            i1=1; j1=0; k1=0; i2=1; j2=0; k2=1;
        } else {
            i1=0; j1=0; k1=1; i2=1; j2=0; k2=1;
        }
    } else {
        if (y0 < z0) {
            i1=0; j1=0; k1=1; i2=0; j2=1; k2=1;
        } else if (x0 < z0) {
            i1=0; j1=1; k1=0; i2=0; j2=1; k2=1;
        } else {
            i1=0; j1=1; k1=0; i2=1; j2=1; k2=0;
        }
    }

    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3;
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    float n0, n1, n2, n3;

    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0.0f) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        int gi0 = hash_int(i, j, k, seed) % 12;
        n0 = t0 * t0 * dot3(grad3[gi0], float3(x0, y0, z0));
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0.0f) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        int gi1 = hash_int(i + i1, j + j1, k + k1, seed) % 12;
        n1 = t1 * t1 * dot3(grad3[gi1], float3(x1, y1, z1));
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0.0f) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        int gi2 = hash_int(i + i2, j + j2, k + k2, seed) % 12;
        n2 = t2 * t2 * dot3(grad3[gi2], float3(x2, y2, z2));
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0.0f) {
        n3 = 0.0f;
    } else {
        t3 *= t3;
        int gi3 = hash_int(i + 1, j + 1, k + 1, seed) % 12;
        n3 = t3 * t3 * dot3(grad3[gi3], float3(x3, y3, z3));
    }

    return 32.0f * (n0 + n1 + n2 + n3);
}

// ============================================================================
// 2D Simplex Noise Kernel (Metal)
// ============================================================================

kernel void simplex2d(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float *output [[ buffer(2) ]],
                      constant int &seed [[ buffer(3) ]],
                      constant float &scale [[ buffer(4) ]],
                      constant float &offset_x [[ buffer(5) ]],
                      constant float &offset_y [[ buffer(6) ]],
                      constant int &octaves [[ buffer(7) ]],
                      constant float &persistence [[ buffer(8) ]],
                      constant float &lacunarity [[ buffer(9) ]],
                      constant int &tileable [[ buffer(10) ]],
                      uint gid [[ thread_position_in_grid ]])
{
    int x = X(gid);
    int y = Y(gid);

    if (x >= width || y >= height) return;

    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = scale;
    float max_value = 0.0f;

    for (int octave = 0; octave < octaves; octave++) {
        float nx = (x / (float)width) * frequency + offset_x;
        float ny = (y / (float)height) * frequency + offset_y;

        int tile_w = (int)(width / frequency);
        int tile_h = (int)(height / frequency);
        if (tile_w < 1) tile_w = 1;
        if (tile_h < 1) tile_h = 1;

        float noise = simplex2d_single(nx, ny, seed + octave, tileable, tile_w, tile_h);
        total += noise * amplitude;

        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    float value = (total / max_value + 1.0f) * 0.5f;
    value = clamp(value, 0.0f, 1.0f);

    output[gid] = value;
}

// ============================================================================
// 3D Simplex Noise Kernel (Metal)
// ============================================================================

kernel void simplex3d(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      constant int &depth [[ buffer(2) ]],
                      device float *output [[ buffer(3) ]],
                      constant int &seed [[ buffer(4) ]],
                      constant float &scale [[ buffer(5) ]],
                      constant float &offset_x [[ buffer(6) ]],
                      constant float &offset_y [[ buffer(7) ]],
                      constant float &offset_z [[ buffer(8) ]],
                      constant int &octaves [[ buffer(9) ]],
                      constant float &persistence [[ buffer(10) ]],
                      constant float &lacunarity [[ buffer(11) ]],
                      uint gid [[ thread_position_in_grid ]])
{
    int z = gid / (width * height);
    int remainder = gid % (width * height);
    int x = X(remainder);
    int y = Y(remainder);

    if (x >= width || y >= height || z >= depth) return;

    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = scale;
    float max_value = 0.0f;

    for (int octave = 0; octave < octaves; octave++) {
        float nx = (x / (float)width) * frequency + offset_x;
        float ny = (y / (float)height) * frequency + offset_y;
        float nz = (z / (float)depth) * frequency + offset_z;

        float noise = simplex3d_single(nx, ny, nz, seed + octave);
        total += noise * amplitude;

        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    float value = (total / max_value + 1.0f) * 0.5f;
    value = clamp(value, 0.0f, 1.0f);

    output[gid] = value;
}
