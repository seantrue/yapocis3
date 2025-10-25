<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* CUDA FFT Kernels - Production Implementation */

#define M_PI 3.14159265358979323846f
#define M_2PI 6.28318530717958647692f

/* Complex multiplication */
__device__ __inline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* Complex addition */
__device__ __inline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

/* Twiddle factor */
__device__ __inline__ float2 twiddle(int k, int N, int direction) {
    float angle = direction * M_2PI * k / N;
    return make_float2(cosf(angle), sinf(angle));
}

/* 1D FFT kernel using DFT
 * direction: 1 for forward FFT, -1 for inverse FFT
 */
extern "C" __kernel void fft1d(float2 *input, float2 *output, int const direction)
{
    size_t gid = get_global_id(0);
    size_t n = gridDim.x * blockDim.x;

    if (gid >= n) return;

    // DFT computation
    float2 sum = make_float2(0.0f, 0.0f);
    for (size_t k = 0; k < n; k++) {
        float angle = direction * M_2PI * gid * k / (float)n;
        float2 w = make_float2(cosf(angle), sinf(angle));
        sum = cadd(sum, cmul(input[k], w));
    }

    // Scale by 1/N for inverse FFT
    if (direction == -1) {
        sum.x /= n;
        sum.y /= n;
    }

    output[gid] = sum;
}

/* Window function kernel
 * window_type: 0=rectangular, 1=hann, 2=hamming, 3=blackman
 */
extern "C" __kernel void window(float *input, int const window_type, float *output)
{
    size_t gid = get_global_id(0);
    size_t n = gridDim.x * blockDim.x;

    float w = 1.0f;
    float pos = (float)gid / (n - 1);

    switch (window_type) {
        case 1: // Hann window
            w = 0.5f * (1.0f - cosf(M_2PI * pos));
            break;
        case 2: // Hamming window
            w = 0.54f - 0.46f * cosf(M_2PI * pos);
            break;
        case 3: // Blackman window
            w = 0.42f - 0.5f * cosf(M_2PI * pos) + 0.08f * cosf(4.0f * M_PI * pos);
            break;
        default: // Rectangular (no windowing)
            w = 1.0f;
            break;
    }

    output[gid] = input[gid] * w;
}

/* Power spectrum kernel - compute magnitude squared */
extern "C" __kernel void power_spectrum(float2 *fft_output, float *power)
{
    size_t gid = get_global_id(0);
    size_t n = gridDim.x * blockDim.x;

    if (gid >= n) return;

    float2 val = fft_output[gid];
    power[gid] = val.x * val.x + val.y * val.y;
}
