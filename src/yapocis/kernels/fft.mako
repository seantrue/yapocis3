/* FFT Kernels - Production Implementation */

#define M_PI 3.14159265358979323846f
#define M_2PI 6.28318530717958647692f

/* Complex multiplication */
inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* Complex addition */
inline float2 cadd(float2 a, float2 b) {
    return (float2)(a.x + b.x, a.y + b.y);
}

/* Twiddle factor */
inline float2 twiddle(int k, int N, int direction) {
    float angle = direction * M_2PI * k / N;
    return (float2)(cos(angle), sin(angle));
}

/* Bit reversal for FFT */
inline int bit_reverse_int(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/* 1D FFT kernel using DFT
 * direction: 1 for forward FFT, -1 for inverse FFT
 *
 * Note: This uses DFT O(N²) for correctness and simplicity.
 * For production at scale, consider using clFFT or similar optimized libraries.
 */
__kernel void fft1d(__global float2 *input, __global float2 *output, int const direction)
{
    size_t gid = get_global_id(0);
    size_t n = get_global_size(0);

    if (gid >= n) return;

    // DFT computation
    float2 sum = (float2)(0.0f, 0.0f);
    for (size_t k = 0; k < n; k++) {
        float angle = direction * M_2PI * gid * k / (float)n;
        float2 w = (float2)(cos(angle), sin(angle));
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
__kernel void window(__global float *input, int const window_type, __global float *output)
{
    size_t gid = get_global_id(0);
    size_t n = get_global_size(0);

    float w = 1.0f;
    float pos = (float)gid / (n - 1);

    switch (window_type) {
        case 1: // Hann window
            w = 0.5f * (1.0f - cos(M_2PI * pos));
            break;
        case 2: // Hamming window
            w = 0.54f - 0.46f * cos(M_2PI * pos);
            break;
        case 3: // Blackman window
            w = 0.42f - 0.5f * cos(M_2PI * pos) + 0.08f * cos(4.0f * M_PI * pos);
            break;
        default: // Rectangular (no windowing)
            w = 1.0f;
            break;
    }

    output[gid] = input[gid] * w;
}

/* Power spectrum kernel - compute magnitude squared */
__kernel void power_spectrum(__global float2 *fft_output, __global float *power)
{
    size_t gid = get_global_id(0);
    size_t n = get_global_size(0);

    if (gid >= n) return;

    float2 val = fft_output[gid];
    power[gid] = val.x * val.x + val.y * val.y;
}
