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

/* Complex subtraction */
__device__ __inline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

/* Bit reversal for FFT */
__device__ __inline__ uint bit_reverse(uint x, uint bits) {
    uint result = 0;
    for (uint i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/* Compute twiddle factor W_N^k = e^(-i*2π*k/N) for forward FFT
 * For inverse FFT (direction = -1), we use the conjugate: e^(+i*2π*k/N)
 */
__device__ __inline__ float2 twiddle_factor(uint k, uint N, int direction) {
    // Forward FFT: direction = 1,  angle = -2πk/N (negative)
    // Inverse FFT: direction = -1, angle = +2πk/N (positive)
    float angle = -direction * M_2PI * (float)k / (float)N;
    return make_float2(cosf(angle), sinf(angle));
}

/* Optimized Radix-2 FFT using shared memory and barriers
 * Complexity: O(N log N) vs O(N²) for DFT
 *
 * This implements the Cooley-Tukey decimation-in-time algorithm with:
 * - Shared memory for efficient communication
 * - Parallel butterfly operations
 * - Proper synchronization barriers
 * - In-place computation
 *
 * Block size must equal FFT size (for sizes up to 1024)
 * direction: 1 for forward FFT, -1 for inverse FFT
 */
extern "C" __global__ void fft1d(float2 *input, int n, float2 *output, int direction)
{
    __shared__ float2 shared_data[1024];  // Static shared memory allocation

    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint lid = threadIdx.x;

    if (gid >= n) return;

    // Compute log2(n)
    uint log2n = 0;
    uint temp = n;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }

    // Bit-reversed input loading into shared memory
    uint rev_idx = bit_reverse(lid, log2n);

    // Use optimized FFT only if we have exactly one block of size n
    // This is the only configuration where the shared memory algorithm works correctly
    if (n <= 1024 && gridDim.x == 1 && blockDim.x == n && (n & (n - 1)) == 0) {  // radix-2 needs a power of two
        // Optimized path for small FFTs (≤ 1024) with correct block config
        shared_data[lid] = input[rev_idx];
        __syncthreads();

        // Cooley-Tukey FFT: log2(n) stages
        for (uint stage = 0; stage < log2n; stage++) {
            uint m = 1 << (stage + 1);  // Butterfly span
            uint m2 = m >> 1;            // Half span

            // Calculate position in butterfly
            uint k = lid % m2;           // Twiddle index
            uint block = (lid / m) * m;  // Start of butterfly block
            uint j = block + k;          // Upper butterfly index

            // Only process if within bounds
            if ((lid % m) < m2 && j + m2 < n) {  // one work-item per butterfly (lid and lid + m2 share j)
                float2 w = twiddle_factor(k, m, direction);
                float2 t = cmul(w, shared_data[j + m2]);
                float2 u = shared_data[j];

                shared_data[j] = cadd(u, t);
                shared_data[j + m2] = csub(u, t);
            }

            __syncthreads();
        }

        // Normalize for inverse FFT
        float2 result = shared_data[lid];
        if (direction == -1) {
            result.x /= (float)n;
            result.y /= (float)n;
        }

        output[gid] = result;
    } else {
        // Fallback to DFT for sizes > 1024
        float2 sum = make_float2(0.0f, 0.0f);
        for (uint k = 0; k < n; k++) {
            float angle = -direction * M_2PI * gid * k / (float)n;
            float2 w = make_float2(cosf(angle), sinf(angle));
            sum = cadd(sum, cmul(input[k], w));
        }

        if (direction == -1) {
            sum.x /= (float)n;
            sum.y /= (float)n;
        }

        output[gid] = sum;
    }
}

/* Window function kernel
 * window_type: 0=rectangular, 1=hann, 2=hamming, 3=blackman
 */
extern "C" __global__ void window(float *input, int const window_type, float *output GLOBAL_SIZE_PARAM)
{
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    RETURN_IF_OUTSIDE(gid)
    // The real length, not the block-rounded thread count
    uint n = (uint)yapocis_global_size;

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
extern "C" __global__ void power_spectrum(float2 *fft_output, float *power GLOBAL_SIZE_PARAM)
{
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    // The real length, not the block-rounded thread count
    uint n = (uint)yapocis_global_size;

    if (gid >= n) return;

    float2 val = fft_output[gid];
    power[gid] = val.x * val.x + val.y * val.y;
}

/* Batch 1D FFT - Process multiple FFTs in a single kernel launch
 *
 * This kernel processes multiple independent FFTs simultaneously, amortizing
 * the GPU launch overhead across many FFTs. This is crucial for performance
 * when processing many small FFTs (e.g., spectrograms, batch signal processing).
 *
 * Performance improvement: 10-100x for batches compared to individual launches.
 *
 * Organization:
 * - Grid: (batch_size * fft_size) total threads
 * - Blocks: batch_size blocks, each of size fft_size
 * - Each block processes one independent FFT
 *
 * Memory:
 * - Input/output are 2D arrays: [batch_size][fft_size]
 * - Each block uses shared memory for its FFT
 * - No inter-block communication needed
 *
 * Limitations:
 * - fft_size must be power of 2
 * - fft_size <= 256 (current radix-2 implementation limit)
 * - batch_size limited by available GPU memory
 */
extern "C" __global__ void fft1d_batch(
    float2 *input,
    int const batch_size,
    int const fft_size,
    int const direction,
    float2 *output
) {
    __shared__ float2 shared_data[256];  // Static shared memory allocation

    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint lid = threadIdx.x;
    uint group_id = blockIdx.x;

    // Determine which batch this thread belongs to
    uint batch_idx = group_id;  // Each block handles one FFT
    uint n = fft_size;

    if (batch_idx >= batch_size || lid >= n) return;

    // Calculate offset into input/output arrays
    uint batch_offset = batch_idx * n;

    // Compute log2(n)
    uint log2n = 0;
    uint temp = n;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }

    // Bit-reversed input loading into shared memory
    uint rev_idx = bit_reverse(lid, log2n);

    if (n <= 256 && (n & (n - 1)) == 0) {  // radix-2 needs a power of two
        // Optimized radix-2 FFT path
        shared_data[lid] = input[batch_offset + rev_idx];
        __syncthreads();

        // Cooley-Tukey FFT: log2(n) stages
        for (uint stage = 0; stage < log2n; stage++) {
            uint m = 1 << (stage + 1);  // Butterfly span
            uint m2 = m >> 1;            // Half span

            // Calculate position in butterfly
            uint k = lid % m2;           // Twiddle index
            uint block = (lid / m) * m;  // Start of butterfly block
            uint j = block + k;          // Upper butterfly index

            // Only process if within bounds
            if ((lid % m) < m2 && j + m2 < n) {  // one work-item per butterfly (lid and lid + m2 share j)
                float2 w = twiddle_factor(k, m, direction);
                float2 t = cmul(w, shared_data[j + m2]);
                float2 u = shared_data[j];

                shared_data[j] = cadd(u, t);
                shared_data[j + m2] = csub(u, t);
            }

            __syncthreads();
        }

        // Normalization for inverse FFT
        float2 result = shared_data[lid];
        if (direction == -1) {
            result.x /= (float)n;
            result.y /= (float)n;
        }

        output[batch_offset + lid] = result;
    } else {
        // Fallback to DFT for sizes > 256
        float2 sum = make_float2(0.0f, 0.0f);
        for (uint k = 0; k < n; k++) {
            float angle = -direction * M_2PI * lid * k / (float)n;
            float2 w = make_float2(cosf(angle), sinf(angle));
            sum = cadd(sum, cmul(input[batch_offset + k], w));
        }

        if (direction == -1) {
            sum.x /= (float)n;
            sum.y /= (float)n;
        }

        output[batch_offset + lid] = sum;
    }
}
