/* Optimized Radix-2 FFT Implementation for Metal
 *
 * This implements the Cooley-Tukey FFT algorithm using parallel butterfly operations.
 * Key features:
 * - O(N log N) complexity (vs O(N²) for DFT)
 * - Uses threadgroup memory for efficient data sharing
 * - Parallel butterfly operations with proper barriers
 * - In-place computation
 * - Supports power-of-2 sizes
 *
 * Algorithm: Decimation-in-time (DIT) radix-2 FFT
 */

#include <metal_stdlib>
using namespace metal;

#ifndef M_PI_F
  #define M_PI_F 3.14159265358979323846f
#endif

#ifndef M_2PI_F
  #define M_2PI_F 6.28318530717958647692f
#endif

/* Complex number operations */
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 cadd(float2 a, float2 b) {
    return float2(a.x + b.x, a.y + b.y);
}

inline float2 csub(float2 a, float2 b) {
    return float2(a.x - b.x, a.y - b.y);
}

/* Bit reversal for FFT permutation */
inline uint bit_reverse(uint x, uint log2n) {
    uint result = 0;
    for (uint i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/* Compute twiddle factor W_N^k = e^(-2πik/N) for forward FFT
 * For inverse FFT, we use conjugate: W_N^k = e^(+2πik/N)
 */
inline float2 twiddle_factor(uint k, uint N, int direction) {
    float angle = direction * M_2PI_F * float(k) / float(N);
    return float2(cos(angle), sin(angle));
}

/* Radix-2 FFT Kernel using threadgroup memory
 *
 * Each threadgroup processes one FFT of size up to 1024 (Metal's max threadgroup size)
 * For larger FFTs, we use a hierarchical approach with multiple kernel dispatches
 *
 * Parameters:
 *   input: Input complex array
 *   output: Output complex array
 *   direction: 1 for forward FFT, -1 for inverse FFT
 *   n: FFT size (must be power of 2)
 *   log2n: log2(n)
 *
 * Threadgroup size must equal n (for n <= 1024)
 */
kernel void fft_radix2_1024(
    device const float2 *input [[ buffer(0) ]],
    device float2 *output [[ buffer(1) ]],
    constant int &direction [[ buffer(2) ]],
    constant uint &n [[ buffer(3) ]],
    constant uint &log2n [[ buffer(4) ]],
    threadgroup float2 *shared_data [[ threadgroup(0) ]],
    uint gid [[ thread_position_in_grid ]],
    uint lid [[ thread_position_in_threadgroup ]],
    uint tid [[ threadgroup_position_in_grid ]],
    uint tcount [[ threadgroups_per_grid ]]
) {
    // Each threadgroup processes one independent FFT
    // For arrays larger than n, we process multiple FFTs
    uint fft_id = tid;
    uint base_offset = fft_id * n;

    if (gid >= n * tcount) return;

    // Bit-reversed input loading
    uint rev_lid = bit_reverse(lid, log2n);
    uint input_idx = base_offset + rev_lid;

    // Load input with bit-reversal into shared memory
    if (input_idx < n * tcount) {
        shared_data[lid] = input[input_idx];
    } else {
        shared_data[lid] = float2(0.0f, 0.0f);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey FFT: log2n stages
    for (uint stage = 0; stage < log2n; stage++) {
        uint m = 1 << (stage + 1);  // Butterfly span (2, 4, 8, 16, ...)
        uint m2 = m >> 1;            // Half span (1, 2, 4, 8, ...)

        // Determine butterfly partner for this thread
        uint butterfly_block = lid / m;         // Which butterfly block (0, 1, 2, ...)
        uint pos_in_block = lid % m;            // Position within block

        bool is_upper = pos_in_block < m2;
        uint partner_offset = is_upper ? m2 : -m2;
        uint partner = lid + partner_offset;

        // Twiddle factor index
        uint k = pos_in_block % m2;
        float2 w = twiddle_factor(k, m, direction);

        // Butterfly operation
        float2 a = shared_data[lid];
        float2 b = shared_data[partner];

        if (is_upper) {
            // Upper half: out = a + w*b
            shared_data[lid] = cadd(a, cmul(w, b));
        } else {
            // Lower half: out = a - w*b (but a and b are swapped from upper perspective)
            // Actually: upper computed a + w*b, we compute a - w*b
            // But we need to swap: this is the 'b' in upper's computation
            shared_data[lid] = csub(b, cmul(w, a));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale for inverse FFT
    float2 result = shared_data[lid];
    if (direction == -1) {
        result = result / float(n);
    }

    // Write output
    uint output_idx = base_offset + lid;
    if (output_idx < n * tcount) {
        output[output_idx] = result;
    }
}

/* Radix-2 FFT for small sizes (≤ 256) - optimized version
 * This is more efficient for small FFTs by using less shared memory
 */
kernel void fft_radix2_256(
    device const float2 *input [[ buffer(0) ]],
    device float2 *output [[ buffer(1) ]],
    constant int &direction [[ buffer(2) ]],
    constant uint &n [[ buffer(3) ]],
    constant uint &log2n [[ buffer(4) ]],
    threadgroup float2 *shared [[ threadgroup(0) ]],
    uint gid [[ thread_position_in_grid ]],
    uint lid [[ thread_position_in_threadgroup ]],
    uint tid [[ threadgroup_position_in_grid ]]
) {
    uint base = tid * n;

    if (lid >= n) return;

    // Bit-reversed load
    uint rev = bit_reverse(lid, log2n);
    shared[lid] = input[base + rev];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FFT stages
    for (uint s = 0; s < log2n; s++) {
        uint m = 1 << (s + 1);
        uint m2 = m >> 1;

        uint k = lid % m2;
        uint j = 2 * m2 * (lid / m) + k;

        float2 w = twiddle_factor(k, m, direction);
        float2 t = cmul(w, shared[j + m2]);
        float2 u = shared[j];

        if (lid < n / 2) {
            shared[j] = cadd(u, t);
            shared[j + m2] = csub(u, t);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize for inverse
    float2 result = shared[lid];
    if (direction == -1) {
        result = result / float(n);
    }

    output[base + lid] = result;
}

/* Four-step FFT for large sizes (> 1024)
 * Decomposes large FFT into smaller FFTs that fit in threadgroup memory
 * Uses: FFT(N) = FFT(N1) × FFT(N2) where N = N1 × N2
 *
 * Step 1: Transpose input
 * Step 2: FFT rows
 * Step 3: Twiddle factors
 * Step 4: FFT columns
 * Step 5: Transpose output
 */

// Transpose kernel for four-step FFT
kernel void transpose_complex(
    device const float2 *input [[ buffer(0) ]],
    device float2 *output [[ buffer(1) ]],
    constant uint &rows [[ buffer(2) ]],
    constant uint &cols [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    if (gid.x >= cols || gid.y >= rows) return;

    uint in_idx = gid.y * cols + gid.x;
    uint out_idx = gid.x * rows + gid.y;

    output[out_idx] = input[in_idx];
}

// Twiddle multiplication for four-step FFT
kernel void apply_twiddle_factors(
    device float2 *data [[ buffer(0) ]],
    constant int &direction [[ buffer(1) ]],
    constant uint &N1 [[ buffer(2) ]],
    constant uint &N2 [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    if (gid.x >= N2 || gid.y >= N1) return;

    uint idx = gid.y * N2 + gid.x;
    uint N = N1 * N2;

    // Twiddle factor: W_N^(k*l) where k=gid.y, l=gid.x
    uint exponent = gid.y * gid.x;
    float2 w = twiddle_factor(exponent, N, direction);

    data[idx] = cmul(data[idx], w);
}

/* Batch FFT kernel for processing multiple independent FFTs
 * Useful for spectrograms where we compute FFT for each time window
 */
kernel void batch_fft_radix2(
    device const float2 *input [[ buffer(0) ]],
    device float2 *output [[ buffer(1) ]],
    constant int &direction [[ buffer(2) ]],
    constant uint &fft_size [[ buffer(3) ]],
    constant uint &log2n [[ buffer(4) ]],
    constant uint &batch_count [[ buffer(5) ]],
    threadgroup float2 *shared [[ threadgroup(0) ]],
    uint gid [[ thread_position_in_grid ]],
    uint lid [[ thread_position_in_threadgroup ]],
    uint tid [[ threadgroup_position_in_grid ]]
) {
    // Each threadgroup handles one FFT
    if (tid >= batch_count) return;
    if (lid >= fft_size) return;

    uint offset = tid * fft_size;

    // Bit-reversed load
    uint rev = bit_reverse(lid, log2n);
    shared[lid] = input[offset + rev];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FFT computation
    for (uint stage = 0; stage < log2n; stage++) {
        uint m = 1 << (stage + 1);
        uint m2 = m >> 1;

        uint k = lid % m2;
        uint block_start = (lid / m) * m;
        uint j = block_start + k;

        if (j < fft_size && j + m2 < fft_size) {
            float2 w = twiddle_factor(k, m, direction);
            float2 t = cmul(w, shared[j + m2]);
            float2 u = shared[j];

            shared[j] = cadd(u, t);
            shared[j + m2] = csub(u, t);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize for inverse
    float2 result = shared[lid];
    if (direction == -1) {
        result = result / float(fft_size);
    }

    output[offset + lid] = result;
}
