// Audio processing kernels for VAD and signal analysis

// Compute variance along frequency axis (axis=0) of spectrogram
// spectrogram is stored in row-major order (freq, time)
__kernel void spectral_variance(int width, int height, __global float *spectrogram, __global float *variance) {
    int t = get_global_id(0);  // time index
    if (t >= width) return;

    // Compute mean across frequencies for this time slice
    float sum = 0.0f;
    for (int f = 0; f < height; f++) {
        sum += spectrogram[f * width + t];
    }
    float mean = sum / (float)height;

    // Compute variance
    float var_sum = 0.0f;
    for (int f = 0; f < height; f++) {
        float diff = spectrogram[f * width + t] - mean;
        var_sum += diff * diff;
    }
    variance[t] = var_sum / (float)height;
}

// Apply log transform and normalize to [0, 1]
__kernel void log_normalize(__global float *input, __global float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    // Apply log (add small epsilon to avoid log(0))
    value = log(value + 1e-10f);
    output[i] = value;
}

// Note: Normalization requires global min/max, so we split into two passes
// Pass 1: Find min/max (done on CPU for now)
// Pass 2: Normalize with known min/max
__kernel void normalize_with_minmax(__global float *input, float min_val, float max_val, __global float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    float range = max_val - min_val;
    if (range > 1e-10f) {
        value = (value - min_val) / range;
    } else {
        value = 0.0f;
    }
    output[i] = value;
}

// Fused kernel: bandpass + variance + log
// This operates on spectrogram and produces log-variance features
__kernel void fused_vad_features(__global float *spectrogram,
                                  int lower_freq_idx, int upper_freq_idx,
                                  int width, int height,
                                  __global float *features) {
    int t = get_global_id(0);  // time index
    if (t >= width) return;

    int band_height = upper_freq_idx - lower_freq_idx;
    if (band_height <= 0) {
        features[t] = 0.0f;
        return;
    }

    // Compute mean in bandpass region
    float sum = 0.0f;
    for (int f = lower_freq_idx; f < upper_freq_idx; f++) {
        if (f >= 0 && f < height) {
            sum += spectrogram[f * width + t];
        }
    }
    float mean = sum / (float)band_height;

    // Compute variance in bandpass region
    float var_sum = 0.0f;
    for (int f = lower_freq_idx; f < upper_freq_idx; f++) {
        if (f >= 0 && f < height) {
            float diff = spectrogram[f * width + t] - mean;
            var_sum += diff * diff;
        }
    }
    float variance = var_sum / (float)band_height;

    // Apply log transform
    float log_var = log(variance + 1e-10f);

    // Store result (will need normalization pass afterward)
    features[t] = log_var;
}

// 1D box filter smoothing
// Note: n parameter is for Metal compatibility; OpenCL uses get_global_size(0)
__kernel void smooth_1d(__global float *input, int window_size, int n_param, __global float *output) {
    int i = get_global_id(0);
    size_t n = get_global_size(0);  // Array length (use OpenCL's built-in)

    if (i >= n) return;

    int half_window = window_size / 2;

    float sum = 0.0f;
    int count = 0;

    for (int offset = -half_window; offset <= half_window; offset++) {
        int idx = i + offset;
        // Boundary handling: clamp to edges
        if (idx < 0) idx = 0;
        if (idx >= n) idx = n - 1;

        sum += input[idx];
        count++;
    }

    output[i] = sum / (float)count;
}
