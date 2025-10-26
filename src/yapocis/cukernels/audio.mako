<%include file="constants.mako"/>
<%include file="helpers.mako"/>

// CUDA kernels for audio processing and VAD

extern "C" __kernel void spectral_variance(int width, int height, float *spectrogram, float *variance) {
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

extern "C" __kernel void log_normalize(float *input, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    // Apply log (add small epsilon to avoid log(0))
    value = logf(value + 1e-10f);
    output[i] = value;
}

extern "C" __kernel void normalize_with_minmax(float *input, float min_val, float max_val, float *output) {
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

extern "C" __kernel void fused_vad_features(float *spectrogram,
                                            int lower_freq_idx, int upper_freq_idx,
                                            int width, int height,
                                            float *features) {
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
    float log_var = logf(variance + 1e-10f);

    features[t] = log_var;
}

extern "C" __kernel void smooth_1d(float *input, int window_size, int n_param, float *output) {
    int i = get_global_id(0);
    size_t n = gridDim.x * blockDim.x;  // Array length (use CUDA's grid size)

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
