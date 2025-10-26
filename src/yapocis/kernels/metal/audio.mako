// Metal kernels for audio processing and VAD
<%include file="constants.mako" />

kernel void spectral_variance(constant int &width [[ buffer(0) ]],
                               constant int &height [[ buffer(1) ]],
                               const device float *spectrogram [[ buffer(2) ]],
                               device float *variance [[ buffer(3) ]],
                               uint t [[ thread_position_in_grid ]]) {
    if (t >= (uint)width) return;

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

kernel void log_normalize(const device float *input [[ buffer(0) ]],
                          device float *output [[ buffer(1) ]],
                          uint i [[ thread_position_in_grid ]]) {
    float value = input[i];
    // Apply log (add small epsilon to avoid log(0))
    value = log(value + 1e-10f);
    output[i] = value;
}

kernel void normalize_with_minmax(const device float *input [[ buffer(0) ]],
                                   constant float &min_val [[ buffer(1) ]],
                                   constant float &max_val [[ buffer(2) ]],
                                   device float *output [[ buffer(3) ]],
                                   uint i [[ thread_position_in_grid ]]) {
    float value = input[i];
    float range = max_val - min_val;
    if (range > 1e-10f) {
        value = (value - min_val) / range;
    } else {
        value = 0.0f;
    }
    output[i] = value;
}

kernel void fused_vad_features(const device float *spectrogram [[ buffer(0) ]],
                                constant int &lower_freq_idx [[ buffer(1) ]],
                                constant int &upper_freq_idx [[ buffer(2) ]],
                                constant int &width [[ buffer(3) ]],
                                constant int &height [[ buffer(4) ]],
                                device float *features [[ buffer(5) ]],
                                uint t [[ thread_position_in_grid ]]) {
    if (t >= (uint)width) return;

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

    features[t] = log_var;
}

kernel void smooth_1d(const device float *input [[ buffer(0) ]],
                      constant int &window_size [[ buffer(1) ]],
                      constant int &n [[ buffer(2) ]],
                      device float *output [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]]) {
    if (i >= (uint)n) return;

    int half_window = window_size / 2;

    float sum = 0.0f;
    int count = 0;

    for (int offset = -half_window; offset <= half_window; offset++) {
        int idx = (int)i + offset;
        // Boundary handling: clamp to edges
        if (idx < 0) idx = 0;
        if (idx >= n) idx = n - 1;

        sum += input[idx];
        count++;
    }

    output[i] = sum / (float)count;
}
