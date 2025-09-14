// Metal translations of imageops kernels
<%include file="constants.mako" />

kernel void clamp_pixels(device float *input [[ buffer(0) ]],
                         constant float &low [[ buffer(1) ]],
                         constant float &high [[ buffer(2) ]],
                         device float *output [[ buffer(3) ]],
                         uint gid [[ thread_position_in_grid ]]) {
    float value = input[gid];
    value = value < low ? low:  value;
    value = value > high ? high : value;
    output[gid] = value;
}

kernel void stretch(device float *input [[ buffer(0) ]],
                    constant float &input_min [[ buffer(1) ]],
                    constant float &input_max [[ buffer(2) ]],
                    constant float &low [[ buffer(3) ]],
                    constant float &high [[ buffer(4) ]],
                    device float *output [[ buffer(5) ]],
                    uint gid [[ thread_position_in_grid ]]) {
    float value = input[gid];
    value = value < input_min ? input_min : value;
    value = value > input_max ? input_max : value;
    value = (value - input_min)/(input_max-input_min);
    value = low + value * (high-low);
    output[gid] = value;
}

kernel void binarize(device float *input [[ buffer(0) ]],
                     constant float &threshold [[ buffer(1) ]],
                     device float *output [[ buffer(2) ]],
                     uint gid [[ thread_position_in_grid ]]) {
    float value = input[gid];
    value = value >= threshold ? 1.0 : 0.0;
    output[gid] = value;
}

kernel void histogram(device float *input [[ buffer(0) ]],
                      constant int &size_of_input [[ buffer(1) ]],
                      constant float &low [[ buffer(2) ]],
                      constant float &high [[ buffer(3) ]],
                      device int *hst [[ buffer(4) ]],
                      constant int &size_of_hst [[ buffer(5) ]],
                      uint gid [[ thread_position_in_grid ]]) {
    (void)size_of_input; (void)size_of_hst; // global size determines coverage
    int n_bins = size_of_hst;
    float value = input[gid];
    value = value < low ? low : value;
    value = value > high ? high : value;
    uint bucket = (uint)((n_bins*value)/(high-low));
    bucket = min((uint)(n_bins-1), bucket);
    // Atomic increment on device memory
    atomic_inc((device atomic_int*)(hst + bucket));
}
