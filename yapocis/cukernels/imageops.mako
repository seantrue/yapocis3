<%include file="constants.mako"/>
<%include file="helpers.mako"/>

// kernels imageops
__kernel void clamp_pixels(float *input, float low, float high, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < low ? low:  value;
    value = value > high ? high : value;
    output[i] = value;

}
__kernel void stretch(float *input, float input_min, float input_max, float low, float high, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < input_min ? input_min : value;
    value = value > input_max ? input_max : value;
    value = (value - input_min)/(input_max-input_min);
    value = low + value * (high-low);
    output[i] = value;
}

__kernel void binarize(float *input, float threshold, float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value >= threshold ? 1.0 : 0.0;
    output[i] = value;
}

__kernel void histogram(float *input, int size_of_input, float low, float high, int *hst, int size_of_hst) {
    int n_bins = size_of_hst;
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < low ? low : value;
    value = value > high ? high : value;
    size_t bucket = (size_t)((n_bins*value)/(high-low));
    atomic_inc(hst+bucket);
    barrier(CLK_GLOBAL_MEM_FENCE);

}



