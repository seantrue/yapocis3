// kernels imageops
__kernel void clamp_pixels(__global float *input, float low, float high, __global float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < low ? low:  value;
    value = value > high ? high : value;
    output[i] = value;

}
__kernel void stretch(__global float *input, float input_min, float input_max, float low, float high, __global float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < input_min ? input_min : value;
    value = value > input_max ? input_max : value;
    value = (value - input_min)/(input_max-input_min);
    value = low + value * (high-low);
    output[i] = value;
}

__kernel void binarize(__global float *input, float threshold, __global float *output) {
    size_t i = get_global_id(0);
    float value = input[i];
    value = value >= threshold ? 1.0 : 0.0;
    output[i] = value;
}

__kernel void histogram(__global float *input, int size_of_input, float low, float high, __global int *hst, int size_of_hst) {
    int n_bins = size_of_hst;
    size_t i = get_global_id(0);
    float value = input[i];
    value = value < low ? low : value;
    value = value > high ? high : value;
    size_t bucket = (size_t)((n_bins*value)/(high-low));
    bucket = min((size_t)(n_bins-1), bucket);  // value == high, as Metal clamps
    atomic_inc(hst+bucket);
    barrier(CLK_GLOBAL_MEM_FENCE);

}



