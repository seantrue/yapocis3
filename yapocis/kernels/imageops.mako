// kernels imageops
__kernel void clip(__global float *input, float low, float high, __global float *output) {
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

__kernel void histogram(__global float *input, int size_of_input, int n_bins, float low, float high, int n_workers, __global int *partials, __global int *histogram) {
    size_t i = get_global_id(0);
    size_t partial_base = i * n_bins;
    size_t first_pixel = i*(size_of_input/n_workers);
    size_t last_pixel = first_pixel + n_bins <= size_of_input ? first_pixel + n_bins: size_of_input;
    for (int ipixel=first_pixel;  ipixel < last_pixel; ipixel++) {
        float value = input[ipixel];
        value = value < low ? low : value;
        value = value > high ? high : value;
        int bucket = (int)((n_bins*value)/(high-low));
        partials[partial_base+bucket] += 1;
    }
}



