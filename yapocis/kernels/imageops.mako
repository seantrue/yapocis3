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
% if FASTBINS:
#define FASTBINS ${FASTBINS}
% endif
__kernel void histogram(__global float *input, int size_of_input, float low, float high, __global int *partials, int local_size, int height_of_partials, int width_of_partials) {
    int n_bins = height_of_partials;
    int n_workers = width_of_partials;
    size_t i = get_global_id(0);
    size_t partial_base = i * n_bins;
    size_t first_pixel = i*(size_of_input/n_workers);
    size_t last_pixel = first_pixel + n_bins <= (size_t)size_of_input ? first_pixel + n_bins: (size_t)size_of_input;
    for (size_t ipixel=first_pixel;  ipixel < last_pixel; ipixel++) {
        float value = input[ipixel];
        value = value < low ? low : value;
        value = value > high ? high : value;
        size_t bucket = (size_t)((n_bins*value)/(high-low));
        partials[partial_base+bucket] += 1;
    }
#endif
    barrier(CLK_GLOBAL_MEM_FENCE);

}



