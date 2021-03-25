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

#define FASTBINS 1024
__kernel void histogram(__global float *input, int size_of_input, float low, float high, __global int *partials, int local_size, int n_bins) {
    size_t i = get_global_id(0);
    size_t partial_base = i * n_bins;
    size_t first_pixel = i*local_size;
    size_t last_pixel = first_pixel + local_size < size_of_input ? first_pixel + local_size-1: size_of_input-1;
    // printf("i %d pb %d fp %d lp %d\n", i, partial_base, first_pixel, last_pixel);
    __local int buffer[FASTBINS];
    if (n_bins > FASTBINS) {
        for (int ipixel=first_pixel;  ipixel <= last_pixel; ipixel++) {
            float value = input[ipixel];
            value = value < low ? low : value;
            value = value > high ? high : value;
            int bucket = (int)((n_bins*value)/(high-low));
            partials[partial_base+bucket] += 1;
        }
    }
    else {
        for (int j=0; j < n_bins; j++)
                buffer[j] = 0;
            for (int ipixel=first_pixel;  ipixel <= last_pixel; ipixel++) {
                float value = input[ipixel];
                value = value < low ? low : value;
                value = value > high ? high : value;
                int bucket = (int)((n_bins*value)/(high-low));
                buffer[bucket] += 1;
        }
        for (int j = 0; j < n_bins; j ++) {
            partials[partial_base+j] = buffer[j];
        }
    }
}



