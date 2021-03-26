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
    size_t first_pixel = i*local_size;
    size_t last_pixel = first_pixel + local_size < size_of_input ? first_pixel + local_size-1: size_of_input-1;
    //printf("i %d/%d sz %d lw %.3f hg %.3f pb %d fp %d lp %d ls %d hp %d wp %d\n", (int)i,n_workers, (int)size_of_input, low, high, (int)partial_base, (int)first_pixel, (int)last_pixel, (int)local_size, (int)height_of_partials, (int)width_of_partials);
    float range = (high-low);
 #ifdef FASTBINS
    if (n_bins > FASTBINS) {
#endif
        for (int j=0; j < n_bins; j++)
            partials[partial_base+j] = 0;
        for (int ipixel=first_pixel;  ipixel <= last_pixel; ipixel++) {
            float value = input[ipixel];
            value = value < low ? low : value;
            value = value > high ? high : value;
            value -= low;
            int bucket = (int)((n_bins*value)/range);
            partials[partial_base+bucket] += 1;
        }
#ifdef FASTBINS
    }
    else {
        volatile __local int histogram[FASTBINS];
        for (int j = 0; j < n_bins; j++) {
            histogram[j] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ipixel=first_pixel;  ipixel <= last_pixel; ipixel++) {
            float value = input[ipixel];
            value = value < low ? low : value;
            value = value > high ? high : value;
            value -= low;
            int bucket = (int)((n_bins*value)/range);
            atomic_inc(&histogram[bucket]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < n_bins; j ++) {
            partials[partial_base+j] = histogram[j];
        }
    }
#endif
    barrier(CLK_GLOBAL_MEM_FENCE);

}



