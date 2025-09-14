/* Test kernel that sums elements of two arrays into a third array. (Metal) */

<%include file="constants.mako" />

kernel void sum(device const float *a [[ buffer(0) ]],
                device const float *b [[ buffer(1) ]],
                device float *c [[ buffer(2) ]],
                uint gid [[ thread_position_in_grid ]])
{
    c[gid] = a[gid] + b[gid];
}
