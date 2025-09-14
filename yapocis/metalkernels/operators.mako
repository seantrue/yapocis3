/* Metal kernels to implement pixel-wise operations on images */

<%include file="constants.mako" />

% for name, operator in operators:
kernel void ${name}(device float* input1 [[ buffer(0) ]],
                    device float* input2 [[ buffer(1) ]],
                    device float *output [[ buffer(2) ]],
                    uint gid [[ thread_position_in_grid ]])
{
    output[gid] = input1[gid] ${operator} input2[gid];
}
% endfor
