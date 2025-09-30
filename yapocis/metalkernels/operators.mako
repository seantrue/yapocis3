/* Metal kernels to implement pixel-wise operations on images */

<%include file="constants.mako" />

% for name, operator in operators:
kernel void ${name}(const device float* input1 [[ buffer(0) ]],
                    const device float* input2 [[ buffer(1) ]],
                    device float *output [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    output[i] = input1[i] ${operator} input2[i];
}
% endfor
