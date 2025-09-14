/* Kernels to implement pixel-wise operations on images */

%for name, operator in operators:
__kernel
void ${name}(__global const float* restrict input1,
             __global const float* restrict input2,
             __global float* restrict output)
{
    size_t i = get_global_id(0);
    output[i] = input1[i] ${operator} input2[i];
}
%endfor
