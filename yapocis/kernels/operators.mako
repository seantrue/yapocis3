%for name, operator in operators:
__kernel
void ${name}(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] ${operator} input2[i];
}
%endfor
