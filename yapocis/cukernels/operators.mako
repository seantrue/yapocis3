<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Kernels to implement pixel-wise operations on images */

%for name, operator in operators:
__kernel
void ${name}(float* input1, float* input2, float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] ${operator} input2[i];
}
%endfor
