/* Separable convolution kernels for OpenCL, with kernel names and constants determined at compile time */
% for name, conv in convs:
<% margin = len(conv)//2 %>
__kernel
void ${name}(int vertical, int width, int height,
             __global const float* restrict a,
             __global float* restrict ret)
{
    size_t i = get_global_id(0);
    size_t n = width*height;
    float sum = 0.0;
    if (!vertical && ((size_t)(${margin}*height) < i) && (i < (n-${margin}*height))){
       // Horizontal convolution
  	%for j in range(len(conv)):
		sum = sum + a[i+(${j-margin}*height)] * ${conv[j]};
	%endfor
    }
    if (vertical && (${margin} < i) && (i < (n-${margin}))) {
       // Vertical convolution
  	%for j in range(len(conv)):
		sum = sum + a[i+(${j-margin})] * ${conv[j]};
	%endfor
    }
    ret[i] = sum;
    return;
}
__kernel
void ${name}_h(int width, int height,
               __global const float* restrict a,
               __global float* restrict ret)
{
    size_t i = get_global_id(0);
    size_t n = width*height;
    float sum = 0.0;

    // Horizontal convolution
    int inside = (((size_t)(${margin}*height) < i) && (i < (n-${margin}*height)));
  	%for j in range(len(conv)):
    sum += inside ? a[i+(${j-margin}*height)] * ${conv[j]} : 0.0f;
	%endfor
    ret[i] = sum;
    return;
}
__kernel
void ${name}_v(int width, int height,
               __global const float* restrict a,
               __global float* restrict ret)
{
    size_t i = get_global_id(0);
    size_t n = width*height;

    // Vertical convolution
    float sum = 0.0;
    int inside = (${margin} < i) && (i < (n-${margin}));
  	%for j in range(len(conv)):
    sum +=  inside ?  a[i+(${j-margin})] * ${conv[j]} : 0.0;
	%endfor

    ret[i] = sum;
    return;
}
% endfor
