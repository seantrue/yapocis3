% for name, conv in convs:
<% margin = len(conv)//2 %>
__kernel
void ${name}(int vertical, int width, int height, __global float* a, __global float* ret )
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
% endfor