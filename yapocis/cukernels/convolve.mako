<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* One dimensional convolution with a kernel conv that is determined at compile time */

<% left = len(conv)//2 %>
#define FAST 1000
__kernel
void ${name}(int width,  float* a, float* ret )
{
    size_t i = get_global_id(0);
    float sum;
    size_t right = width-${left};
    sum = 0.0;
    if ((i > (size_t)${left}) &&  (i < right)) {
	%for j in range(len(conv)):
       sum = sum + a[i+(${j-left})] * ${conv[j]};
    %endfor
    }
    ret[i] = sum;
    return;
}

__kernel void ${name}_fast(int width,  float* a, float* ret, int chunk)
{
    size_t i = get_global_id(0) * chunk;
    float sum;
    size_t right = width-${left};
    volatile __local float buffer[FAST];
    for (int j = 0; j < ${len(conv)}; j++) {
        buffer[j] = i+j >= width ? 0.0 : a[i+j];
    }
    for (int j = 0; j < chunk-${len(conv)}; j++) {
        sum = 0.0;
        int ipixel = i+j >= right ? right-1 : i+j;
        if ((i > (size_t)${left}) &&  (i < right)) {
        %for j in range(len(conv)):
           sum = sum + a[j] * ${conv[j]};
        %endfor
        }
        ret[ipixel] = sum;
    }
    return;
}
