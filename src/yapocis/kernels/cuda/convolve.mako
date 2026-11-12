<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* One dimensional convolution with a kernel conv that is determined at compile time */

<% left = len(conv)//2 %>
extern "C" __kernel
void ${name}(int width,  float* a, float* ret )
{
    size_t i = get_global_id(0);
    if (i >= (size_t)width) return;
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

/* Work item g computes ret[g*chunk .. g*chunk+chunk-1], each pixel exactly as ${name}
   computes it, and stops at width. Needs a global size of at least ceil(width/chunk). */
extern "C" __kernel void ${name}_fast(int width,  float* a, float* ret, int chunk)
{
    int n = chunk > 0 ? chunk : 1;
    size_t right = width-${left};
    size_t start = (size_t)get_global_id(0) * (size_t)n;
    for (int j = 0; j < n; j++) {
        size_t i = start + (size_t)j;
        if (i >= (size_t)width) return;
        float sum = 0.0;
        if ((i > (size_t)${left}) &&  (i < right)) {
        %for k in range(len(conv)):
           sum = sum + a[i+(${k-left})] * ${conv[k]};
        %endfor
        }
        ret[i] = sum;
    }
    return;
}
