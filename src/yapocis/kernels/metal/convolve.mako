/* One dimensional convolution (Metal) with compile-time kernel constants */

<%include file="constants.mako" />

<% left = len(conv)//2 %>

kernel void ${name}(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-${left};
    if ((i > (size_t)${left}) &&  (i < right)) {
        % for j in range(len(conv)):
        sum = sum + a[i+(${j-left})] * ${conv[j]};
        % endfor
    }
    ret[i] = sum;
}

/* Thread g computes ret[g*chunk .. g*chunk+chunk-1], each pixel exactly as ${name}
   computes it, and stops at width. Needs a global size of at least ceil(width/chunk);
   a larger one (the default, one thread per element) is correct but idles. */
kernel void ${name}_fast(constant int &width [[ buffer(0) ]],
                         device float* a [[ buffer(1) ]],
                         device float* ret [[ buffer(2) ]],
                         constant int &chunk [[ buffer(3) ]],
                         uint g [[ thread_position_in_grid ]])
{
    int n = chunk > 0 ? chunk : 1;
    size_t right = width-${left};
    size_t start = (size_t)g * (size_t)n;
    for (int j = 0; j < n; j++) {
        size_t i = start + (size_t)j;
        if (i >= (size_t)width) return;
        float sum = 0.0f;
        if ((i > (size_t)${left}) &&  (i < right)) {
            % for k in range(len(conv)):
            sum = sum + a[i+(${k-left})] * ${conv[k]};
            % endfor
        }
        ret[i] = sum;
    }
}
