/* One dimensional convolution (Metal) with compile-time kernel constants */

<%include file="constants.mako" />

<% left = len(conv)//2 %>
#define FAST 1000

kernel void ${name}(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint gid [[ thread_position_in_grid ]])
{
    size_t i = gid;
    float sum = 0.0f;
    size_t right = width-${left};
    if ((i > (size_t)${left}) &&  (i < right)) {
        % for j in range(len(conv)):
        sum = sum + a[i+(${j-left})] * ${conv[j]};
        % endfor
    }
    ret[i] = sum;
}

kernel void ${name}_fast(constant int &width [[ buffer(0) ]],
                         device float* a [[ buffer(1) ]],
                         device float* ret [[ buffer(2) ]],
                         constant int &chunk [[ buffer(3) ]],
                         uint gid [[ thread_position_in_grid ]])
{
    size_t i = gid * (size_t)chunk;
    float sum = 0.0f;
    size_t right = width-${left};
    threadgroup float buffer[FAST];
    for (int j = 0; j < ${len(conv)}; j++) {
        buffer[j] = i+j >= (size_t)width ? 0.0f : a[i+j];
    }
    for (int j = 0; j < chunk-${len(conv)}; j++) {
        sum = 0.0f;
        size_t ipixel = i+j >= right ? (right-1) : (i+j);
        if ((i > (size_t)${left}) &&  (i < right)) {
            % for k in range(len(conv)):
            sum = sum + buffer[${k}] * ${conv[k]};
            % endfor
        }
        ret[ipixel] = sum;
    }
}
