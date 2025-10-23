/* One dimensional convolution (Metal) with compile-time kernel constants */

<%include file="constants.mako" />

<% left = len(conv)//2 %>
#define FAST 1000

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

kernel void ${name}_fast(constant int &width [[ buffer(0) ]],
                         device float* a [[ buffer(1) ]],
                         device float* ret [[ buffer(2) ]],
                         constant int &chunk [[ buffer(3) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i * (size_t)chunk;
    float sum = 0.0f;
    size_t right = width-${left};
    threadgroup float buffer[FAST];
    for (int j = 0; j < ${len(conv)}; j++) {
        buffer[j] = idx+j >= (size_t)width ? 0.0f : a[idx+j];
    }
    for (int j = 0; j < chunk-${len(conv)}; j++) {
        sum = 0.0f;
        size_t ipixel = idx+j >= right ? (right-1) : (idx+j);
        if ((idx > (size_t)${left}) &&  (idx < right)) {
            % for k in range(len(conv)):
            sum = sum + buffer[${k}] * ${conv[k]};
            % endfor
        }
        ret[ipixel] = sum;
    }
}
