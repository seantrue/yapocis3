/* Separable convolution kernels (Metal), names and constants determined at compile time */

<%include file="constants.mako" />

% for name, conv in convs:
<% margin = len(conv)//2 %>
kernel void ${name}(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint gid [[ thread_position_in_grid ]])
{
    size_t i = gid;
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(${margin}*height) < i) && (i < (n-${margin}*height))){
       // Horizontal convolution
       % for j in range(len(conv)):
       sum = sum + a[i+(${j-margin}*height)] * ${conv[j]};
       % endfor
    }
    if (vertical && (${margin} < i) && (i < (n-${margin}))) {
       // Vertical convolution
       % for j in range(len(conv)):
       sum = sum + a[i+(${j-margin})] * ${conv[j]};
       % endfor
    }
    ret[i] = sum;
}

kernel void ${name}_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint gid [[ thread_position_in_grid ]])
{
    size_t i = gid;
    size_t n = width*height;
    float sum = 0.0f;
    int inside = (((size_t)(${margin}*height) < i) && (i < (n-${margin}*height)));
    % for j in range(len(conv)):
    sum += inside ? a[i+(${j-margin}*height)] * ${conv[j]} : 0.0f;
    % endfor
    ret[i] = sum;
}

kernel void ${name}_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint gid [[ thread_position_in_grid ]])
{
    size_t i = gid;
    size_t n = width*height;
    float sum = 0.0f;
    int inside = (${margin} < i) && (i < (n-${margin}));
    % for j in range(len(conv)):
    sum += inside ? a[i+(${j-margin})] * ${conv[j]} : 0.0f;
    % endfor
    ret[i] = sum;
}
% endfor
