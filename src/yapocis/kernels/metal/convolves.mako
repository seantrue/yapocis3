/* Multiple one-dimensional convolutions (Metal) */

<%include file="constants.mako" />

% for name, conv in convs:
<% left = len(conv)//2 %>
kernel void ${name}(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-${left};
    if ((i > ${left}) &&  (i < right)) {
        % for j in range(len(conv)):
        sum = sum + a[i+(${j-left})] * ${conv[j]};
        % endfor
    }
    ret[i] = sum;
}
% endfor
