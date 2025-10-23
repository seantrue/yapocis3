<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Multiple one-dimensional convolutions given names and kernels at compile time */

% for name, conv in convs:
<% left = len(conv)//2 %>
extern "C" __kernel
void ${name}(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-${left};
    i = get_global_id(0);
    sum = 0.0;
    if ((i > ${left}) &&  (i < right)) {
	%for j in range(len(conv)):
       sum = sum + a[i+(${j-left})] * ${conv[j]};
    %endfor
    }
    ret[i] = sum;
    return;
}
% endfor