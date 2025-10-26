/* Metal kernels to find zero-crossings in an image. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Find zero-crossings for pixels of 1-4 channels. */
%for planes in [1,2,3,4]:
#define ADDRESS${planes}(xx,yy) (ADDRESS(xx,yy))*${planes}

kernel void zcs${planes}(constant int &width [[ buffer(0) ]],
                         constant int &height [[ buffer(1) ]],
                         constant float &threshold [[ buffer(2) ]],
                         const device float* input [[ buffer(3) ]],
                         device float* zcs [[ buffer(4) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i*${planes};
    int x = max((size_t)1,X(idx));
    x = min(x,width-2);
    int y = max((size_t)1,Y(idx));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    %for plane in range(planes):
    v = input[idx];
    above = input[ADDRESS${planes}(x,y+1)];
    below = input[ADDRESS${planes}(x,y-1)];
    left = input[ADDRESS${planes}(x-1,y)];
    right = input[ADDRESS${planes}(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    %endfor
}
%endfor

kernel void sub(device float* input1 [[ buffer(0) ]],
                device float* input2 [[ buffer(1) ]],
                device float *output [[ buffer(2) ]],
                uint i [[ thread_position_in_grid ]])
{
    output[i] = input1[i] - input2[i];
}
