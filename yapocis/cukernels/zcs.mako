<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Kernels to find zero-crossings in an image. */

/* Find zero-crossings for pixels of 1-4 channels. */
%for planes in [1,2,3,4]:
#define ADDRESS${planes}(xx,yy) (ADDRESS(xx,yy))*${planes}

__kernel
void zcs${planes}(int width, int height, float threshold, float* input, float* zcs )
{
    size_t i = get_global_id(0)*${planes};
    int x = max((size_t)1,X(i));
    x = min(x,width-2);
    int y = max((size_t)1,Y(i));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    %for plane in range(planes):
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    %endfor
}
%endfor

__kernel
void sub(float* input1, float* input2, float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] - input2[i];
}
