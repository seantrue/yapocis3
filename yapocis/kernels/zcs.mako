#define Y(i) ((i)/height)
#define X(i) ((i)-(Y(i)*height))

#define ZERO 0.0f
#define ONE 1.0f

%for planes in [1,2,3,4]:
#define ADDRESS${planes}(xx,yy) ((((yy)*height)+(xx))*${planes})

__kernel
void zcs${planes}(int width, int height, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0);
    int x = max((size_t)1,X(i));
    int y = max((size_t)1,Y(i));
    float zclt, zcgt;
    float v, below, left;
    %for plane in range(planes):
    v = input[i];
    below = input[ADDRESS${planes}(x,y-1)];
    left = input[ADDRESS${planes}(x-1,y)];
    zclt = ((below < ZERO || left < ZERO) && v >= ZERO) ? ONE : ZERO;
    zcgt = ((below >= ZERO || left >= ZERO) && v < ZERO) ? ONE : ZERO;
    zcs[i] = max(zclt,zcgt);
    i += 1;
    %endfor
}
%endfor

__kernel
void sub(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] - input2[i];
}
