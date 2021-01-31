#define ADDRESS(xx,yy) (((yy)*height)+(xx))
#define Y(i) ((i)/height)
#define X(i) ((i)-(Y(i)*height))

__kernel
void zcs(int width, int height, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0);
    int x = max((size_t)1,X(i));
    int y = max((size_t)1,Y(i));
    float zclt, zcgt;
    float v, below, left;
    v = input[i];
    below = input[ADDRESS(x,y-1)];
    left = input[ADDRESS(x-1,y)];
    zclt = ((below < 0.0 || left < 0.0) && v >= 0.0) ? 1.0 : 0.0;
    zcgt = ((below >= 0.0 || left >= 0.0) && v < 0.0) ? 1.0 : 0.0;
    zcs[i] = max(zclt,zcgt);
}
