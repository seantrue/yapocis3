/*
Assume row major storage.
addr(x,y) = (x*height)+y
x = addr/height
y = addr - (addr/height)
*/

#define ADDRESS(xx,yy) (((xx)*height)+(yy))
#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)

__kernel
void gradient(int width, int height, __global float* input, int reach, __global float* grad, __global float *theta )
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    float gr = 0.0f;
    float th = 0.0f;
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
       float dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
       float dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
    gr = sqrt(dx*dx+dy*dy);
    th = atan2pi(dy, dx);
    //th = dy < 0 ? -1.0f+th : th;
    }
    grad[i] = isnan(gr) ? 0.0f: gr;
    theta[i] = isnan(th) ? 0.0f: th;
}
