/*
Assume row major storage.
addr(x,y) = (x*height)+y
x = addr/height
y = addr - (addr/height)
*/

#define ADDRESS(xx,yy) (((xx)*height)+(yy))
#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)


#define swap(xxx,yyy) {int tmp = xxx; xxx=yyy; yyy=tmp;}
#define plot(x,y) a[ADDRESS((x),(y))] = color
__kernel
void draw(__global int4 *lines, int width, int height,  __global float *a, float color)
{
    size_t i =  get_global_id(0);
    int4 line = lines[i];
    int x0 = line.s0;
    int y0 = line.s1;
    int x1 = line.s2;
    int y1 = line.s3;
    int steep =  abs(y1-y0) > abs(x1-x0);
    if (steep) {
       swap(x0, y0);
       swap(x1, y1);
    }
    if (x0>x1) {
        swap(x0, x1);
	swap(y0, y1);
    }
    int dx=x1-x0;
    int dy=abs(y1-y0);
    int e=dx/2;
    int y=y0;
    int ystep;
    if (y0 < y1)
       ystep = 1;
    else
       ystep = -1;
    int x;
    for (x=x0; x <= x1; x++) {
    if (steep)
       plot(y,x);
    else
       plot(x,y);
   e -= dy;
   if (e < 0) {
      y = y + ystep;
      e += dx;
      }
   }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

