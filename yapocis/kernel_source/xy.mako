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
void addr(int width, int height, __global int* input1, __global int* ad, __global int* x, __global int* y, __global int *xy)
{
    size_t i = get_global_id(0);
    unsigned int x1 = X(i);
    unsigned int y1 = Y(i);
    size_t j = ADDRESS(x1,y1);
    size_t white = ADDRESS(width,height)-1;
    ad[j] = i;
    x[j] = x1;
    y[j] = y1;
    xy[j] = i;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if ((x1&63)== 0) {
       xy[j] = 0;
       }
    barrier(CLK_GLOBAL_MEM_FENCE);
    int dx = 32;
    if ((x1&127)==0 && (y1&127)==0) {
       for(int k = 0; k < dx; k++) {
       	       size_t addr1 = ADDRESS(x1+k,y1+k);
	       if (addr1 <= white)
	           xy[addr1] = white;
	       addr1 = ADDRESS(x1+k, y1-k);
	       if (addr1 <= white)
       	           xy[addr1]  = white;
	       addr1 = ADDRESS(x1-k, y1+k);
	       if (addr1 <= white)
       	           xy[addr1]  = white;
	       addr1 = ADDRESS(x1-k, y1-k);
	       if (addr1 <= white)
       	           xy[addr1]  = white;
	
	}
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
