// Address macros
#define ADDRESS(xx,yy) (((xx)*height)+(yy))
#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)

__inline float dxdy2(float2 a, float2 b) {
    float2 dxdy =  (a-b)*(a-b);
    return dxdy.x + dxdy.y;
}

