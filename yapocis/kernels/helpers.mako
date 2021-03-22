// Address macros
#define ADDRESS(xx,yy) (((xx)*height)+(yy))

#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)

#define SWAPINT(xxx,yyy) {int tmp = xxx; xxx=yyy; yyy=tmp;}
#define SWAPFLOAT(xxx,yyy) {float tmp = xxx; xxx=yyy; yyy=tmp;}

#define ABS(x) (((x) < 0) ? -(x) : (x))

__inline float dxdy2(float2 a, float2 b) {
    float2 dxdy =  (a-b)*(a-b);
    return dxdy.x + dxdy.y;
}

__inline float dot3(float3 a, float3 b) {
   return (float) (a.x*b.x + a.y*b.y + a.z*b.z);
}

__inline float3 vxm3(float3 v, float3 a, float3 b, float3 c) {
    float3 result = (float3) (dot3(v,a), dot3(v,b), dot3(v,c));
    return result;
}

__inline void p3(__constant char *s, float3 v) {
    printf("%s %.5f %.5f %.5f\n", s, v.x, v.y, v.z);
}

__inline void p4(__constant char *s, float4 v) {
    printf("%s %.5f %.5f %.5f %.5f\n", s, v.x, v.y, v.z, v.w);
}

#define ANYISLESS(a,b) (a.x < b.x || a.y < b.y)
#define ANYISGTE(a,b) (a.x >= b.x || a.y >= b.y)