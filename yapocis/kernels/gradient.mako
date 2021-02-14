#define ADDRESS(xx,yy) (((xx)*height)+(yy))
#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)

#define ZERO 0.0f
#define ZERO2 (float2)(ZERO,ZERO)
#define ZERO3 (float3)(ZERO,ZERO,ZERO)
#define ZERO4 (float4)(ZERO,ZERO,ZERO,ZERO)
#define ONE 1.0f
#define ONE2 (float2)(ONE,ONE)
#define ONE3 (float3)(ONE,ONE,ONE)
#define ONE4 (float4)(ONE,ONE,ONE,ONE)

%for planes in ["",2,3,4]:
#define ZEROS ZERO${planes}
#define ONES ONE${planes}
#define FLOAT float${planes}    
__kernel
void gradient${planes}(int width, int height, __global float${planes}* input, int reach, __global float${planes}* grad, __global float${planes} *theta )
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    FLOAT gr = ZEROS;
    FLOAT th = ZEROS;
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = isnan(gr) ? ZEROS: gr;
    theta[i] = isnan(th) ? ZEROS: th;
}
%endfor