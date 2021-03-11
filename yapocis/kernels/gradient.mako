
<%include file="constants.mako" />
<%include file="helpers.mako" />


%for planes in ["",2,3,4]:
__kernel
void gradient${planes}(int width, int height, __global FLOAT${planes}* input, int reach, __global FLOAT${planes}* grad, __global FLOAT${planes} *theta )
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    FLOAT${planes} gr = ZERO${planes};
    FLOAT${planes} th = ZERO${planes};
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT${planes} dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT${planes} dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = isnan(gr) ? ZERO${planes}: gr;
    theta[i] = isnan(th) ? ZERO${planes}: th;
}
%endfor