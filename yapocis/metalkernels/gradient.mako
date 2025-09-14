/* Metal kernels to compute direction and magnitude of gradient of an image. */

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Compute gradient for images with 1,2,3,4 channels per pixel. */

%for planes in ["",2,3,4]:
kernel void gradient${planes}(constant int &width [[ buffer(0) ]],
                              constant int &height [[ buffer(1) ]],
                              device FLOAT${planes}* input [[ buffer(2) ]],
                              constant int &reach [[ buffer(3) ]],
                              device FLOAT${planes}* grad [[ buffer(4) ]],
                              device FLOAT${planes}* theta [[ buffer(5) ]],
                              uint gid [[ thread_position_in_grid ]])
{
    size_t i = gid;
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
    // Replace NaNs component-wise with zero using Metal's select(a,b,cond)
    // select(a, b, cond) returns cond ? b : a, works for scalar and vector types
    grad[i] = select(gr, ZERO${planes}, isnan(gr));
    theta[i] = select(th, ZERO${planes}, isnan(th));
}
%endfor
