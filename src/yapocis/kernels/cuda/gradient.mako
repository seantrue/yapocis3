<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Kernels to compute direction and magnitude of gradient of an image. */

/* Compute gradient for images with 1,2,3,4 channels per pixel. */

%for planes in ["",2,3,4]:
extern "C" __kernel
void gradient${planes}(int width, int height, FLOAT${planes}* input, int reach, FLOAT${planes}* grad, FLOAT${planes} *theta )
{
    size_t i = get_global_id(0);
    if (i >= (size_t)width * height) return;
    int x = X(i);
    int y = Y(i);
    FLOAT${planes} gr = ZERO${planes};
    FLOAT${planes} th = ZERO${planes};
    // CUDA X(i) runs along height and Y(i) along width (helpers.mako)
    if (0 <= (x-reach)  && (x+reach) < height && 0 <= (y-reach) && (y+reach) < width) {
        FLOAT${planes} dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT${planes} dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        // Here x is the column and y the row, the reverse of OCL/Metal, whose
        // atan2pi(dy, dx) is atan2(d/dcol, d/drow): swap the arguments to match.
        th = atan2pi(dx, dy);
    }
    grad[i] = isnan(gr) ? ZERO${planes}: gr;
    theta[i] = isnan(th) ? ZERO${planes}: th;
}
%endfor