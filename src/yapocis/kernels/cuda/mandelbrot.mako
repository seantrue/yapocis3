<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Kernel to compute mandelbrot set values for a given set of points */

extern "C" __global__ void mandelbrot(float2 *q, short *output, int const maxiter GLOBAL_SIZE_PARAM)
{
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t gid = get_global_id(0);
    float nreal, real = 0;
    float imag = 0;

    output[gid] = 0;

    for(int curiter = 0; curiter < maxiter; curiter++) {
        nreal = real*real - imag*imag + q[gid].x;
        imag = 2* real*imag + q[gid].y;
        real = nreal;

        if (real*real + imag*imag > 4.0f)
             output[gid] = curiter;
    }
}
