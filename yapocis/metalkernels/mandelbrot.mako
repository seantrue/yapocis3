/* Metal kernel to compute mandelbrot set values for a given set of points */

<%include file="constants.mako" />

kernel void mandelbrot(device float2 *q [[ buffer(0) ]],
                       device short *output [[ buffer(1) ]],
                       constant int &maxiter [[ buffer(2) ]],
                       uint i [[ thread_position_in_grid ]])
{
    float nreal, real = 0.0f;
    float imag = 0.0f;

    output[i] = 0;

    for (int curiter = 0; curiter < maxiter; curiter++) {
        nreal = real*real - imag*imag + q[i].x;
        imag = 2.0f * real*imag + q[i].y;
        real = nreal;

        if (real*real + imag*imag > 4.0f)
             output[i] = (short)curiter;
    }
}
