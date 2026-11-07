/* Kernel to compute mandelbrot set values for a given set of points */

__kernel void mandelbrot(__global float2 *q, __global short *output, int const maxiter)
{
    size_t gid = get_global_id(0);
    float nreal, real = 0;
    float imag = 0;

    output[gid] = 0;

    for(int curiter = 0; curiter < maxiter; curiter++) {
        nreal = real*real - imag*imag + q[gid].x;
        imag = 2* real*imag + q[gid].y;
        real = nreal;

        if (real*real + imag*imag > 4.0f) {
            // The first escape, as the numba port records it. An escape at iteration 0 reads
            // as 0 ("inside"), so that point keeps going, as the port's out == 0 test lets it.
            output[gid] = curiter;
            if (curiter > 0) break;
        }
    }
}
