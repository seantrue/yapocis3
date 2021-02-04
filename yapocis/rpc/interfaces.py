'''\
py
Created on Aug 12, 2011
Copyright (c) 2011, Sean D. True
'''
#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

# Standard demo, interface spec supports c=sum(a,b)
demo = """
interface demo {
    kernel sum(in float *a, in float *b, outlike a);
    };
"""

# Test addressing calculation
xy = """
    interface xy {
        kernel addr(widthof a, heightof a, in int *a, outlike a, outlike a, outlike a, outlike a);
    };
"""

# 1D convolution kernel, with parameterized name
convolve = """
    interface convolve {
        kernel ${name}(sizeof a, in float* a, outlike a);
    };
"""

# Multiple 1D kernels, with name and conv (mask) parameterized
convolves = """
    interface convolves {
%for name,conv in convs:
        kernel ${name}(sizeof a, in float* a, outlike a);
%endfor 
    };
"""
# Separable 2-D kernels, with name and conv (mask) parameterized
convolvesep = """
    interface convolvesep {
%for name,conv in convs:
        kernel ${name}(in int vertical, widthof input, heightof input, in float* input, outlike input);
        alias ${name}_res as ${name}(in int vertical, widthof input, heightof input, resident float* input, resident float *output);
%endfor 
    };
"""
# Array operations
operators = """
    interface operators {
%for name, operator in operators:
    kernel ${name}(in float *input1, in float *input2, outlike input1);
    alias ${name}_res as ${name}(resident float *input1, resident float *input2, resident float *output);
%endfor
    };
"""

# Zero crossing
zcs = """
    interface zcs {
        kernel zcs(widthof int input, heightof int input, in float *input, outlike input);
        alias zcs_res as zcs(widthof int input, heightof int input, resident float *input, resident float *zcs);
    };
"""

# 2-D median filter, with support for fast iteration
median3x3 = """
    interface median3x3 {
        kernel median3x3(sizeof int a, heightof int a, in float* a, outlike a );
        alias first as median3x3(sizeof int a, heightof int a, in float* a, in float* ret );
        alias step as median3x3(sizeof int a, heightof int a, resident float* a, resident float* ret );
        alias last as median3x3(sizeof int a, heightof int a, resident float* a, out float* ret );
    };
"""

# Conversion to and from Hue/Saturation/Intensity color space
hsi = """
    interface hsi {
        kernel rgb2hsi(in float *r, in float *g, in float *b, outlike r, outlike r, outlike r);
        kernel hsi2rgb(in float *h, in float *s, in float *i, outlike h, outlike h, outlike h);
    };
"""

# Standard Mandelbrot kernel with return value, use as:
# output = mandelbrot(input, iterations)
mandelbrot = """
    interface mandelbrot {
        kernel mandelbrot(in complex64 *q, outlike short *q, in int maxiter);
};
"""

gradient = """
    interface gradient {
        kernel gradient(widthof int input, heightof int input, in float* input, in int reach,  outlike input, outlike input );
        alias gradient_res as gradient(widthof int input, heightof int input, resident float* input, in int reach, resident float *grad, resident float *theta );
    };
"""

peronamalik = """
    interface peronamalik {
        kernel filterImage(widthof input, heightof input, in float *input, in float scale, in float size_step, outlike input);
        alias filterImage_res as filterImage(widthof input, heightof input, resident float *input, in float scale, in float size_step, resident float *output);
        };
"""
