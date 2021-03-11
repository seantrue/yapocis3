/* Perona-Malik filter */

<%include file="constants.mako" />
<%include file="helpers.mako" />

%for planes in ["",2,3,4]:

__kernel 
void filterImage${planes}(int width, int height, __global FLOAT${planes} *input, float scale, float step_size, __global FLOAT${planes} *output)
{
    size_t i = get_global_id(0);
    size_t last = width*height;
    int x = X(i);
    int y = Y(i);
    FLOAT${planes} CF;
    FLOAT${planes} central = input[i];
    FLOAT${planes} invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT${planes} n = input[i+1];
      FLOAT${planes} s = input[i-1];
      FLOAT${planes} e = input[i-height];
      FLOAT${planes} w = input[i+height];
      FLOAT${planes} di = n-central;
      CF = di*invscale;
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      FLOAT${planes} accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONE${planes}/(ONE${planes} + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
    barrier(CLK_GLOBAL_MEM_FENCE);
}
%endfor