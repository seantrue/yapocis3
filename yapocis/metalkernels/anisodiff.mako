/* Perona-Malik filter (Metal) */

<%include file="constants.mako" />
<%include file="helpers.mako" />

/* Perona-Malik filter for 1,2,3,4 channel images*/
%for planes in ["",2,3,4]:

kernel void filterImage${planes}(constant int &width [[ buffer(0) ]],
                                 constant int &height [[ buffer(1) ]],
                                 const device FLOAT${planes} *input [[ buffer(2) ]],
                                 constant float &scale [[ buffer(3) ]],
                                 constant float &step_size [[ buffer(4) ]],
                                 device FLOAT${planes} *output [[ buffer(5) ]],
                                 uint i [[ thread_position_in_grid ]])
{
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
}
%endfor
