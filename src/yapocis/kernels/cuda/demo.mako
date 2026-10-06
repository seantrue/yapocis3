<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Test kernel that sums elements of two arrays into a third array. */

extern "C" __kernel void sum(const float *a, const float *b, float *c GLOBAL_SIZE_PARAM)
    {
      RETURN_IF_OUTSIDE(get_global_id(0))
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
