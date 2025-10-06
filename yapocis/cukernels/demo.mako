<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Test kernel that sums elements of two arrays into a third array. */

__kernel void sum(const float *a, const float *b, float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
