<%include file="constants.mako"/>
<%include file="helpers.mako"/>

/* Kernel to compute 3x3 median filter on a 2D image */

static __device__ __inline__ float median9(float *a, int n) {
    float l,h;
    bool lt;
    % for i in range(9//2+1):
        % for j in range(i+1, 9):
            lt = a[${j}] < a[${i}];
            l = lt ? a[${j}] : a[${i}];
            h = lt ? a[${i}] : a[${j}];
            a[${i}] = l;
            a[${j}] = h;
        %endfor
    %endfor
    return a[n/2];
}


extern "C" __kernel
void median3x3(int width, int height, float* a, float* ret )
{
    float aa[9];
    size_t i;
    float m;
    size_t left = height + 1;
    size_t right = width - left;
    int h = height;
    i = get_global_id(0);
    m = a[i];
    if ((i >= left) &&  (i < right)) {
		aa[0] =  a[(i-h)-1];
		aa[1] =  a[i-h];
		aa[2] =  a[(i-h)+1];
		aa[3] =  a[i-1];
		aa[4] =  a[i];
		aa[5] =  a[i+1];
		aa[6] =  a[(i+h)-1];
		aa[7] =  a[i+h];
		aa[8] =  a[(i+h)+1];
	    m = median9(&aa[0], 9);
    }
    ret[i] = m;
    return;
}