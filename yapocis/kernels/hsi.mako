//https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
#define eps 1.0e-10


#define RGBHSIK ((float4)(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f))
__inline float3 _rgb2hsi(float3 c) {
    float4 p = c.y < c.z ? (float4)(c.zy,RGBHSIK.wz) : (float4)(c.yz, RGBHSIK.xy);
    float4 q = c.x < p.x ? (float4)(p.xyw, c.x) : (float4)(c.x, p.yzx);
    float d = q.x - min(q.w, q.y);
    float hh = q.z + (q.w - q.y) / (6.0f * d + eps);
    hh = hh < 0.0 ? -hh : hh;
    float3 hsi = (float3)(hh , d / (q.x + eps), q.x);
    hsi = clamp(hsi, 0.0f, 1.0f-eps);
    return hsi;
}

#define HSIRGBK ((float4)(1.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f))
__inline float3 _hsi2rgb(float3 hsi) {
    float3 floorv;
    float3 p = fabs(fract(hsi.xxx + HSIRGBK.xyz, &floorv) * 6.0f - HSIRGBK.www);
    float3 c =  hsi.z * mix(HSIRGBK.xxx, clamp(p - HSIRGBK.xxx, 0.0f, 1.0f), hsi.y);
    return c;
}

__kernel
void rgb2hsi(__global float *ra, __global float *ga, __global float *ba, __global float *ha, __global float *sa, __global float *ia) {
    size_t i = get_global_id(0);
    float3 c = (float3)(ra[i], ba[i], ga[i]);
    float3 hsi = _rgb2hsi(c);
    ha[i] = hsi.x;
    sa[i] = hsi.y;
    ia[i] = hsi.z;
}


__kernel	
void hsi2rgb(__global float *ha, __global float *sa, __global float *ia, __global float *ra, __global float *ga, __global float *ba) {
	size_t i = get_global_id(0);
	float3 hsi = (float3)(ha[i],sa[i],ia[i]);
    float3 c = _hsi2rgb(hsi);
    ra[i] = c.x;
    ga[i] = c.y;
    ba[i] = c.z;
}



__kernel
void rgb2hsi3(int depth, __global float *input, __global float *output) {
    size_t i = get_global_id(0) * depth;
    float3 c = (float3)(input[i], input[i+2],input[i+1]);
    float3 hsi = _rgb2hsi(c);
    output[i] =   hsi.x;
    output[i+1] = hsi.y;
    output[i+2]=  hsi.z;
}


__kernel
void hsi2rgb3(int depth, __global float *input, __global float *output) {
	size_t i = get_global_id(0)*depth;
	float3 hsi = (float3)(input[i],input[i+1],input[i+2]);
    float3 c = _hsi2rgb(hsi);
    output[i] =   c.x;
    output[i+1] = c.y;
    output[i+2] = c.z;
}
