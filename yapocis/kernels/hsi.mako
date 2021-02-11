//https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
#define eps 1.0e-10

__kernel
void rgb2hsi(__global float *ra, __global float *ga, __global float *ba, __global float *ha, __global float *sa, __global float *ia) {
    size_t i = get_global_id(0);
    float3 c = (float3)(ra[i], ba[i], ga[i]);
    float4 K = (float4)(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f);
    float4 p = c.y < c.z ? (float4)(c.zy,K.wz) : (float4)(c.yz, K.xy);
    float4 q = c.x < p.x ? (float4)(p.xyw, c.x) : (float4)(c.x, p.yzx);
    float d = q.x - min(q.w, q.y);
    float hh = q.z + (q.w - q.y) / (6.0f * d + eps);
    hh = hh < 0.0 ? -hh : hh;
    float3 hsi = (float3)(hh , d / (q.x + eps), q.x);
    hsi = clamp(hsi, 0.0f, 1.0f-eps);
    ha[i] = hsi.x;
    sa[i] = hsi.y;
    ia[i] = hsi.z;
}


__kernel	
void hsi2rgb(__global float *ha, __global float *sa, __global float *ia, __global float *ra, __global float *ga, __global float *ba) {
	size_t i = get_global_id(0);
	float3 hsi = (float3)(ha[i],sa[i],ia[i]);

    float4 K = (float4)(1.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f);
    float3 floorv;
    float3 p = fabs(fract(hsi.xxx + K.xyz, &floorv) * 6.0f - K.www);
    float3 c =  hsi.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), hsi.y);
    ra[i] = c.x;
    ga[i] = c.y;
    ba[i] = c.z;
}
