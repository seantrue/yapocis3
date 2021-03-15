//https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
<%include file="constants.mako" />
<%include file="helpers.mako" />

#define eps 1.0e-10

# For color math, see: http://www.brucelindbloom.com/index.html?Math.html
# For white points, see: https://en.wikipedia.org/wiki/Standard_illuminant

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



## def rgb2lab(inputColor):
##
##     num = 0
##     RGB = [0, 0, 0]
##
##     for value in inputColor:
##         value = float(value) / 255
##
##         if value > 0.04045:
##             value = ((value + 0.055) / 1.055) ** 2.4
##         else:
##             value = value / 12.92
##
##         RGB[num] = value * 100
##         num = num + 1
##
##     XYZ = [0, 0, 0, ]
##
##     X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
##     Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
##     Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
##     XYZ[0] = round(X, 4)
##     XYZ[1] = round(Y, 4)
##     XYZ[2] = round(Z, 4)
##
##     # Observer= 2°, Illuminant= D65
##     XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
##     XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
##     XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883
##
##     num = 0
##     for value in XYZ:
##
##         if value > 0.008856:
##             value = value ** (0.3333333333333333)
##         else:
##             value = (7.787 * value) + (16 / 116)
##
##         XYZ[num] = value
##         num = num + 1
##
##     Lab = [0, 0, 0]
##
##     L = (116 * XYZ[1]) - 16
##     a = 500 * (XYZ[0] - XYZ[1])
##     b = 200 * (XYZ[1] - XYZ[2])
##
##     Lab[0] = round(L, 4)
##     Lab[1] = round(a, 4)
##     Lab[2] = round(b, 4)
##
##     return Lab

#define XK (float3)(.4124,.3576,.1805)
#define YK (float3)(.2126,.7152,.0722)
#define ZK (float3)(.0193,.1192,.9505)
#define REFERENCE_WHITE (float3)(95.047,100.0,108.883)

__inline adjustrgb(float v) {
    v = v <= .04045f ? v/12.92f : native_powr((v + 0.055f) / 1.055f, 2.4f);
    return v*100.0;
}

__inline float3 _rgb2xyz(float3 rgb) {
   float3 rgb3 = ZERO3;
   rgb3[0] = adjustrgb(rgb.x);
   rgb3[1] = adjustrgb(rgb.y);
   rgb3[2] = adjustrgb(rgb.z);
   float X = dot3(rgb3,XK);
   float Y = dot3(rgb3,YK);
   float Z = dot3(rgb3,ZK);
   float3 XYZ = (float3)(X,Y,Z);
   XYZ /= REFERENCE_WHITE;
   return XYZ;
}

__inline float adjustxyz(float v) {
    v = v <= .008856f ? (7.787f * v) + (16.f / 116.f) : native_powr(v, 0.3333333333333333f);
    return v;
}

__inline float3 _xyz2lab(float3 xyz) {
   float3 xyz3 = ZERO3;
   xyz3[0] = adjustxyz(xyz.x);
   xyz3[1] = adjustxyz(xyz.y);
   xyz3[2] = adjustxyz(xyz.z);
   float3 lab = ZERO3;
   lab.x = (116*xyz3.y) -16;
   lab.y = 500 * (xyz3.x - xyz3.y);
   lab.z = 200 * (xyz3.y-xyz3.z);
   return lab;
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

__kernel
void rgb2lab3(int depth, __global float *input, __global float *output) {
    size_t i = get_global_id(0) * depth;
    float3 c = (float3)(input[i], input[i+2],input[i+1]);
    float3 xyz = _rgb2xyz(c);
    float3 lab = _xyz2lab(xyz);
    output[i] =   lab.x;
    output[i+1] = lab.y;
    output[i+2]=  lab.z;
}


__kernel
void lab2rgb3(int depth, __global float *input, __global float *output) {
	size_t i = get_global_id(0)*depth;
	float3 lab = (float3)(input[i],input[i+1],input[i+2]);
    float3 c = _lab2rgb(lab);
    output[i] =   c.x;
    output[i+1] = c.y;
    output[i+2] = c.z;
}
