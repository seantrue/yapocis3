<%include file="constants.mako"/>
#define DOT3
<%include file="helpers.mako"/>

// https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
// For color math, see: http://www.brucelindbloom.com/index.html?Math.html
// For white points, see: https://en.wikipedia.org/wiki/Standard_illuminant

#define eps 1.0e-10


#define RGBHSIK make_float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f)
__device__ __inline__ float3 _rgb2hsi(float3 c) {
    float4 RGBHSIK_val = RGBHSIK;
    float4 p = c.y < c.z ? make_float4(c.z, c.y, RGBHSIK_val.w, RGBHSIK_val.z) : make_float4(c.y, c.z, RGBHSIK_val.x, RGBHSIK_val.y);
    float4 q = c.x < p.x ? make_float4(p.x, p.y, p.w, c.x) : make_float4(c.x, p.y, p.z, p.x);
    float d = q.x - min(q.w, q.y);
    float hh = q.z + (q.w - q.y) / (6.0f * d + eps);
    hh = hh < 0.0 ? -hh : hh;
    float3 hsi = make_float3(hh , d / (q.x + eps), q.x);
    hsi = clamp(hsi, 0.0f, 1.0f-eps);
    return hsi;
}

#define HSIRGBK make_float4(1.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f)
__device__ __inline__ float3 _hsi2rgb(float3 hsi) {
    float3 floorv;
    float4 HSIRGBK_val = HSIRGBK;
    float3 p = fabs(fract(make_float3(hsi.x, hsi.x, hsi.x) + make_float3(HSIRGBK_val.x, HSIRGBK_val.y, HSIRGBK_val.z), &floorv) * 6.0f - make_float3(HSIRGBK_val.w, HSIRGBK_val.w, HSIRGBK_val.w));
    float3 c =  hsi.z * mix(make_float3(HSIRGBK_val.x, HSIRGBK_val.x, HSIRGBK_val.x), clamp(p - make_float3(HSIRGBK_val.x, HSIRGBK_val.x, HSIRGBK_val.x), 0.0f, 1.0f), hsi.y);
    return c;
}




#define XK make_float3(.4124,.3576,.1805)
#define YK make_float3(.2126,.7152,.0722)
#define ZK make_float3(.0193,.1192,.9505)
#define REFERENCE_WHITE make_float3(95.047,100.0,108.883)

static __device__ __inline__ float adjustrgb(float v) {
    v = v <= .04045f ? v/12.92f : native_powr((v + 0.055f) / 1.055f, 2.4f);
    return v*100.0;
}

static __device__ __inline__ float3 _rgb2xyz(float3 rgb) {
   float3 rgb3 = ZERO3;
   rgb3.x = adjustrgb(rgb.x);
   rgb3.y = adjustrgb(rgb.y);
   rgb3.z = adjustrgb(rgb.z);
   float X = dot3(rgb3,XK);
   float Y = dot3(rgb3,YK);
   float Z = dot3(rgb3,ZK);
   float3 XYZ = make_float3(X,Y,Z);
   XYZ /= REFERENCE_WHITE;
   return XYZ;
}

static __device__ __inline__ float adjustxyz(float v) {
    v = v <= .008856f ? (7.787f * v) + (16.f / 116.f) : native_powr(v, 0.3333333333333333f);
    return v;
}

static __device__ __inline__ float3 _xyz2lab(float3 xyz) {
   float3 xyz3 = ZERO3;
   xyz3.x = adjustxyz(xyz.x);
   xyz3.y = adjustxyz(xyz.y);
   xyz3.z = adjustxyz(xyz.z);
   float3 lab = ZERO3;
   lab.x = (116*xyz3.y) -16;
   lab.y = 500 * (xyz3.x - xyz3.y);
   lab.z = 200 * (xyz3.y-xyz3.z);
   return lab;
}


extern "C" __kernel
void rgb2hsi(float *ra, float *ga, float *ba, float *ha, float *sa, float *ia GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t i = get_global_id(0);
    float3 c = make_float3(ra[i], ba[i], ga[i]);
    float3 hsi = _rgb2hsi(c);
    ha[i] = hsi.x;
    sa[i] = hsi.y;
    ia[i] = hsi.z;
}


extern "C" __kernel
void hsi2rgb(float *ha, float *sa, float *ia, float *ra, float *ga, float *ba GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
	size_t i = get_global_id(0);
	float3 hsi = make_float3(ha[i],sa[i],ia[i]);
    float3 c = _hsi2rgb(hsi);
    ra[i] = c.x;
    ga[i] = c.y;
    ba[i] = c.z;
}



extern "C" __kernel
void rgb2hsi3(int depth, float *input, float *output GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t i = get_global_id(0) * depth;
    float3 c = make_float3(input[i], input[i+2],input[i+1]);
    float3 hsi = _rgb2hsi(c);
    output[i] =   hsi.x;
    output[i+1] = hsi.y;
    output[i+2]=  hsi.z;
    if (depth > 3) {
        output[i+3] = input[i+3];  // Copy alpha channel through
    }
}


extern "C" __kernel
void hsi2rgb3(int depth, float *input, float *output GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
	size_t i = get_global_id(0)*depth;
	float3 hsi = make_float3(input[i],input[i+1],input[i+2]);
    float3 c = _hsi2rgb(hsi);
    output[i] =   c.x;
    output[i+1] = c.y;
    output[i+2] = c.z;
    if (depth > 3) {
        output[i+3] = input[i+3];  // Copy alpha channel through
    }
}

extern "C" __kernel
void rgb2lab3(int depth, float *input, float *output GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t i = get_global_id(0) * depth;
    float3 c = make_float3(input[i], input[i+2],input[i+1]);
    float3 xyz = _rgb2xyz(c);
    float3 lab = _xyz2lab(xyz);
    output[i] =   lab.x;
    output[i+1] = lab.y;
    output[i+2]=  lab.z;
}


//__kernel
//void lab2rgb3(int depth, float *input, float *output) {
// size_t i = get_global_id(0)*depth;
//	float3 lab = (float3)(input[i],input[i+1],input[i+2]);
//    float3 c = _lab2rgb(lab);
//    output[i] =   c.x;
//    output[i+1] = c.y;
//    output[i+2] = c.z;
//}
