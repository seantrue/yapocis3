// HSI and related conversions (Metal)

<%include file="constants.mako" />
#define DOT3
<%include file="helpers.mako" />

#define eps 1.0e-10

#define RGBHSIK float4(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f)
static inline float3 _rgb2hsi(float3 c) {
    float4 p = c.y < c.z ? float4(c.zy, RGBHSIK.wz) : float4(c.yz, RGBHSIK.xy);
    float4 q = c.x < p.x ? float4(p.xyw, c.x) : float4(c.x, p.yzx);
    float d = q.x - min(q.w, q.y);
    float hh = q.z + (q.w - q.y) / (6.0f * d + eps);
    hh = hh < 0.0 ? -hh : hh;
    float3 hsi = float3(hh , d / (q.x + eps), q.x);
    hsi = clamp(hsi, 0.0f, 1.0f-eps);
    return hsi;
}

#define HSIRGBK float4(1.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f)
static inline float3 _hsi2rgb(float3 hsi) {
    float3 tmp = hsi.xxx + HSIRGBK.xyz;
    float3 floorv = floor(tmp);
    float3 p = fabs((tmp - floorv) * 6.0f - HSIRGBK.www);
    float3 c =  hsi.z * mix(HSIRGBK.xxx, clamp(p - HSIRGBK.xxx, 0.0f, 1.0f), hsi.y);
    return c;
}

#define XK float3(.4124,.3576,.1805)
#define YK float3(.2126,.7152,.0722)
#define ZK float3(.0193,.1192,.9505)
#define REFERENCE_WHITE float3(95.047,100.0,108.883)

static inline float adjustrgb(float v) {
    v = v <= .04045f ? v/12.92f : pow((v + 0.055f) / 1.055f, 2.4f);
    return v*100.0f;
}

static inline float3 _rgb2xyz(float3 rgb) {
   float3 rgb3 = float3(0.0f);
   rgb3.x = adjustrgb(rgb.x);
   rgb3.y = adjustrgb(rgb.y);
   rgb3.z = adjustrgb(rgb.z);
   float X = dot3(rgb3,XK);
   float Y = dot3(rgb3,YK);
   float Z = dot3(rgb3,ZK);
   float3 XYZ = float3(X,Y,Z);
   XYZ /= REFERENCE_WHITE;
   return XYZ;
}

static inline float adjustxyz(float v) {
    v = v <= .008856f ? (7.787f * v) + (16.f / 116.f) : pow(v, 0.3333333333333333f);
    return v;
}

static inline float3 _xyz2lab(float3 xyz) {
   float3 xyz3 = float3(0.0f);
   xyz3.x = adjustxyz(xyz.x);
   xyz3.y = adjustxyz(xyz.y);
   xyz3.z = adjustxyz(xyz.z);
   float3 lab = float3(0.0f);
   lab.x = (116*xyz3.y) -16;
   lab.y = 500 * (xyz3.x - xyz3.y);
   lab.z = 200 * (xyz3.y-xyz3.z);
   return lab;
}

kernel void rgb2hsi(device float *ra [[ buffer(0) ]],
                    device float *ga [[ buffer(1) ]],
                    device float *ba [[ buffer(2) ]],
                    device float *ha [[ buffer(3) ]],
                    device float *sa [[ buffer(4) ]],
                    device float *ia [[ buffer(5) ]],
                    uint i [[ thread_position_in_grid ]]) {
    float3 c = float3(ra[i], ba[i], ga[i]);
    float3 hsi = _rgb2hsi(c);
    ha[i] = hsi.x;
    sa[i] = hsi.y;
    ia[i] = hsi.z;
}

kernel void hsi2rgb(device float *ha [[ buffer(0) ]],
                    device float *sa [[ buffer(1) ]],
                    device float *ia [[ buffer(2) ]],
                    device float *ra [[ buffer(3) ]],
                    device float *ga [[ buffer(4) ]],
                    device float *ba [[ buffer(5) ]],
                    uint i [[ thread_position_in_grid ]]) {
    float3 hsi = float3(ha[i],sa[i],ia[i]);
    float3 c = _hsi2rgb(hsi);
    ra[i] = c.x;
    ga[i] = c.y;
    ba[i] = c.z;
}

kernel void rgb2hsi3(constant int &depth [[ buffer(0) ]],
                     device float *input [[ buffer(1) ]],
                     device float *output [[ buffer(2) ]],
                     uint i [[ thread_position_in_grid ]]) {
    size_t idx = i * (size_t)depth;
    float3 c = float3(input[idx], input[idx+2], input[idx+1]);
    float3 hsi = _rgb2hsi(c);
    output[idx] =   hsi.x;
    output[idx+1] = hsi.y;
    output[idx+2]=  hsi.z;
}

kernel void hsi2rgb3(constant int &depth [[ buffer(0) ]],
                     device float *input [[ buffer(1) ]],
                     device float *output [[ buffer(2) ]],
                     uint i [[ thread_position_in_grid ]]) {
    size_t idx = i * (size_t)depth;
    float3 hsi = float3(input[idx],input[idx+1],input[idx+2]);
    float3 c = _hsi2rgb(hsi);
    output[idx] =   c.x;
    output[idx+1] = c.y;
    output[idx+2] = c.z;
}

kernel void rgb2lab3(constant int &depth [[ buffer(0) ]],
                     device float *input [[ buffer(1) ]],
                     device float *output [[ buffer(2) ]],
                     uint i [[ thread_position_in_grid ]]) {
    size_t idx = i * (size_t)depth;
    float3 c = float3(input[idx], input[idx+2], input[idx+1]);
    float3 xyz = _rgb2xyz(c);
    float3 lab = _xyz2lab(xyz);
    output[idx] =   lab.x;
    output[idx+1] = lab.y;
    output[idx+2]=  lab.z;
}
