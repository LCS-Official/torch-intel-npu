/* Philox4x32-10, expressed with 32-bit limbs; one counter block per element.
 * Algorithm reference: Salmon et al., Parallel Random Numbers: As Easy as 1, 2, 3.
 * This implementation and the Box-Muller helpers are written for this carrier.
 */
typedef struct { unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location; } MemRef;
#define INLINE static inline __attribute__((always_inline))
INLINE unsigned bits(float x) { union { float f; unsigned u; } v; v.f=x; return v.u; }
INLINE float value(unsigned x) { union { float f; unsigned u; } v; v.u=x; return v.f; }
INLINE unsigned mulhi(unsigned a, unsigned b) {
    unsigned al=a&65535u, ah=a>>16, bl=b&65535u, bh=b>>16;
    unsigned p=al*bl, q=ah*bl+(p>>16), r=al*bh+(q&65535u);
    return ah*bh+(q>>16)+(r>>16);
}
INLINE float logarithm(float x) {
    unsigned u=bits(x);
    int e=(int)(u>>23)-127;
    float m=value((u&0x7fffffu)|0x3f800000u);
    if (m>1.4142135623730951f) { m*=0.5f; ++e; }
    float z=(m-1.0f)/(m+1.0f), t=z*z, p=1.0f/15.0f;
    p=1.0f/13.0f+t*p; p=1.0f/11.0f+t*p; p=1.0f/9.0f+t*p;
    p=1.0f/7.0f+t*p; p=1.0f/5.0f+t*p; p=1.0f/3.0f+t*p; p=1.0f+t*p;
    return (e*0.693145751953125f+2.0f*z*p)+e*0.000001428606765330187f;
}
INLINE float root(float x) {
    float r=value((bits(x)>>1)+0x1fc00000u);
    for (unsigned i=0; i<5; ++i) r=0.5f*(r+x/r);
    return r;
}
INLINE float cosine(float u) {
    float x=u*6.283185307179586f;
    if (x>3.141592653589793f) x-=6.283185307179586f;
    float sign=1.0f;
    if (x>1.5707963267948966f) { x=3.141592653589793f-x; sign=-1.0f; }
    if (x< -1.5707963267948966f) { x= -3.141592653589793f-x; sign=-1.0f; }
    float t=x*x, p=1.0f/479001600.0f;
    p= -1.0f/3628800.0f+t*p; p=1.0f/40320.0f+t*p;
    p= -1.0f/720.0f+t*p; p=1.0f/24.0f+t*p; p= -0.5f+t*p;
    return sign*(1.0f+t*p);
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m=(MemRef*)p;
    unsigned words=0;
    for (unsigned t=0; t<3; ++t) {
        if (!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1) return;
        unsigned n=1; const unsigned *d=(const unsigned*)m[t].dims;
        for (unsigned a=0; a<4; ++a) { if (!d[a] || d[a]>65536u/n) return; n*=d[a]; }
        if (t && words!=n) return;
        words=n;
    }
    if (words<8) return;
    const unsigned *h=(const unsigned*)m[0].lo;
    unsigned count=h[4], mode=h[5], half=h[6];
    if (!count || count>words || mode>1 || half>1) return;
    float *out=(float*)m[2].lo;
    for (unsigned i=0; i<words; ++i) out[i]=0.0f;
    for (unsigned i=0; i<count; ++i) {
        unsigned c0=h[2]+i, c1=h[3]+(c0<h[2]), c2=0, c3=0, k0=h[0], k1=h[1];
        for (unsigned round=0; round<10; ++round) {
            unsigned hi0=mulhi(0xd2511f53u,c0), lo0=0xd2511f53u*c0;
            unsigned hi1=mulhi(0xcd9e8d57u,c2), lo1=0xcd9e8d57u*c2;
            unsigned next0=hi1^c1^k0, next2=hi0^c3^k1;
            c0=next0; c1=lo1; c2=next2; c3=lo0;
            k0+=0x9e3779b9u; k1+=0xbb67ae85u;
        }
        if (!mode) out[i]=half ? (float)(c0>>21)*0.00048828125f : (float)(c0>>8)*0.000000059604644775390625f;
        else {
            float u=((float)(c0>>9)+0.5f)*0.00000011920928955078125f;
            float v=(float)(c1>>8)*0.000000059604644775390625f;
            out[i]=root(-2.0f*logarithm(u))*cosine(v);
        }
    }
}
