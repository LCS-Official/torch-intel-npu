/* Static dense Lunar Lake / 4000xx ABI. IEEE binary16 storage, FP32 arithmetic. */
typedef struct { unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location; } MemRef;
#ifndef TYPE
#define TYPE float
#endif
#ifndef DTYPE
#define DTYPE 1
#endif
#ifndef OP
#define OP 0
#endif
#if OP < 3 || OP >= 7
#define INPUTS 1
#else
#define INPUTS 2
#endif

/* SHAVE's hardware f16 conversion flushes subnormals; preserve them with bits. */
#if DTYPE == 2
static inline float load_value(const TYPE *p, unsigned i) {
    unsigned h = p[i], s = (h & 0x8000u) << 16;
    unsigned e = (h >> 10) & 31u, m = h & 1023u;
    union { unsigned u; float f; } v;
    if (!e) {
        if (!m) v.u = s;
        else {
            unsigned fe = 113u;
            while (!(m & 1024u)) { m <<= 1; --fe; }
            v.u = s | (fe << 23) | ((m & 1023u) << 13);
        }
    } else if (e == 31u) v.u = s | 0x7f800000u | (m << 13);
    else v.u = s | ((e + 112u) << 23) | (m << 13);
    return v.f;
}
#else
static inline float load_value(const TYPE *p, unsigned i) { return p[i]; }
#endif

static inline unsigned short half_bits(float value) {
    union { unsigned u; float f; } v; v.f = value;
    unsigned sign = (v.u >> 16) & 0x8000u;
    unsigned bits = v.u & 0x7fffffffu;
    unsigned fraction = bits & 0x7fffffu;
    int exponent = (int)(bits >> 23) - 127;
    if (exponent == 128) return sign | 0x7c00u | (fraction ? ((fraction >> 13) | 0x200u) : 0u);
    if (exponent >= 16) return sign | 0x7c00u;
    if (exponent >= -14) return sign | ((bits + 0xfffu + ((bits >> 13) & 1u) - 0x38000000u) >> 13);
    if (exponent < -25) return sign;
    unsigned shift = (unsigned)(-exponent - 1);
    unsigned mantissa = fraction | 0x800000u;
    unsigned rounded = mantissa >> shift;
    unsigned remainder = mantissa & ((1u << shift) - 1u);
    unsigned halfway = 1u << (shift - 1u);
    return sign | (rounded + (remainder > halfway || (remainder == halfway && (rounded & 1u))));
}

typedef unsigned long long U64;
static inline unsigned bits_of(float x) { union { float f; unsigned u; } v; v.f = x; return v.u; }
static inline float from_bits(unsigned x) { union { float f; unsigned u; } v; v.u = x; return v.f; }

static inline U64 round_shift(U64 v, int shift) {
    if (shift <= 0) return v << -shift;
    if (shift >= 64) return 0;
    U64 q = v >> shift, remainder = v & (((U64)1 << shift) - 1);
    U64 half = (U64)1 << (shift - 1);
    return q + (remainder > half || (remainder == half && (q & 1)));
}

/* Encode v*2^exponent exactly with round-to-nearest-even, including subnormals. */
static inline float encode64(U64 v, int exponent, unsigned sign) {
    if (!v) return from_bits(sign);
    unsigned p = 0;
    U64 t = v;
    while (t >>= 1) ++p;
    int e = (int)p + exponent;
    if (e > 127) return from_bits(sign | 0x7f800000u);
    if (e < -126) return from_bits(sign | (unsigned)round_shift(v, -exponent - 149));
    U64 mantissa = round_shift(v, (int)p - 23);
    if (mantissa == 0x1000000u) { mantissa >>= 1; ++e; }
    if (e > 127) return from_bits(sign | 0x7f800000u);
    return from_bits(sign | ((unsigned)(e + 127) << 23) | ((unsigned)mantissa & 0x7fffffu));
}

/* Keep normal operations in SAU.f32; integer edge paths prevent FTZ. */
static inline float add32(float a, float b) {
    unsigned ua = bits_of(a), ub = bits_of(b);
    unsigned ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u;
    if (ea <= 25u && eb <= 25u) {
        U64 ma = ua & 0x7fffffu, mb = ub & 0x7fffffu;
        if (ea) ma = (ma | 0x800000u) << (ea - 1);
        if (eb) mb = (mb | 0x800000u) << (eb - 1);
        long long sa = ua >> 31 ? -(long long)ma : (long long)ma;
        long long sb = ub >> 31 ? -(long long)mb : (long long)mb;
        long long total = sa + sb;
        unsigned sign = total < 0 ? 0x80000000u : total == 0 ? ua & ub & 0x80000000u : 0;
        return encode64(total < 0 ? (U64)-total : (U64)total, -149, sign);
    }
    return a + b;
}

static inline float mul32(float a, float b) {
    unsigned ua = bits_of(a), ub = bits_of(b);
    unsigned ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u;
    if (ea == 255u || eb == 255u) {
        unsigned aa = ua & 0x7fffffffu, bb = ub & 0x7fffffffu;
        if (aa > 0x7f800000u || bb > 0x7f800000u) return a * b;
        if (!aa || !bb) return from_bits(0x7fc00000u);
        return from_bits(((ua ^ ub) & 0x80000000u) | 0x7f800000u);
    }
    if (ea != 255u && eb != 255u && (!ea || !eb || ea + eb <= 128u)) {
        U64 ma = ua & 0x7fffffu, mb = ub & 0x7fffffu;
        if (ea) ma |= 0x800000u;
        if (eb) mb |= 0x800000u;
        int exponent = (ea ? (int)ea - 127 : -126) + (eb ? (int)eb - 127 : -126) - 46;
        return encode64(ma * mb, exponent, (ua ^ ub) & 0x80000000u);
    }
    return a * b;
}

static inline float div32(float a, float b) {
    unsigned ua = bits_of(a), ub = bits_of(b);
    unsigned ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u;
    unsigned ma = ua & 0x7fffffu, mb = ub & 0x7fffffu;
    unsigned sign = (ua ^ ub) & 0x80000000u;
    if (ea == 255u || eb == 255u) {
        if (ma && ea == 255u || mb && eb == 255u) return a / b;
        if (ea == 255u && eb == 255u) return from_bits(0x7fc00000u);
        return from_bits(sign | (ea == 255u ? 0x7f800000u : 0u));
    }
    if (!eb && !mb) return from_bits(!ea && !ma ? 0x7fc00000u : sign | 0x7f800000u);
    if (!ea && !ma) return from_bits(sign);
    if (ea && eb && (int)ea - (int)eb > -125) return a / b;
    int ae = ea ? (int)ea - 127 : -126, be = eb ? (int)eb - 127 : -126;
    if (ea) ma |= 0x800000u;
    if (eb) mb |= 0x800000u;
    while (!(ma & 0x800000u)) { ma <<= 1; --ae; }
    while (!(mb & 0x800000u)) { mb <<= 1; --be; }
    int e = ae - be;
    if (ma < mb) { ma <<= 1; --e; }
    if (e > 127) return from_bits(sign | 0x7f800000u);
    if (e < -150) return from_bits(sign);
    if (e == -150) return from_bits(sign | (ma != mb));
    unsigned count = e < -126 ? (unsigned)(e + 149) : 23u;
    unsigned q = 1, remainder = ma - mb;
    for (unsigned i = 0; i < count; ++i) {
        remainder <<= 1; q <<= 1;
        if (remainder >= mb) { remainder -= mb; q |= 1u; }
    }
    q += remainder * 2 > mb || (remainder * 2 == mb && (q & 1u));
    if (e < -126) return from_bits(sign | q);
    if (q == 0x1000000u) { q >>= 1; ++e; }
    if (e > 127) return from_bits(sign | 0x7f800000u);
    return from_bits(sign | ((unsigned)(e + 127) << 23) | (q & 0x7fffffu));
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    if (!m[0].isStatic || !m[0].rank || m[0].rank > 15 || !m[0].dims) return;
    unsigned *dims = (unsigned*)m[0].dims;
    unsigned n = 1;
    for (unsigned i = 0; i < m[0].rank; ++i) {
        if (!dims[i] || n > 65536u / dims[i]) return;
        n *= dims[i];
    }
    for (unsigned i = 0; i <= INPUTS; ++i)
        if (!m[i].lo || m[i].hi || m[i].dtype != (OP == 7 && i == INPUTS ? 1 : DTYPE)) return;
    const TYPE *a = (const TYPE*)m[0].lo;
#if OP == 7
    float *out = (float*)m[INPUTS].lo;
#elif OP == 8
    unsigned *out = (unsigned*)m[INPUTS].lo;
#else
    TYPE *out = (TYPE*)m[INPUTS].lo;
#endif
#if INPUTS == 2
    const TYPE *b = (const TYPE*)m[1].lo;
#endif
    for (unsigned i = 0; i < n; ++i) {
#if OP == 2 && DTYPE == 1
        /* SAU.ABS.f32 flushes subnormals. Integer loads/stores preserve bits. */
        ((unsigned*)out)[i] = ((const unsigned*)a)[i] & 0x7fffffffu;
#else
#if OP == 0
        float v = add32(load_value(a, i), 1.0f);
#elif OP == 1
        float v = -load_value(a, i);
#elif OP == 2
        float v = __builtin_fabsf(load_value(a, i));
#elif OP == 3
        float v = add32(load_value(a, i), load_value(b, i));
#elif OP == 4
        float v = add32(load_value(a, i), from_bits(bits_of(load_value(b, i)) ^ 0x80000000u));
#elif OP == 5
        float v = mul32(load_value(a, i), load_value(b, i));
#elif OP == 6
        float v = div32(load_value(a, i), load_value(b, i));
#elif OP == 7 || OP == 8
        float v = load_value(a, i);
#elif OP == 9
        unsigned u = bits_of(load_value(a, i));
        float v = from_bits((u & 0x7fffffffu) == 0 || (u & 0x7fffffffu) > 0x7f800000u ?
                            0u : (u & 0x80000000u) | 0x3f800000u);
#endif
#if OP == 7
        out[i] = v;
#elif OP == 8
        out[i] = half_bits(v);
#elif DTYPE == 2
        out[i] = half_bits(v);
#else
        out[i] = (TYPE)v;
#endif
#endif
    }
}
