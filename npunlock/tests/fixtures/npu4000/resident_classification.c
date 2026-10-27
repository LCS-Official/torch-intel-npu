/* Appended to arithmetic.c helpers. Complete contiguous FP32 rows only.
 * exp/log below are original range-reduced Taylor/atanh implementations.
 * No host libm, double arithmetic, lookup tables or copied third-party code.
 */
static float exp_negative(float x) {
    unsigned bits = bits_of(x), magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return from_bits(0x7fc00000u);
    if (x < -104.0f) return 0.0f;
    if (x == 0.0f) return 1.0f;
    /* All finite callers have x<=0. Split ln(2) to retain small remainders. */
    int k = (int)(x * 1.4426950408889634f - 0.5f);
    float r = (x - k * 0.693145751953125f) - k * 0.000001428606765330187f;
    float p = 1.0f/40320.0f;
    p = 1.0f/5040.0f + r*p;
    p = 1.0f/720.0f + r*p;
    p = 1.0f/120.0f + r*p;
    p = 1.0f/24.0f + r*p;
    p = 1.0f/6.0f + r*p;
    p = 0.5f + r*p;
    p = 1.0f + r*p;
    p = 1.0f + r*p;
    /* encode64 keeps FP32 subnormal exponentials, avoiding SHAVE FTZ. */
    unsigned u = bits_of(p);
    return encode64((u & 0x7fffffu) | 0x800000u, k + (int)(u >> 23) - 150, 0);
}

static float log_positive(float x) {
    unsigned u = bits_of(x);
    int exponent = (int)(u >> 23) - 127;
    float m = from_bits((u & 0x7fffffu) | 0x3f800000u);
    if (m > 1.4142135623730951f) { m *= 0.5f; ++exponent; }
    float z = (m-1.0f)/(m+1.0f), square = z*z;
    float p = 1.0f/15.0f;
    p = 1.0f/13.0f + square*p;
    p = 1.0f/11.0f + square*p;
    p = 1.0f/9.0f + square*p;
    p = 1.0f/7.0f + square*p;
    p = 1.0f/5.0f + square*p;
    p = 1.0f/3.0f + square*p;
    p = 1.0f + square*p;
    return (exponent*0.693145751953125f + 2.0f*z*p) + exponent*0.000001428606765330187f;
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    for (unsigned i = 0; i < 3; ++i)
        if (!m[i].isStatic || m[i].rank != 4 || !m[i].dims || !m[i].lo || m[i].hi || m[i].dtype != 1) return;
    const unsigned *dims = (const unsigned*)m[0].dims;
    unsigned width = dims[0], rows = dims[1];
    if (dims[2] > 1) { width = dims[2]; rows = 1; }
    if (!width || !rows || rows > 65536u/width) return;
    const float *a = (const float*)m[0].lo, *b = (const float*)m[1].lo;
    float *out = (float*)m[2].lo;
    for (unsigned row = 0; row < rows; ++row) {
        unsigned base = row*width;
#if CLASS_MODE == 0
        float maximum = from_bits(0xff800000u);
        unsigned has_nan = 0;
        for (unsigned j = 0; j < width; ++j) {
            float x = a[base+j];
            if ((bits_of(x)&0x7fffffffu) > 0x7f800000u) has_nan = 1;
            if (x > maximum) maximum = x;
        }
        if (has_nan || (bits_of(maximum)&0x7fffffffu) == 0x7f800000u) {
            for (unsigned j = 0; j < width; ++j) out[base+j] = from_bits(0x7fc00000u);
            continue;
        }
        float total = 0.0f, correction = 0.0f;
        for (unsigned j = 0; j < width; ++j) {
            float v = exp_negative(add32(a[base+j], -maximum));
            float next = add32(total, v);
            correction = add32(correction, total >= v ? add32(add32(total, -next), v) : add32(add32(v, -next), total));
            total = next;
        }
        float logarithm = log_positive(add32(total, correction));
        for (unsigned j = 0; j < width; ++j)
            out[base+j] = add32(add32(a[base+j], -maximum), -logarithm);
#elif CLASS_MODE == 1
        float total = 0.0f, correction = 0.0f;
        for (unsigned j = 0; j < width; ++j) {
            float v = b[base+j], next = add32(total, v);
            if ((bits_of(total)&0x7fffffffu) < 0x7f800000u && (bits_of(v)&0x7fffffffu) < 0x7f800000u &&
                (bits_of(next)&0x7fffffffu) < 0x7f800000u)
                correction = add32(correction, __builtin_fabsf(total) >= __builtin_fabsf(v) ?
                    add32(add32(total, -next), v) : add32(add32(v, -next), total));
            else correction = 0.0f;
            total = next;
        }
        total = add32(total, correction);
        for (unsigned j = 0; j < width; ++j)
            out[base+j] = add32(b[base+j], -mul32(exp_negative(a[base+j]), total));
#elif CLASS_MODE == 2
        if (width < 5) return;
        unsigned classes = width-4;
        const unsigned *words = (const unsigned*)a;
        unsigned low = words[base+classes], high = words[base+classes+1];
        unsigned ignored = low == words[base+classes+2] && high == words[base+classes+3];
        unsigned valid = !ignored && !high && low < classes;
        for (unsigned j = 0; j < classes; ++j) out[base+j] = valid && j == low ? 1.0f : 0.0f;
        out[base+classes] = valid ? -a[base+low] : 0.0f;
        out[base+classes+1] = !ignored && !valid ? 1.0f : 0.0f;
        out[base+classes+2] = valid ? 1.0f : 0.0f;
        out[base+classes+3] = 0.0f;
#else
        if (width < 5) return;
        unsigned classes = width-4;
        for (unsigned j = 0; j < width; ++j)
            out[base+j] = j < classes && a[base+j] == 1.0f ? -b[base+j] : 0.0f;
#endif
    }
}
