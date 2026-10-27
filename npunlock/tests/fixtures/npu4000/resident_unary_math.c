/* Public scalar math; reuses the project's IEEE, exp/log and tanh helpers. */
static float exp_all(float x) {
    unsigned u = bits_of(x), magnitude = u & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return from_bits(u | 0x400000u);
    if (x > 89.0f) return from_bits(0x7f800000u);
    if (x < -104.0f) return 0.0f;
    int k = (int)(x * 1.4426950408889634f + (x < 0.0f ? -0.5f : 0.5f));
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
    unsigned bits = bits_of(p);
    return encode64((bits & 0x7fffffu) | 0x800000u, k + (int)(bits >> 23) - 150, 0);
}

static float log_all(float x) {
    unsigned u = bits_of(x), magnitude = u & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return from_bits(u | 0x400000u);
    if (!magnitude) return from_bits(0xff800000u);
    if (u >> 31) return from_bits(0x7fc00000u);
    if (magnitude == 0x7f800000u) return x;
    if (magnitude >= 0x800000u) return log_positive(x);
    int exponent = -126;
    while (!(magnitude & 0x800000u)) { magnitude <<= 1; --exponent; }
    float logarithm = log_positive(from_bits(0x3f800000u | (magnitude & 0x7fffffu)));
    return add32(add32(exponent*0.693145751953125f, logarithm), exponent*0.000001428606765330187f);
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned n = 1;
    if (!m[0].isStatic || !m[0].rank || m[0].rank > 4 || !m[0].dims) return;
    unsigned *dims = (unsigned*)m[0].dims;
    for (unsigned d = 0; d < m[0].rank; ++d) {
        if (!dims[d] || n > 65536u/dims[d]) return;
        n *= dims[d];
    }
    for (unsigned j = 0; j < 3; ++j)
        if (!m[j].isStatic || !m[j].lo || m[j].hi || m[j].dtype != 1) return;
    const float *a = (const float*)m[0].lo;
    /* Power carrier provides code space; its second read-only input is unused. */
    float *out = (float*)m[2].lo;
    for (unsigned i = 0; i < n; ++i) {
        float x = a[i];
#if MATH_OP == 0
        out[i] = exp_all(x);
#elif MATH_OP == 1
        out[i] = log_all(x);
#elif MATH_OP == 2
        out[i] = (bits_of(x) & 0x7fffffffu) ? tanh32(x) : x;
#else
        float sigmoid = div32(1.0f, add32(1.0f, exp_all(-x)));
#if MATH_OP == 3
        out[i] = sigmoid;
#else
        out[i] = mul32(x, sigmoid);
#endif
#endif
    }
}
