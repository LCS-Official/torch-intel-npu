/* Original SHAVE implementation of GELU and its analytic first derivative.
 * erf uses Abramowitz & Stegun 7.1.26 (p.299), not the tanh approximation.
 * Coefficients are published mathematical constants; no libm/source copied.
 * Appended to arithmetic.c and resident_classification.c helper prefixes.
 */
static float erf32(float x) {
    unsigned bits = bits_of(x), mag = bits & 0x7fffffffu;
    if (!mag) return x;
    float z = from_bits(mag);
    if (z > 10.0f) return bits >> 31 ? -1.0f : 1.0f;
    float t = div32(1.0f, add32(1.0f, mul32(0.3275911f, z)));
    float p = add32(mul32(1.061405429f, t), -1.453152027f);
    p = add32(mul32(p, t), 1.421413741f);
    p = add32(mul32(p, t), -0.284496736f);
    p = add32(mul32(p, t), 0.254829592f);
    float result = add32(1.0f, -mul32(mul32(p, t), exp_negative(-mul32(z, z))));
    return bits >> 31 ? -result : result;
}

static float tanh32(float x) {
    unsigned bits = bits_of(x);
    float z = from_bits(bits & 0x7fffffffu);
    if (z > 10.0f) return bits >> 31 ? -1.0f : 1.0f;
    /* Use an odd small-x series to avoid cancellation in 1-exp(-2*abs(x)). */
    if (z < 0.03125f) {
        float square = mul32(x, x);
        float p = add32(-1.0f/3.0f, mul32(square, 2.0f/15.0f));
        return add32(x, mul32(mul32(x, square), p));
    }
    float e = exp_negative(-mul32(2.0f, z));
    float result = div32(add32(1.0f, -e), add32(1.0f, e));
    return bits >> 31 ? -result : result;
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned inputs = 2;
    if (!m[0].isStatic || !m[0].rank || m[0].rank > 4 || !m[0].dims) return;
    unsigned n = 1, *dims = (unsigned*)m[0].dims;
    for (unsigned d = 0; d < m[0].rank; ++d) {
        if (!dims[d] || n > 65536u/dims[d]) return;
        n *= dims[d];
    }
    for (unsigned j = 0; j <= inputs; ++j)
        if (!m[j].isStatic || !m[j].lo || m[j].hi || m[j].dtype != 1) return;
    const float *a = (const float*)m[0].lo;
    float *out = (float*)m[inputs].lo;
    for (unsigned i = 0; i < n; ++i) {
        float x = a[i];
        unsigned bits = bits_of(x), mag = bits & 0x7fffffffu;
        float result;
        if (mag > 0x7f800000u) result = from_bits(bits | 0x400000u);
        else if (mag == 0x7f800000u)
            result = GELU_BACKWARD || bits >> 31 ? from_bits(0x7fc00000u) : x;
        else {
#if GELU_TANH
            float square = mul32(x, x);
            float inner = mul32(0.7978845608028654f, add32(x, mul32(0.044715f, mul32(square, x))));
            float t = tanh32(inner);
            float cdf = mul32(0.5f, add32(1.0f, t));
#if GELU_BACKWARD
            float dt = mul32(0.7978845608028654f, add32(1.0f, mul32(0.134145f, square)));
            float tail = mul32(mul32(mul32(0.5f, x), add32(1.0f, -mul32(t, t))), dt);
            result = add32(cdf, tail);
#else
            result = mul32(mul32(0.5f, x), add32(1.0f, t));
#endif
#else
            float cdf = mul32(0.5f, add32(1.0f, erf32(mul32(x, 0.7071067811865475f))));
#if GELU_BACKWARD
            float pdf = mul32(0.3989422804014327f, exp_negative(-mul32(mul32(0.5f, x), x)));
            result = add32(cdf, mul32(x, pdf));
#else
            result = mul32(x, cdf);
#endif
#endif
        }
#if GELU_BACKWARD
        result = mul32(((const float*)m[1].lo)[i], result);
#endif
        out[i] = result;
    }
}
