/* Optimizer forward helpers; appended to existing FP32/sqrt32 helpers. */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned inputs = OPT_MODE == 0 ? 1 : 2;
    if (!m[0].isStatic || !m[0].rank || m[0].rank > 4 || !m[0].dims) return;
    unsigned n = 1, *dims = (unsigned*)m[0].dims;
    for (unsigned d = 0; d < m[0].rank; ++d) {
        if (!dims[d] || n > 65536u/dims[d]) return;
        n *= dims[d];
    }
    for (unsigned j = 0; j <= inputs; ++j)
        if (!m[j].isStatic || !m[j].lo || m[j].hi || m[j].dtype != 1) return;
    const unsigned *a = (const unsigned*)m[0].lo;
    unsigned *out = (unsigned*)m[inputs].lo;
    for (unsigned i = 0; i < n; ++i) {
#if OPT_MODE == 0
        out[i] = bits_of(sqrt32(from_bits(a[i])));
#else
        unsigned x = a[i], y = ((const unsigned*)m[1].lo)[i];
        unsigned ax = x & 0x7fffffffu, ay = y & 0x7fffffffu;
        if (ax > 0x7f800000u) out[i] = x | 0x400000u;
        else if (ay > 0x7f800000u) out[i] = y | 0x400000u;
        else if (!ax && !ay) out[i] = x;
        else {
            /* Integer ordering preserves FP32 subnormal comparisons. */
            unsigned kx = x >> 31 ? ~x : x ^ 0x80000000u;
            unsigned ky = y >> 31 ? ~y : y ^ 0x80000000u;
            out[i] = kx >= ky ? x : y;
        }
#endif
    }
}
