/* Appended to arithmetic.c's helper prefix; raw bits preserve subnormals/NaNs. */
#ifdef RSQRT
/* Correctly rounded positive binary32 sqrt via an exact integer square root.
 * All positive subnormal inputs normalize before the calculation. */
static float sqrt32(float x) {
    unsigned u = bits_of(x), magnitude = u & 0x7fffffffu;
    if (!magnitude || magnitude >= 0x7f800000u)
        return from_bits(magnitude > 0x7f800000u ? u | 0x400000u :
                         magnitude && u >> 31 ? 0x7fc00000u : u);
    if (u >> 31) return from_bits(0x7fc00000u);
    int e = (int)(magnitude >> 23) - 127;
    unsigned mantissa = magnitude & 0x7fffffu;
    if (magnitude >> 23) mantissa |= 0x800000u;
    else {
        e = -126;
        while (!(mantissa & 0x800000u)) { mantissa <<= 1; --e; }
    }
    unsigned odd = (unsigned)e & 1u;
    U64 remainder = (U64)(mantissa << odd) << 23, root = 0, bit = (U64)1 << 46;
    while (bit) {
        if (remainder >= root + bit) {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        } else root >>= 1;
        bit >>= 2;
    }
    root += remainder > root;
    int result_e = (e - (int)odd) / 2;
    if (root == 0x1000000u) { root >>= 1; ++result_e; }
    return from_bits(((unsigned)(result_e + 127) << 23) | ((unsigned)root & 0x7fffffu));
}
#endif

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    if (!m[0].isStatic || !m[0].rank || m[0].rank > 4 || !m[0].dims) return;
    unsigned n = 1, *dims = (unsigned*)m[0].dims;
    for (unsigned i = 0; i < m[0].rank; ++i) {
        if (!dims[i] || n > 65536u / dims[i]) return;
        n *= dims[i];
    }
    for (unsigned i = 0; i <= INPUTS; ++i)
        if (!m[i].lo || m[i].hi || m[i].dtype != DTYPE) return;
#if DTYPE == 1
    typedef unsigned Word;
    const unsigned sign = 0x80000000u, infinity = 0x7f800000u;
#else
    typedef unsigned short Word;
    const unsigned sign = 0x8000u, infinity = 0x7c00u;
#endif
    const Word *x = (const Word*)m[0].lo;
    Word *out = (Word*)m[INPUTS].lo;
    for (unsigned i = 0; i < n; ++i) {
        unsigned bits = x[i], magnitude = bits & (sign - 1);
#ifdef RSQRT
        out[i] = bits_of(div32(1.0f, sqrt32(from_bits(bits))));
#elif INPUTS == 2
        /* threshold_backward: NaN input passes the incoming gradient. */
        out[i] = (magnitude > infinity || (magnitude && !(bits & sign))) ?
                 ((const Word*)m[1].lo)[i] : 0;
#else
        out[i] = (!(bits & sign) || !magnitude || magnitude > infinity) ? bits : 0;
#endif
    }
}
