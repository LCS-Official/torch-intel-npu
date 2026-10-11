/* Same dot order and IEEE FP32 helpers as square_gemm.c. Two compiler-created
 * row tasks share a full B upload: the host verifies DMA widths, ordering,
 * completion barriers, exact row coverage and disjoint output spans.
 * A's first CMX address is part of this bounded, fail-closed carrier contract.
 */
#if GEMM_FAST
/* Keep the normal SAU operations in the hot loop. The predicates are exactly
 * those of arithmetic.c; rare software/IEEE cases reuse its original helpers.
 * Otherwise Movi outlines add32/mul32 and spills around every scalar operation.
 */
static __attribute__((always_inline)) inline float gemm_add(float a, float b) {
    unsigned ea = (bits_of(a) >> 23) & 255u, eb = (bits_of(b) >> 23) & 255u;
    if (ea > 25u || eb > 25u) return a + b;
    return add32(a, b);
}
static __attribute__((always_inline)) inline float gemm_mul(float a, float b) {
    unsigned ea = (bits_of(a) >> 23) & 255u, eb = (bits_of(b) >> 23) & 255u;
    if (ea && eb && ea < 255u && eb < 255u && ea + eb > 128u) return a * b;
    return mul32(a, b);
}
#define add32 gemm_add
#define mul32 gemm_mul
#endif
static inline void accumulate(float value, float *total, float *correction) {
    float next = add32(*total, value);
    unsigned left = bits_of(*total) & 0x7fffffffu;
    unsigned right = bits_of(value) & 0x7fffffffu;
    if ((bits_of(next) & 0x7f800000u) != 0x7f800000u) {
        float error = left >= right ? add32(add32(*total, -next), value)
                                   : add32(add32(value, -next), *total);
        *correction = add32(*correction, error);
    } else *correction = 0.0f;
    *total = next;
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    for (unsigned i = 0; i < 3; ++i)
        if (!m[i].lo || m[i].hi || !m[i].isStatic || m[i].rank != 4 ||
            m[i].dtype != 1 || m[i].location != 2 || !m[i].dims || !m[i].strides ||
            m[i].orderLo != 0x4321u || m[i].orderHi) return;
#if GEMM_FOUR_TILE
    unsigned *shape = (unsigned*)m[0].dims;
    unsigned count = shape[0]*shape[1], size = 8, tile = 0;
    while (size < 64 && size*size < count) size *= 2;
    if (count != size*size || shape[0] != count/2 || shape[1] != 2) return;
    unsigned select = m[0].lo & 0x01e00000u;
    while (tile < 4 && select != (1u << (21+tile))) ++tile;
    if (tile == 4 || m[0].lo != 0x40018000u + select) return;
    unsigned rows = size, offset = 0;
#else
    unsigned size = ((unsigned*)m[0].dims)[0], rows = ((unsigned*)m[0].dims)[1];
    if (size < 2 || size > 64 || !rows || rows > size || m[0].lo < 0x40218000u) return;
    unsigned offset = m[0].lo - 0x40218000u;
    if (offset % (4u*size) || offset / (4u*size) + rows > size || m[1].lo < offset) return;
#endif
    for (unsigned i = 0; i < 3; ++i) {
        unsigned *d = (unsigned*)m[i].dims;
        U64 *s = (U64*)m[i].strides;
#if GEMM_FOUR_TILE
        if (d[0] != count/2 || d[1] != 2 || d[2] != 1 || d[3] != 1 ||
            s[0] != 32 || s[1] != 16u*count || s[2] != 32u*count || s[3] != s[2]) return;
#else
        if (d[0] != size || d[1] != rows || d[2] != 1 || d[3] != 1 ||
            s[0] != 32 || s[1] != 32u*size ||
            s[2] != 32u*size*rows || s[3] != s[2]) return;
#endif
    }
    const float *a = (const float*)m[0].lo;
    const float *b = (const float*)(m[1].lo - offset);
    float *c = (float*)m[2].lo;
    for (unsigned row = 0; row < rows; ++row) {
#if GEMM_FOUR_TILE
        /* Four partial outputs contain disjoint rows and zero elsewhere.
         * Existing device reduction assembles C without an M*K*N expansion. */
        if (row % 4 != tile) {
            for (unsigned col = 0; col < size; ++col) c[row*size+col] = 0.0f;
            continue;
        }
#endif
        unsigned col = 0;
#if GEMM_BLOCK4
        for (; col + 4 <= size; col += 4) {
            float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
            float e0 = 0, e1 = 0, e2 = 0, e3 = 0;
            for (unsigned k = 0; k < size; ++k) {
                float left = a[row*size+k];
                const float *right = b+k*size+col;
                accumulate(mul32(left, right[0]), &t0, &e0);
                accumulate(mul32(left, right[1]), &t1, &e1);
                accumulate(mul32(left, right[2]), &t2, &e2);
                accumulate(mul32(left, right[3]), &t3, &e3);
            }
            c[row*size+col] = add32(t0, e0);
            c[row*size+col+1] = add32(t1, e1);
            c[row*size+col+2] = add32(t2, e2);
            c[row*size+col+3] = add32(t3, e3);
        }
#endif
        for (; col < size; ++col) {
            float total = 0, correction = 0;
            for (unsigned k = 0; k < size; ++k)
                accumulate(mul32(a[row*size+k], b[k*size+col]), &total, &correction);
            c[row*size+col] = add32(total, correction);
        }
    }
}
