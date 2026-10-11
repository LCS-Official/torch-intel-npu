/* Full packed A/B per tile. Shape scalars prefix A; all accesses stay inside
 * the verified capacity. Reuse the parallel GEMM FP32/Neumaier helpers. */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *ref = (MemRef*)p;
    unsigned capacity = 0;
    for (unsigned i = 0; i < 3; ++i) {
        if (!ref[i].lo || ref[i].hi || !ref[i].isStatic || ref[i].rank != 4 ||
            ref[i].dtype != 1 || ref[i].location != 2 || !ref[i].dims || !ref[i].strides ||
            ref[i].orderLo != 0x4321u || ref[i].orderHi) return;
        unsigned *d = (unsigned*)ref[i].dims;
        U64 *s = (U64*)ref[i].strides;
        unsigned rows = GEMM_EIGHT ? 8u : 2u;
        unsigned count = rows*d[0];
        if (count < 64 || count > 16384 || count % 64 || d[1] != rows || d[2] != 1 || d[3] != 1 ||
            s[0] != 32 || s[1] != 32u*count/rows || s[2] != 32u*count || s[3] != s[2] ||
            (capacity && count != capacity)) return;
        capacity = count;
    }
    unsigned select = ref[0].lo & 0x01e00000u, tile = 0;
    while (tile < 4 && select != (1u << (21+tile))) ++tile;
    if (tile == 4) return;
    unsigned lane = 0;
#if GEMM_EIGHT
    unsigned base = 0x40018000u+select;
    if (ref[0].lo < base || (ref[0].lo-base) % (capacity*4u)) return;
    lane = (ref[0].lo-base)/(capacity*4u);
    if (lane > 1) return;
#else
    if (ref[0].lo != 0x40018000u+select) return;
#endif
    const float *header = (const float*)ref[0].lo;
    if (!(header[0] == 12345.0f && header[1] >= 1 && header[1] <= 64 &&
          header[2] >= 1 && header[2] <= 256 && header[3] >= 1 && header[3] <= 64)) return;
    unsigned m = (unsigned)header[1], ksize = (unsigned)header[2], n = (unsigned)header[3];
    if (header[1] != (float)m || header[2] != (float)ksize || header[3] != (float)n ||
        4u+m*ksize > capacity || ksize*n > capacity || m*n > capacity) return;
    const float *a = header+4, *b = (const float*)ref[1].lo;
    float *c = (float*)ref[2].lo;
    for (unsigned i = 0; i < capacity; ++i) c[i] = 0.0f;
    unsigned worker = GEMM_EIGHT ? tile*2u+lane : tile;
    for (unsigned row = worker; row < m; row += (GEMM_EIGHT ? 8u : 4u)) {
        unsigned col = 0;
        for (; col+4 <= n; col += 4) {
            float t0 = 0, t1 = 0, t2 = 0, t3 = 0, e0 = 0, e1 = 0, e2 = 0, e3 = 0;
            for (unsigned k = 0; k < ksize; ++k) {
                float left = a[row*ksize+k];
                const float *right = b+k*n+col;
                accumulate(mul32(left, right[0]), &t0, &e0);
                accumulate(mul32(left, right[1]), &t1, &e1);
                accumulate(mul32(left, right[2]), &t2, &e2);
                accumulate(mul32(left, right[3]), &t3, &e3);
            }
            c[row*n+col] = add32(t0, e0);
            c[row*n+col+1] = add32(t1, e1);
            c[row*n+col+2] = add32(t2, e2);
            c[row*n+col+3] = add32(t3, e3);
        }
        for (; col < n; ++col) {
            float total = 0, correction = 0;
            for (unsigned k = 0; k < ksize; ++k)
                accumulate(mul32(a[row*ksize+k], b[k*n+col]), &total, &correction);
            c[row*n+col] = add32(total, correction);
        }
    }
}
