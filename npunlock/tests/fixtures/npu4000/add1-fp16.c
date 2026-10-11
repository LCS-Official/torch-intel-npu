
typedef struct { unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location; } MemRef;
void controlled_act(unsigned p) {
    MemRef *m = (MemRef*)p;
    if (!p || !m[0].rank || m[0].rank > 15) return;
    unsigned *dims = (unsigned*)m[0].dims;
    unsigned n = 1;
    for (unsigned i = 0; i < m[0].rank; ++i) n *= dims[i];
    const __fp16 *in = (const __fp16*)m[0].lo;
    __fp16 *out = (__fp16*)m[1].lo;
    for (unsigned i = 0; i < n; ++i) out[i] = (__fp16)((float)in[i] + 1.0f);
}
