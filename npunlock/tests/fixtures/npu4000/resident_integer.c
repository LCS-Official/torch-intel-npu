/* The F32 carrier transports two raw little-endian words, not float values. */
typedef struct {
    unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location;
} MemRef;

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    for (unsigned i = 0; i < 3; ++i) {
        if (!m[i].isStatic || !m[i].rank || m[i].rank > 4 || !m[i].dims ||
            !m[i].lo || m[i].hi || m[i].dtype != 1) return;
        unsigned count = 1;
        const unsigned *dims = (const unsigned*)m[i].dims;
        for (unsigned axis = 0; axis < m[i].rank; ++axis) {
            if (!dims[axis] || count > 2 / dims[axis]) return;
            count *= dims[axis];
        }
        if (count != 2) return;
    }
    const unsigned *a = (const unsigned*)m[0].lo;
    const unsigned *b = (const unsigned*)m[1].lo;
    unsigned *out = (unsigned*)m[2].lo;
    unsigned low = a[0] + b[0];
    out[0] = low;
    out[1] = a[1] + b[1] + (low < a[0]);
}
