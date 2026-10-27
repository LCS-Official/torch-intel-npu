/* Safe row softmax: b[0] is query length (0 disables causal), b[1] row offset. */
static inline float score_at(const float *a, unsigned index, unsigned column, unsigned allowed) {
    return column <= allowed ? a[index] : add32(a[index], from_bits(0xff800000u));
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
    if (!(b[0] >= 0.0f && b[0] <= 1048576.0f)) return;
    unsigned causal = (unsigned)b[0];
    if ((float)causal != b[0]) return;
    unsigned offset = 0;
    /* A one-element tile has one key column, whose causal mask is always open. */
    if (causal && rows*width > 1) {
        if (!(b[1] >= 0.0f && b[1] <= (float)causal)) return;
        offset = (unsigned)b[1];
        if ((float)offset != b[1]) return;
        offset %= causal; /* Legacy uniform-length metadata also means offset zero. */
    }
    for (unsigned row = 0; row < rows; ++row) {
        unsigned base = row*width, allowed = causal ? (row+offset)%causal : width-1;
        float maximum = from_bits(0xff800000u);
        unsigned has_nan = 0;
        for (unsigned j = 0; j < width; ++j) {
            float x = score_at(a, base+j, j, allowed);
            if ((bits_of(x)&0x7fffffffu) > 0x7f800000u) has_nan = 1;
            if (x > maximum) maximum = x;
        }
        if (has_nan || (bits_of(maximum)&0x7fffffffu) == 0x7f800000u) {
            float result = !has_nan && bits_of(maximum) == 0xff800000u ? 0.0f : from_bits(0x7fc00000u);
            for (unsigned j = 0; j < width; ++j) out[base+j] = result;
            continue;
        }
        float total = 0.0f, correction = 0.0f;
        for (unsigned j = 0; j < width; ++j) {
            float v = exp_negative(add32(score_at(a, base+j, j, allowed), -maximum));
            float next = add32(total, v);
            correction = add32(correction, total >= v ? add32(add32(total, -next), v) : add32(add32(v, -next), total));
            total = next;
        }
        total = add32(total, correction);
        for (unsigned j = 0; j < width; ++j)
            out[base+j] = div32(exp_negative(add32(score_at(a, base+j, j, allowed), -maximum)), total);
    }
}
