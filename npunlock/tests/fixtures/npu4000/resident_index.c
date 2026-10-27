/* A single complete F32 packet: header, raw INT64 indices, F32 payload.
 * All indices are checked before any indexed memory access; only the final
 * validation flag is read by the host. Appended to arithmetic.c helpers.
 */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned size = 0;
    for (unsigned t = 0; t < 3; ++t) {
        if (!m[t].isStatic || m[t].rank != 4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype != 1) return;
        unsigned count = 1;
        const unsigned *dims = (const unsigned*)m[t].dims;
        for (unsigned axis = 0; axis < 4; ++axis) {
            if (!dims[axis] || dims[axis] > 65536u/count) return;
            count *= dims[axis];
        }
        if (t && size != count) return;
        size = count;
    }
    if (size < 9) return;
    const float *a = (const float*)m[0].lo;
    unsigned *out = (unsigned*)m[2].lo;
    for (unsigned i = 0; i < size; ++i) out[i] = 0;
    out[size-1] = 0x3f800000u;
    if (!(a[0] >= 1.0f && a[0] <= 65536.0f && a[1] >= 1.0f && a[1] <= 65536.0f &&
          a[2] >= 1.0f && a[2] <= 65536.0f && a[3] >= 0.0f && a[3] <= a[0] &&
          (a[4] == 0.0f || a[4] == 1.0f || (INDEX_BACKWARD && (a[4] == 2.0f || a[4] == 3.0f))))) return;
    unsigned rows = (unsigned)a[0], width = (unsigned)a[1], count = (unsigned)a[2];
    unsigned padding = (unsigned)a[3], flags = (unsigned)a[4], half = flags & 1u;
    if ((float)rows != a[0] || (float)width != a[1] || (float)count != a[2] || (float)padding != a[3]) return;
#if INDEX_BACKWARD == 2
    if (count > (size-6)/2) return;
    const unsigned *raw = (const unsigned*)(a+5);
    for (unsigned i = 0; i < count; ++i) {
        unsigned lo = raw[2*i], hi = raw[2*i+1], negative = hi >> 31;
        if (negative) { lo = ~lo+1u; hi = ~hi+(lo == 0); }
        unsigned remainder = 0;
        for (unsigned bit = 0; bit < 64; ++bit) {
            unsigned digit = bit < 32 ? (hi >> (31-bit)) & 1u : (lo >> (63-bit)) & 1u;
            remainder = (remainder << 1) | digit;
            if (remainder >= rows) remainder -= rows;
        }
        out[2*i] = negative && remainder ? rows-remainder : remainder;
        out[2*i+1] = 0;
    }
    out[size-1] = 0;
    return;
#endif
    if (rows > 65536u/width || count > 65536u/width || count > (size-6)/2) return;
    if (rows > 1048576u/(count*width)) return;
    unsigned source_count = INDEX_BACKWARD ? count*width : rows*width;
    unsigned result_count = INDEX_BACKWARD ? rows*width : count*width;
    if (source_count > size-6-2*count || result_count >= size) return;
    const unsigned *indices = (const unsigned*)(a+5);
    const unsigned *source = indices+2*count;
    for (unsigned i = 0; i < count; ++i)
        if (indices[2*i+1] || indices[2*i] >= rows) return;
#if INDEX_BACKWARD
    const float *gradient = (const float*)source;
    for (unsigned row = 0; row < rows; ++row) {
        if (padding && row == padding-1) continue;
        for (unsigned c = 0; c < width; ++c) {
            /* Scatter starts at self, preserving the order of cancellation and half opmath. */
            float sum = flags & 2u ? ((const float*)m[1].lo)[row*width+c] : 0.0f;
            for (unsigned i = 0; i < count; ++i) {
                if (indices[2*i] != row) continue;
                sum = add32(sum, gradient[i*width+c]);
                if (half) {
                    unsigned short h = half_bits(sum);
                    sum = load_value(&h, 0);
                }
            }
            out[row*width+c] = bits_of(sum);
        }
    }
#else
    for (unsigned i = 0; i < count; ++i)
        for (unsigned c = 0; c < width; ++c)
            out[i*width+c] = source[indices[2*i]*width+c];
#endif
    out[size-1] = 0;
}
