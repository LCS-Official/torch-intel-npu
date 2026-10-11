/* Appended to arithmetic.c: same IEEE FP32 helpers and Neumaier accumulation
 * as Multiply+ReduceSum. The host requires one whole, row-major CMX binding.
 * This bounded prototype intentionally supports only square sizes 2..64.
 */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    for (unsigned i = 0; i < 3; ++i)
        if (!m[i].lo || m[i].hi || !m[i].isStatic || m[i].rank != 4 ||
            m[i].dtype != 1 || m[i].location != 2 || !m[i].dims || !m[i].strides ||
            m[i].orderLo != 0x4321u || m[i].orderHi) return;
    unsigned size = ((unsigned*)m[0].dims)[0];
    if (size < 2 || size > 64) return;
    for (unsigned i = 0; i < 3; ++i) {
        unsigned *d = (unsigned*)m[i].dims;
        U64 *s = (U64*)m[i].strides;
        if (d[0] != size || d[1] != size || d[2] != 1 || d[3] != 1 ||
            s[0] != 32 || s[1] != 32u*size ||
            s[2] != 32u*size*size || s[3] != s[2]) return;
    }
    const float *a = (const float*)m[0].lo, *b = (const float*)m[1].lo;
    float *c = (float*)m[2].lo;
    for (unsigned row = 0; row < size; ++row) {
        for (unsigned col = 0; col < size; ++col) {
            float total = 0.0f, correction = 0.0f;
            for (unsigned k = 0; k < size; ++k) {
                float value = mul32(a[row*size+k], b[k*size+col]);
                float next = add32(total, value);
                unsigned left = bits_of(total) & 0x7fffffffu;
                unsigned right = bits_of(value) & 0x7fffffffu;
                if ((bits_of(next) & 0x7f800000u) != 0x7f800000u) {
                    float error = left >= right ? add32(add32(total, -next), value)
                                               : add32(add32(value, -next), total);
                    correction = add32(correction, error);
                } else correction = 0.0f;
                total = next;
            }
            c[row*size+col] = add32(total, correction);
        }
    }
}
