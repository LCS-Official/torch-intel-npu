/* Power carrier: state=[sum,correction], rhs=[next values,padding].
 * Match resident_reduce.c's per-element Neumaier order without a full K stack.
 * arithmetic.c supplies MemRef and the subnormal-safe FP32 helpers.
 */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned words = 0;
    for (unsigned t = 0; t < 3; ++t) {
        if (!m[t].isStatic || !m[t].rank || m[t].rank > 4 || !m[t].dims ||
            !m[t].lo || m[t].hi || m[t].dtype != 1) return;
        const unsigned *dims = (const unsigned*)m[t].dims;
        unsigned n = 1;
        for (unsigned a = 0; a < m[t].rank; ++a) {
            if (!dims[a] || dims[a] > 65536u/n) return;
            n *= dims[a];
        }
        if (t && n != words) return;
        words = n;
    }
    if (words < 2 || words % 2) return;
    unsigned count = words/2;
    const float *state = (const float*)m[0].lo;
    const float *values = (const float*)m[1].lo;
    float *out = (float*)m[2].lo;
    for (unsigned i = 0; i < count; ++i) {
        float total = state[i], correction = state[count+i], value = values[i];
        float next = add32(total, value);
        if ((bits_of(next) & 0x7f800000u) != 0x7f800000u) {
            unsigned a = bits_of(total) & 0x7fffffffu;
            unsigned b = bits_of(value) & 0x7fffffffu;
            float error = a >= b ? add32(add32(total, -next), value)
                                : add32(add32(value, -next), total);
            correction = add32(correction, error);
        } else correction = 0.0f;
        out[i] = next;
        out[count+i] = correction;
    }
}
