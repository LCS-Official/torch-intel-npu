/* Original row softmax, appended to arithmetic/classification math helpers. */
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
    for (unsigned row = 0; row < rows; ++row) {
        unsigned base = row*width;
#if SOFTMAX_BACKWARD
        float total = 0.0f, correction = 0.0f;
        for (unsigned j = 0; j < width; ++j) {
            float v = mul32(a[base+j], b[base+j]), next = add32(total, v);
            if ((bits_of(total)&0x7fffffffu) < 0x7f800000u && (bits_of(v)&0x7fffffffu) < 0x7f800000u &&
                (bits_of(next)&0x7fffffffu) < 0x7f800000u)
                correction = add32(correction, __builtin_fabsf(total) >= __builtin_fabsf(v) ?
                    add32(add32(total, -next), v) : add32(add32(v, -next), total));
            else correction = 0.0f;
            total = next;
        }
        total = add32(total, correction);
        for (unsigned j = 0; j < width; ++j)
            out[base+j] = mul32(a[base+j], add32(b[base+j], -total));
#else
        float maximum = from_bits(0xff800000u);
        unsigned has_nan = 0;
        for (unsigned j = 0; j < width; ++j) {
            float x = a[base+j];
            if ((bits_of(x)&0x7fffffffu) > 0x7f800000u) has_nan = 1;
            if (x > maximum) maximum = x;
        }
        if (has_nan || (bits_of(maximum)&0x7fffffffu) == 0x7f800000u) {
            for (unsigned j = 0; j < width; ++j) out[base+j] = from_bits(0x7fc00000u);
            continue;
        }
        float total = 0.0f, correction = 0.0f;
        for (unsigned j = 0; j < width; ++j) {
            float v = exp_negative(add32(a[base+j], -maximum)), next = add32(total, v);
            correction = add32(correction, total >= v ? add32(add32(total, -next), v) : add32(add32(v, -next), total));
            total = next;
        }
        total = add32(total, correction);
        for (unsigned j = 0; j < width; ++j)
            out[base+j] = div32(exp_negative(add32(a[base+j], -maximum)), total);
#endif
    }
}
