/* Native masked-attention inference: explicit binary16 rounding boundaries. */
static float rounded_half(float x) {
    unsigned short h = half_bits(x);
    return load_value(&h, 0);
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
    const float *scores = (const float*)m[0].lo, *mask = (const float*)m[1].lo;
    float *out = (float*)m[2].lo;
    for (unsigned row = 0; row < rows; ++row) {
        unsigned base = row*width, invalid = 0;
        float maximum = from_bits(0xff800000u);
        for (unsigned j = 0; j < width; ++j) {
            if (mask[base+j] != 0.0f) continue;
            float x = scores[base+j];
            if ((bits_of(x)&0x7fffffffu) > 0x7f800000u) invalid = 1;
            if (x > maximum) maximum = x;
        }
        if (invalid || (bits_of(maximum)&0x7fffffffu) == 0x7f800000u) {
            for (unsigned j = 0; j < width; ++j) out[base+j] = from_bits(0x7fc00000u);
            continue;
        }
        float total = 0.0f;
        for (unsigned j = 0; j < width; ++j) {
            float z = mask[base+j] != 0.0f ? 0.0f :
                rounded_half(exp_negative(rounded_half(add32(scores[base+j], -maximum))));
            out[base+j] = z;
            total = add32(total, z);
        }
        float inverse = rounded_half(div32(1.0f, total));
        for (unsigned j = 0; j < width; ++j) out[base+j] = rounded_half(mul32(out[base+j], inverse));
    }
}
