/* Appended to arithmetic.c by the builder; reuses its subnormal-safe FP32 ops.
 * MemRef dimensions use the compiler's physical order, with bit strides.
 * The host validator rejects partial reductions and overlapping buffers.
 */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned rank = m[0].rank;
    if (!rank || rank > 4 || m[1].rank != rank) return;
    for (unsigned k = 0; k < 2; ++k)
        if (!m[k].lo || m[k].hi || !m[k].isStatic || m[k].dtype != 1 ||
            !m[k].dims || !m[k].strides) return;
    unsigned *in_dims = (unsigned*)m[0].dims, *out_dims = (unsigned*)m[1].dims;
    U64 *in_strides = (U64*)m[0].strides, *out_strides = (U64*)m[1].strides;
    unsigned steps[4], reduced[4], reduced_axes = 0;
    unsigned outputs = 1, count = 1;
    for (unsigned axis = 0; axis < rank; ++axis) {
        if (!in_dims[axis] || !out_dims[axis] ||
            (out_dims[axis] != 1 && out_dims[axis] != in_dims[axis]) ||
            in_strides[axis] % 32 || out_strides[axis] % 32 ||
            outputs > 65536u / out_dims[axis]) return;
        outputs *= out_dims[axis];
        steps[axis] = (unsigned)(in_strides[axis] / 32);
        if (out_dims[axis] == 1) {
            if (count > 65536u / in_dims[axis]) return;
            count *= in_dims[axis];
            if (in_dims[axis] > 1) reduced[reduced_axes++] = axis;
        }
    }
    if (outputs > 65536u / count) return;
    const float *input = (const float*)m[0].lo;
    float *output = (float*)m[1].lo;
    /* Compensate cancellation before rounding a dot product back to FP16.
     * Sequential sums can land on the opposite side of a half midpoint and
     * change a downstream ReLU mask. Keep the subnormal-safe FP32 helpers.
     */
    for (unsigned i = 0; i < outputs; ++i) {
        unsigned index = i, base = 0, dst = 0;
        for (unsigned axis = 0; axis < rank; ++axis) {
            unsigned coordinate = index % out_dims[axis];
            index /= out_dims[axis];
            base += coordinate * (unsigned)(in_strides[axis] / 32);
            dst += coordinate * (unsigned)(out_strides[axis] / 32);
        }
        float total = 0.0f, correction = 0.0f;
        unsigned src = base, coordinates[4];
        /* The bare SHAVE image has no CRT memset; force bounded scalar stores
         * for initialization, while keeping the hot coordinates nonvolatile. */
        for (unsigned k = 0; k < reduced_axes; ++k)
            ((volatile unsigned*)coordinates)[k] = 0;
        unsigned step = reduced_axes == 1 ? steps[reduced[0]] : 0;
        for (unsigned j = 0; j < count; ++j) {
            float value = input[src], next = add32(total, value);
            unsigned a = bits_of(total) & 0x7fffffffu;
            unsigned b = bits_of(value) & 0x7fffffffu;
            if ((bits_of(next) & 0x7f800000u) != 0x7f800000u) {
                float error = a >= b ? add32(add32(total, -next), value)
                                    : add32(add32(value, -next), total);
                correction = add32(correction, error);
            } else correction = 0.0f; /* Preserve Inf/NaN sum semantics. */
            total = next;
            if (reduced_axes <= 1) src += step;
            else {
                /* Same physical traversal order as mixed-radix division, but
                 * no integer division/modulo in the per-value loop. */
                for (unsigned k = 0; k < reduced_axes; ++k) {
                    unsigned axis = reduced[k];
                    src += steps[axis];
                    if (++coordinates[k] < in_dims[axis]) break;
                    coordinates[k] = 0;
                    src -= in_dims[axis] * steps[axis];
                }
            }
        }
        total = add32(total, correction);
#if REDUCE_MEAN
        total = div32(total, (float)count);
#endif
        output[dst] = total;
    }
}
