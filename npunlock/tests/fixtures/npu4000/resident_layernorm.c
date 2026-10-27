/* Welford cascade ordering adapted from PyTorch moments_utils.h (BSD-3-Clause;
 * see LICENSES/PyTorch.txt). Eight FP32 lanes, two halves per FP16 vector.
 * IEEE fused multiply-add below is original integer arithmetic, not libm.
 */
static __attribute__((noinline)) unsigned top_bit(U64 v) {
    unsigned n = 0;
    while (v >>= 1) ++n;
    return n;
}

static U64 align_jam(U64 v, int shift) {
    if (shift >= 0) return v << shift;
    unsigned s = (unsigned)-shift;
    if (s >= 64) return v != 0;
    return (v >> s) | ((v & (((U64)1 << s)-1)) != 0);
}

static __attribute__((noinline)) float fused32(float a, float b, float c) {
    unsigned ua = bits_of(a), ub = bits_of(b), uc = bits_of(c);
    unsigned ea = (ua >> 23) & 255u, eb = (ub >> 23) & 255u, ec = (uc >> 23) & 255u;
    unsigned ma = ua & 0x7fffffu, mb = ub & 0x7fffffu, mc = uc & 0x7fffffu;
    if ((ea == 255 && ma) || (eb == 255 && mb) || (ec == 255 && mc)) return from_bits(0x7fc00000u);
    if (ea == 255 || eb == 255) {
        if (!(ua & 0x7fffffffu) || !(ub & 0x7fffffffu)) return from_bits(0x7fc00000u);
        unsigned sign = (ua ^ ub) & 0x80000000u;
        if (ec == 255 && ((uc ^ sign) >> 31)) return from_bits(0x7fc00000u);
        return from_bits(sign | 0x7f800000u);
    }
    if (ec == 255) return c;
    if (ea) ma |= 0x800000u;
    if (eb) mb |= 0x800000u;
    if (ec) mc |= 0x800000u;
    U64 product = (U64)ma*mb;
    if (!product) return add32(mul32(a, b), c);
    int ep = (ea ? (int)ea-127 : -126)+(eb ? (int)eb-127 : -126)-46;
    if (!mc) return encode64(product, ep, (ua ^ ub) & 0x80000000u);
    int eq = (ec ? (int)ec-127 : -126)-23;
    int tp = (int)top_bit(product)+ep, tq = (int)top_bit(mc)+eq;
    int common = (tp > tq ? tp : tq)-61;
    U64 p = align_jam(product, ep-common), q = align_jam(mc, eq-common);
    long long sp = (ua ^ ub) >> 31 ? -(long long)p : (long long)p;
    long long sq = uc >> 31 ? -(long long)q : (long long)q;
    long long sum = sp+sq;
    return encode64(sum < 0 ? (U64)-sum : (U64)sum, common, sum < 0 ? 0x80000000u : 0);
}

static __attribute__((noinline)) void merge_vector(unsigned *n, float *mean, float *variance,
                                                   unsigned count, const float *m, const float *v) {
    float ratio = *n+count ? div32((float)count, (float)(*n+count)) : 0.0f;
    for (unsigned lane = 0; lane < 8; ++lane) {
        float delta = add32(m[lane], -mean[lane]);
        float change = mul32(ratio, delta);
        mean[lane] = add32(mean[lane], change);
        variance[lane] = fused32(mul32(delta, (float)*n), change, add32(variance[lane], v[lane]));
    }
    *n += count;
}

static __attribute__((noinline)) void row_moments(const float *x, unsigned width, float *result) {
    /* At most 256 chunks for the existing 65,536-element task bound. */
    unsigned counts[8];
    float means[64], variances[64], mean[8], variance[8];
    for (unsigned j = 0; j < 8; ++j) ((volatile unsigned*)counts)[j] = 0;
    for (unsigned j = 0; j < 64; ++j) {
        ((volatile float*)means)[j] = 0.0f;
        ((volatile float*)variances)[j] = 0.0f;
    }
    unsigned vectors = width/16, chunks = (vectors+15)/16, depth = 0;
    for (unsigned k = chunks ? chunks-1 : 0; k; k >>= 1) ++depth;
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        unsigned count = vectors-chunk*16;
        if (count > 16) count = 16;
        for (unsigned half = 0; half < 2; ++half) {
            for (unsigned lane = 0; lane < 8; ++lane) {
                ((volatile float*)mean)[lane] = 0.0f;
                ((volatile float*)variance)[lane] = 0.0f;
            }
            for (unsigned j = 0; j < count; ++j) {
                float reciprocal = div32(1.0f, (float)(j+1));
                for (unsigned lane = 0; lane < 8; ++lane) {
                    float value = x[chunk*256+j*16+half*8+lane];
                    float delta = add32(value, -mean[lane]);
                    mean[lane] = fused32(delta, reciprocal, mean[lane]);
                    variance[lane] = fused32(delta, add32(value, -mean[lane]), variance[lane]);
                }
            }
            merge_vector(counts, means, variances, count, mean, variance);
        }
        unsigned mask = chunk+1;
        for (unsigned level = 1; level < depth && !(mask & 1); ++level, mask >>= 1) {
            merge_vector(counts+level, means+level*8, variances+level*8,
                         counts[level-1], means+(level-1)*8, variances+(level-1)*8);
            counts[level-1] = 0;
            for (unsigned lane = 0; lane < 8; ++lane) {
                means[(level-1)*8+lane] = 0.0f;
                variances[(level-1)*8+lane] = 0.0f;
            }
        }
    }
    for (unsigned level = 1; level < depth; ++level)
        merge_vector(counts, means, variances, counts[level], means+level*8, variances+level*8);
    unsigned count = 0;
    float m = 0.0f, v = 0.0f;
    for (unsigned j = vectors*16; j < width; ++j) {
        float delta = add32(x[j], -m);
        m = add32(m, div32(delta, (float)++count));
        v = add32(v, mul32(delta, add32(x[j], -m)));
    }
    unsigned added = vectors*2;
    for (unsigned lane = 0; lane < 8; ++lane) {
        float ratio = count+added ? div32((float)added, (float)(count+added)) : 0.0f;
        float delta = add32(means[lane], -m);
        m = add32(m, mul32(ratio, delta));
        float adjustment = mul32(mul32(mul32(delta, delta), ratio), (float)count);
        v = add32(v, add32(variances[lane], adjustment));
        count += added;
    }
    result[0] = m;
    result[1] = div32(v, (float)width);
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    for (unsigned i = 0; i < 3; ++i)
        if (!m[i].isStatic || !m[i].rank || m[i].rank > 4 || !m[i].dims || !m[i].lo || m[i].hi || m[i].dtype != 1) return;
#if LN_FMA_TEST
    unsigned n = 1, *dims = (unsigned*)m[0].dims;
    for (unsigned j = 0; j < m[0].rank; ++j) {
        if (!dims[j] || n > 65536u/dims[j]) return;
        n *= dims[j];
    }
    if (n % 3) return;
    const float *x = (const float*)m[0].lo;
    float *out = (float*)m[2].lo;
    for (unsigned j = 0; j < n; j += 3) {
        float value = fused32(x[j], x[j+1], x[j+2]);
        out[j] = value; out[j+1] = value; out[j+2] = value;
    }
#else
    if (m[0].rank != 4) return;
    const unsigned *dims = (const unsigned*)m[0].dims;
    unsigned width = dims[0], rows = dims[1];
    if (dims[2] > 1) { width = dims[2]; rows = 1; }
    if (width < 2 || !rows || rows > 65536u/width) return;
    const float *x = (const float*)m[0].lo, *eps = (const float*)m[1].lo;
    float *out = (float*)m[2].lo;
    for (unsigned row = 0; row < rows; ++row) {
        float stats[2];
        row_moments(x+row*width, width, stats);
        unsigned base = row*width;
        out[base] = stats[0];
        out[base+1] = div32(1.0f, sqrt32(add32(stats[1], eps[base])));
        for (unsigned j = 2; j < width; ++j) out[base+j] = 0.0f;
    }
#endif
}
