/* Complete raw-byte packet. Comparisons use integer IEEE ordering to preserve
 * subnormal/NaN semantics; selection copies bytes without evaluating a branch.
 */
typedef struct { unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location; } MemRef;
static inline __attribute__((always_inline)) unsigned word(const unsigned char *p, unsigned width) {
    unsigned v = 0;
    for (unsigned i = 0; i < width; ++i) v |= (unsigned)p[i] << (8*i);
    return v;
}
static inline __attribute__((always_inline)) unsigned truth(const unsigned char *p, unsigned code) {
    if (code == 3) return p[0] != 0;
    unsigned a = word(p, code == 1 ? 2 : 4);
    if (code == 2) return (a | word(p+4, 4)) != 0;
    return (a & (code == 1 ? 0x7fffu : 0x7fffffffu)) != 0;
}
static inline __attribute__((always_inline)) unsigned compare(const unsigned char *a, const unsigned char *b, unsigned code, unsigned op) {
    unsigned eq, lt, unordered = 0;
    if (code == 2) {
        unsigned al = word(a,4), ah = word(a+4,4), bl = word(b,4), bh = word(b+4,4);
        eq = al == bl && ah == bh;
        lt = (ah ^ 0x80000000u) < (bh ^ 0x80000000u) || (ah == bh && al < bl);
    } else if (code == 3) {
        unsigned av = a[0] != 0, bv = b[0] != 0;
        eq = av == bv; lt = av < bv;
    } else {
        unsigned width = code == 1 ? 2 : 4;
        unsigned av = word(a,width), bv = word(b,width);
        unsigned sign = code == 1 ? 0x8000u : 0x80000000u;
        unsigned abs_mask = sign-1, inf = code == 1 ? 0x7c00u : 0x7f800000u;
        unordered = (av & abs_mask) > inf || (bv & abs_mask) > inf;
        eq = av == bv || (!(av & abs_mask) && !(bv & abs_mask));
        unsigned ak = av & sign ? ~av & (sign | abs_mask) : av | sign;
        unsigned bk = bv & sign ? ~bv & (sign | abs_mask) : bv | sign;
        lt = !eq && ak < bk;
    }
    if (unordered) return op == 1;
    return op == 0 ? eq : op == 1 ? !eq : op == 2 ? lt : op == 3 ? lt || eq : op == 4 ? !lt && !eq : !lt;
}
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned words = 0;
    for (unsigned t = 0; t < 3; ++t) {
        if (!m[t].isStatic || m[t].rank != 4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype != 1) return;
        unsigned n = 1;
        const unsigned *dims = (const unsigned*)m[t].dims;
        for (unsigned a = 0; a < 4; ++a) {
            if (!dims[a] || dims[a] > 65536u/n) return;
            n *= dims[a];
        }
        if (t && n != words) return;
        words = n;
    }
    if (words < 4) return;
    const float *header = (const float*)m[0].lo;
    if (!(header[0] >= 1 && header[0] <= 65536 && header[1] >= 0 && header[1] <= 12 && header[2] >= 0 && header[2] <= 3)) return;
    unsigned count = (unsigned)header[0], op = (unsigned)header[1], code = (unsigned)header[2];
    if (header[0] != (float)count || header[1] != (float)op || header[2] != (float)code || (op >= 9 && code != 3)) return;
    unsigned width = code == 0 ? 4 : code == 1 ? 2 : code == 2 ? 8 : 1;
    unsigned input = count*(op == 6 ? 1+2*width : op == 8 || op == 9 ? 1 : op == 7 ? width : 2*width);
    unsigned output = count*(op == 6 || op == 8 ? width : 1);
    if (input > words*4-16 || output > words*4) return;
    const unsigned char *data = (const unsigned char*)m[0].lo+16;
    unsigned char *out = (unsigned char*)m[2].lo;
    for (unsigned i = 0; i < words*4; ++i) out[i] = 0;
    for (unsigned i = 0; i < count; ++i) {
        if (op == 6) {
            const unsigned char *selected = data+count+(data[i] ? 0 : count*width)+i*width;
            for (unsigned b = 0; b < width; ++b) out[i*width+b] = selected[b];
        } else if (op == 7) out[i] = truth(data+i*width, code);
        else if (op == 8) {
            unsigned v = data[i] ? code == 0 ? 0x3f800000u : code == 1 ? 0x3c00u : 1 : 0;
            for (unsigned b = 0; b < width && b < 4; ++b) out[i*width+b] = (unsigned char)(v >> (8*b));
        } else if (op == 9) out[i] = !data[i];
        else if (op >= 10) {
            unsigned a = data[i] != 0, b = data[count+i] != 0;
            out[i] = op == 10 ? a && b : op == 11 ? a || b : a != b;
        } else out[i] = compare(data+i*width, data+(count+i)*width, code, op);
    }
}
