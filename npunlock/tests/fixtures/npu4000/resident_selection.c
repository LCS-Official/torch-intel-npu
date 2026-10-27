/* Source-backed sorted selection and linear indexed gradient accumulation.
 * FP32 is also the raw-word carrier for INT64 indices; no numeric index cast.
 */
static unsigned selection_better(unsigned a, unsigned b, unsigned largest) {
    unsigned aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
    if (aa > 0x7f800000u || bb > 0x7f800000u)
        return largest ? aa > 0x7f800000u && bb <= 0x7f800000u
                       : aa <= 0x7f800000u && bb > 0x7f800000u;
    if (!aa && !bb) return 0;
    unsigned ka = a & 0x80000000u ? ~a : a ^ 0x80000000u;
    unsigned kb = b & 0x80000000u ? ~b : b ^ 0x80000000u;
    return largest ? ka > kb : ka < kb;
}

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    unsigned size = 0;
    for (unsigned t = 0; t < 3; ++t) {
        if (!m[t].isStatic || m[t].rank != 4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype != 1) return;
        unsigned count = 1;
        const unsigned *dims = (const unsigned*)m[t].dims;
        for (unsigned a = 0; a < 4; ++a) {
            if (!dims[a] || dims[a] > 65536u/count) return;
            count *= dims[a];
        }
        if (t && size != count) return;
        size = count;
    }
    const float *h = (const float*)m[0].lo;
    unsigned *out = (unsigned*)m[2].lo;
    for (unsigned i = 0; i < size; ++i) out[i] = 0;
#if SELECTION_MODE == 3
    if (size<12)return;
    out[size-1]=0x3f800000u;
    for(unsigned i=0;i<8;++i)if(!(h[i]>=0 && h[i]<=16777216 && (float)(unsigned)h[i]==h[i]))return;
    unsigned count=(unsigned)h[0],start=(unsigned)h[1],per=(unsigned)h[2],width=(unsigned)h[3];
    unsigned base=(unsigned)h[4],span=(unsigned)h[5],total=(unsigned)h[7];
    if(!count || !per || !width || !span || count>=size || 8+2*count+span>size || base+span>total || (start+count-1)/per>=total/width)return;
    const unsigned *ids=(const unsigned*)(h+8),*data=ids+2*count,*initial=(const unsigned*)m[1].lo;
    for(unsigned i=0;i<count;++i){
        if(ids[2*i]==0xffffffffu && ids[2*i+1]==0xffffffffu){out[i]=0;continue;}
        if(ids[2*i+1] || ids[2*i]>=width)return;
        unsigned source=((start+i)/per)*width+ids[2*i];
        out[i]=(source>=base && source<base+span)?data[source-base]:initial[i];
    }
    out[size-1]=0;
#elif SELECTION_MODE == 2 || SELECTION_MODE == 4
    if (size < 12) return;
    out[size-1] = 0x3f800000u;
#if SELECTION_MODE == 4
    /* Raw uint32 metadata avoids FP32 integer rounding above 2^24. */
    const unsigned *header = (const unsigned*)h;
    for (unsigned i = 0; i < 8; ++i) if (header[i] > 1073741824u) return;
    unsigned count = header[0], start = header[1], per = header[2], width = header[3];
    unsigned base = header[4], span = header[5], flags = header[6], total = header[7];
#else
    for (unsigned i = 0; i < 8; ++i)
        if (!(h[i] >= 0 && h[i] <= 16777216 && (float)(unsigned)h[i] == h[i])) return;
    unsigned count = (unsigned)h[0], start = (unsigned)h[1], per = (unsigned)h[2], width = (unsigned)h[3];
    unsigned base = (unsigned)h[4], span = (unsigned)h[5], flags = (unsigned)h[6], total = (unsigned)h[7];
#endif
    if (!count || !per || !width || !span || flags > 7 || count > (size-9)/3 || span >= size ||
        base+span > total || (start+count-1)/per >= total/width) return;
    const float *grad = h+8, *initial = (const float*)m[1].lo;
    const unsigned *ids = (const unsigned*)(grad+count);
    for (unsigned i = 0; i < count; ++i) {
        if ((flags&4u) && ids[2*i]==0xffffffffu && ids[2*i+1]==0xffffffffu)continue;
        if (ids[2*i+1] || ids[2*i] >= width) return;
    }
    for (unsigned i = 0; i < span; ++i) out[i] = bits_of(initial[i]);
    for (unsigned i = 0; i < count; ++i) {
        if ((flags&4u) && ids[2*i]==0xffffffffu && ids[2*i+1]==0xffffffffu)continue;
        unsigned target = ((start+i)/per)*width+ids[2*i];
        if (target < base || target >= base+span) continue;
        unsigned pos = target-base;
        float sum = flags & 2u ? grad[i] : add32(from_bits(out[pos]),grad[i]);
        if ((flags & 1u) && !(flags & 2u)) { unsigned short code = half_bits(sum); sum = load_value(&code,0); }
        out[pos] = bits_of(sum);
    }
    out[size-1] = 0;
#else
    if (size < 7) return;
    for (unsigned i = 0; i < 5; ++i)
        if (!(h[i] >= 0 && h[i] <= 65536 && (float)(unsigned)h[i] == h[i])) return;
    unsigned rows = (unsigned)h[0], width = (unsigned)h[1], k = (unsigned)h[2];
    unsigned largest = (unsigned)h[3], supplied = (unsigned)h[4];
    if (!(h[5]>=0 && h[5]<=1048576 && (float)(unsigned)h[5]==h[5])) return;
    unsigned offset = (unsigned)h[5];
    if (!rows || !width || !k || k > 128 || k > width || largest > 1 || supplied > 1 ||
        rows > (size-6)/width || rows > size/(3*k) || offset+width>1048576 || (supplied && rows*width > size/2)) return;
    const unsigned *values = (const unsigned*)(h+6), *ids = (const unsigned*)m[1].lo;
    for (unsigned row = 0; row < rows; ++row) {
        unsigned kept = 0, offset = row*k;
        for (unsigned j = 0; j < width; ++j) {
            unsigned candidate = values[row*width+j], index = row*width+j;
#if SELECTION_MODE == 1
            if (ids[2*index+1]) continue; /* Padded window entries have INT64 -1. */
#endif
            unsigned position = kept;
            for (unsigned q = 0; q < kept; ++q) {
                unsigned replace = selection_better(candidate,out[offset+q],largest);
#if SELECTION_MODE == 1
                if ((candidate & 0x7fffffffu) > 0x7f800000u) replace = 1; /* MaxPool uses last NaN. */
#endif
                if (replace) { position = q; break; }
            }
            if (position >= k) continue;
            if (kept < k) ++kept;
            for (unsigned q = kept-1; q > position; --q) {
                out[offset+q] = out[offset+q-1];
                out[rows*k+2*(offset+q)] = out[rows*k+2*(offset+q-1)];
                out[rows*k+2*(offset+q)+1] = out[rows*k+2*(offset+q-1)+1];
            }
            out[offset+position] = candidate;
            out[rows*k+2*(offset+position)] = supplied ? ids[2*index] : (unsigned)h[5]+j;
            out[rows*k+2*(offset+position)+1] = supplied ? ids[2*index+1] : 0;
        }
    }
#endif
}
