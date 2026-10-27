/* Stable floating sort. Indices are raw INT64 words, never floating casts.
 * Mode 0: bounded bottom-up merge sort, with output-resident scratch.
 * Mode 1: a tiled bitonic compare/exchange for rows larger than a carrier.
 */
static unsigned sort_before(unsigned a, unsigned ai, unsigned b, unsigned bi, unsigned descending) {
    if (ai == 0xffffffffu || bi == 0xffffffffu) return ai < bi;
    unsigned aa = a & 0x7fffffffu, bb = b & 0x7fffffffu;
    unsigned an = aa > 0x7f800000u, bn = bb > 0x7f800000u;
    if (an != bn) return descending ? an : bn;
    if (an || (!aa && !bb) || a == b) return ai < bi;
    unsigned ka = a & 0x80000000u ? ~a : a ^ 0x80000000u;
    unsigned kb = b & 0x80000000u ? ~b : b ^ 0x80000000u;
    return descending ? ka > kb : ka < kb;
}

static unsigned integer_before(const unsigned *a, unsigned ai, const unsigned *b, unsigned bi, unsigned descending) {
    if (ai == 0xffffffffu || bi == 0xffffffffu) return ai < bi;
    if (a[0] == b[0] && a[1] == b[1]) return ai < bi;
    unsigned ah = a[1] ^ 0x80000000u, bh = b[1] ^ 0x80000000u;
    unsigned less = ah == bh ? a[0] < b[0] : ah < bh;
    return descending ? !less : less;
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
    unsigned *out = (unsigned*)m[2].lo;
    for (unsigned i = 0; i < size; ++i) out[i] = 0;
    if (size < 13) return;
    const float *h = (const float*)m[0].lo;
    for (unsigned i = 0; i < 7; ++i)
        if (!(h[i] >= 0 && h[i] <= 1048576 && (float)(unsigned)h[i] == h[i])) return;
    unsigned mode = (unsigned)h[0], rows = (unsigned)h[1], width = (unsigned)h[2], desc = (unsigned)h[3];
    if (mode > 3 || !rows || !width || desc > 1 || rows > (size-7)/width) return;
    unsigned n = rows*width;
    const unsigned *in = (const unsigned*)(h+7), *other = (const unsigned*)m[1].lo;
    if (mode >= 2) {
        if (mode == 2) {
            if (n > size/8) return;
            for (unsigned i=0;i<n;++i) {out[2*i]=in[2*i];out[2*i+1]=in[2*i+1];out[2*n+2*i]=i%width;}
            for (unsigned row=0;row<rows;++row) {
                unsigned *src=out,*dst=out+4*n;
                for (unsigned run=1;run<width;run*=2) {
                    for (unsigned base=0;base<width;base+=2*run) {
                        unsigned l=base,mid=base+run<width?base+run:width,r=mid,end=base+2*run<width?base+2*run:width;
                        for (unsigned q=base;q<end;++q) {
                            unsigned li=row*width+l,ri=row*width+r;
                            unsigned take=l<mid && (r==end || integer_before(src+2*li,src[2*n+2*li],src+2*ri,src[2*n+2*ri],desc));
                            unsigned pos=row*width+(take?l++:r++),target=row*width+q;
                            dst[2*target]=src[2*pos];dst[2*target+1]=src[2*pos+1];dst[2*n+2*target]=src[2*n+2*pos];dst[2*n+2*target+1]=0;
                        }
                    }
                    unsigned *temp=src;src=dst;dst=temp;
                }
                if (src!=out) for (unsigned q=row*width;q<(row+1)*width;++q) {
                    out[2*q]=src[2*q];out[2*q+1]=src[2*q+1];out[2*n+2*q]=src[2*n+2*q];out[2*n+2*q+1]=0;
                }
            }
        } else {
            unsigned pos=(unsigned)h[4],stage=(unsigned)h[5],stride=(unsigned)h[6];
            if (rows!=1 || n>(size-7)/4 || !stage || !stride || stride>=stage || pos+n>524288) return;
            for (unsigned i=0;i<n;++i) {
                unsigned ai=in[2*n+2*i],bi=other[2*n+2*i];
                unsigned first=integer_before(in+2*i,ai,other+2*i,bi,desc);
                unsigned take=first==((((pos+i)&stage)==0)==(((pos+i)&stride)==0));
                const unsigned *v=take?in:other;
                out[2*i]=v[2*i];out[2*i+1]=v[2*i+1];out[2*n+2*i]=take?ai:bi;out[2*n+2*i+1]=0;
            }
        }
        return;
    }
    if (!mode) {
        if (n > size/6) return;
        for (unsigned i = 0; i < n; ++i) { out[i] = in[i]; out[n+2*i] = i%width; }
        for (unsigned row = 0; row < rows; ++row) {
            unsigned *src = out, *dst = out+3*n;
            for (unsigned run = 1; run < width; run *= 2) {
                for (unsigned base = 0; base < width; base += 2*run) {
                    unsigned l = base, mid = base+run < width ? base+run : width, r = mid;
                    unsigned end = base+2*run < width ? base+2*run : width;
                    for (unsigned q = base; q < end; ++q) {
                        unsigned take = l < mid && (r == end || sort_before(src[row*width+l],src[n+2*(row*width+l)],
                                                                          src[row*width+r],src[n+2*(row*width+r)],desc));
                        unsigned pos = row*width+(take ? l++ : r++), target = row*width+q;
                        dst[target] = src[pos]; dst[n+2*target] = src[n+2*pos]; dst[n+2*target+1] = 0;
                    }
                }
                unsigned *temp = src; src = dst; dst = temp;
            }
            if (src != out) for (unsigned q = row*width; q < (row+1)*width; ++q) {
                out[q] = src[q]; out[n+2*q] = src[n+2*q]; out[n+2*q+1] = 0;
            }
        }
    } else {
        unsigned pos = (unsigned)h[4], stage = (unsigned)h[5], stride = (unsigned)h[6];
        if (rows != 1 || n > (size-7)/3 || !stage || !stride || stride >= stage || pos+n > 524288) return;
        for (unsigned i = 0; i < n; ++i) {
            unsigned a = in[i], ai = in[n+2*i], b = other[i], bi = other[n+2*i];
            unsigned forward = ((pos+i)&stage) == 0, low = ((pos+i)&stride) == 0;
            unsigned first = sort_before(a,ai,b,bi,desc), take = first == (forward == low);
            out[i] = take ? a : b; out[n+2*i] = take ? ai : bi; out[n+2*i+1] = 0;
        }
    }
}
