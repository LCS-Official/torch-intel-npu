/* Appended to this project's IEEE helper prefix. Raw INT64 output, F32 weights. */
#define INLINE static inline __attribute__((always_inline))
INLINE unsigned discrete_mulhi(unsigned a, unsigned b) {
    unsigned al=a&65535u, ah=a>>16, bl=b&65535u, bh=b>>16;
    unsigned p=al*bl, q=ah*bl+(p>>16), r=al*bh+(q&65535u);
    return ah*bh+(q>>16)+(r>>16);
}
INLINE U64 random64(const unsigned *h, unsigned index, U64 attempt) {
    unsigned c0=h[2]+index, c1=h[3]+(c0<h[2]), c2=(unsigned)attempt, c3=(unsigned)(attempt>>32);
    unsigned k0=h[0], k1=h[1];
    for (unsigned round=0; round<10; ++round) {
        unsigned hi0=discrete_mulhi(0xd2511f53u,c0), lo0=0xd2511f53u*c0;
        unsigned hi1=discrete_mulhi(0xcd9e8d57u,c2), lo1=0xcd9e8d57u*c2;
        unsigned next0=hi1^c1^k0, next2=hi0^c3^k1;
        c0=next0; c1=lo1; c2=next2; c3=lo0;
        k0+=0x9e3779b9u; k1+=0xbb67ae85u;
    }
    return ((U64)c1<<32)|c0;
}
#if INTEGER_RNG
INLINE U64 remainder64(U64 value, U64 span) {
    U64 remainder=0;
    for (int bit=63; bit>=0; --bit) {
        unsigned carry=(unsigned)(remainder>>63);
        remainder=(remainder<<1)|((value>>bit)&1u);
        if (carry || remainder>=span) remainder-=span;
    }
    return remainder;
}
#endif

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m=(MemRef*)p;
    unsigned words=0;
    for (unsigned t=0; t<3; ++t) {
        if (!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1) return;
        unsigned count=1, *dims=(unsigned*)m[t].dims;
        for (unsigned d=0; d<4; ++d) {
            if (!dims[d] || dims[d]>65536u/count) return;
            count*=dims[d];
        }
        if (t && words!=count) return;
        words=count;
    }
    if (words<18) return;
    const unsigned *h=(const unsigned*)m[0].lo;
    unsigned *out=(unsigned*)m[2].lo;
    for (unsigned i=0; i<words; ++i) out[i]=0;
    out[words-1]=0x3f800000u;
#if INTEGER_RNG
    unsigned count=h[5];
    if (h[4] || !count || count>(words-2)/2) return;
    U64 low=((U64)h[10]<<32)|h[9], span=((U64)h[12]<<32)|h[11];
    U64 threshold=((U64)h[14]<<32)|h[13];
    if (span ? threshold>=span : threshold!=0) return;
    for (unsigned i=0; i<count; ++i) {
        U64 attempt=0, draw;
        do { draw=random64(h,i,attempt); ++attempt; } while (draw<threshold);
        U64 result=low+(span ? remainder64(draw,span) : draw);
        out[2*i]=(unsigned)result; out[2*i+1]=(unsigned)(result>>32);
    }
#else
    unsigned mode=h[4], rows=h[5], columns=h[6], samples=h[7], replacement=h[8];
    if ((mode!=1 && mode!=2) || !rows || !columns || columns>65536u || !samples || samples>(words-2)/2 || replacement>1 ||
        rows>(words-16)/columns || rows>(words-2)/(columns+2*samples) ||
        !replacement && samples>columns) return;
    const float *weights=(const float*)(h+16);
    for (unsigned row=0; row<rows; ++row) {
        unsigned positive=0;
        for (unsigned c=0; c<columns; ++c) {
            unsigned u=bits_of(weights[row*columns+c]), magnitude=u&0x7fffffffu;
            if (magnitude>=0x7f800000u || (u>>31 && magnitude)) return;
            positive+=magnitude!=0;
        }
        if (!positive || (!replacement && samples>positive)) return;
    }
    if (mode==2) {
        float *scratch=(float*)(out+2*rows*samples);
        for (unsigned row=0; row<rows; ++row) {
            for (unsigned c=0; c<columns; ++c) scratch[row*columns+c]=weights[row*columns+c];
            for (unsigned s=0; s<samples; ++s) {
                unsigned maximum_bits=0;
                for (unsigned c=0; c<columns; ++c) {
                    unsigned u=bits_of(scratch[row*columns+c])&0x7fffffffu;
                    if (u>maximum_bits) maximum_bits=u;
                }
                float maximum=from_bits(maximum_bits);
                float total=0.0f;
                for (unsigned c=0; c<columns; ++c) total=add32(total,div32(scratch[row*columns+c],maximum));
                float uniform=(float)((unsigned)random64(h,row*samples+s,0)>>8)*0.000000059604644775390625f;
                float target=mul32(uniform,total), cumulative=0.0f;
                unsigned selected=columns;
                for (unsigned c=0; c<columns; ++c) {
                    float w=div32(scratch[row*columns+c],maximum);
                    cumulative=add32(cumulative,w);
                    unsigned positive=(bits_of(w)&0x7fffffffu)!=0;
                    if (positive) selected=c;
                    if (positive && cumulative>target) break;
                }
                if (selected==columns) return;
                out[2*(row*samples+s)]=selected;
                if (!replacement) scratch[row*columns+selected]=0.0f;
            }
        }
    }
#endif
    out[words-1]=0;
}
