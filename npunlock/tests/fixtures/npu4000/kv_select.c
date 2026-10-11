/* Bounded NPU40xx FP16 KV row copy and attention-mask Select. */
typedef struct { unsigned lo,hi,isStatic,rank,dims,strides,dtype,orderLo,orderHi,location; } MemRef;
typedef unsigned long long U64;
typedef unsigned Word4 __attribute__((ext_vector_type(4)));
typedef unsigned short Half8 __attribute__((ext_vector_type(8)));

void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m=(MemRef*)p;
    unsigned *d[4];U64 *s[4];
    for (unsigned t=0;t<4;++t) {
        if (!m[t].lo || m[t].hi || !m[t].isStatic || m[t].rank!=4 || m[t].dtype!=2 ||
            !m[t].dims || !m[t].strides || m[t].orderLo!=0x4321u || m[t].orderHi || m[t].location!=2) return;
        d[t]=(unsigned*)m[t].dims;s[t]=(U64*)m[t].strides;
    }
    const unsigned short *c=(const unsigned short*)m[0].lo,*x=(const unsigned short*)m[1].lo,*y=(const unsigned short*)m[2].lo;
    unsigned short *z=(unsigned short*)m[3].lo;
    if (d[3][0]!=64 || !d[3][1] || !d[3][2] || d[3][3]!=1 ||
        d[3][1]>64 || d[3][2]>24 || (((unsigned)c|(unsigned)x|(unsigned)y|(unsigned)z)&15u)) return;
    if (d[0][0]==64 && d[0][1]==1 && d[0][2]==1 && d[0][3]==1 &&
        d[3][1]==1 && d[3][2]==1) {
        for (unsigned t=1;t<3;++t)
            if (d[t][0]!=1 || d[t][1]!=1 || d[t][2]!=1 || d[t][3]!=1) return;
        if (s[0][0]!=16 || s[3][0]!=16) return;
        Half8 yes=(Half8)x[0],no=(Half8)y[0];
        for (unsigned i=0;i<64;i+=8) {
            Half8 raw=*(const Half8*)(c+i),mask=(Half8)((raw&0x7fffu)!=0);
            *(Half8*)(z+i)=(mask&yes)|(~mask&no);
        }
        return;
    }
    unsigned rows=d[3][1],heads=d[3][2];
    if (d[0][0]!=1 || d[0][1]!=rows || d[0][2]!=1 || d[0][3]!=1 || s[0][1]!=16 ||
        d[1][0]!=64 || d[1][1]!=1 || d[1][2]!=heads || d[1][3]!=1 || s[1][0]!=16 || s[1][2]!=1024) return;
    for (unsigned t=2;t<4;++t)
        if (d[t][0]!=64 || d[t][1]!=rows || d[t][2]!=heads || d[t][3]!=1 ||
            s[t][0]!=16 || s[t][1]!=1024 || s[t][2]!=1024u*rows) return;
    const Word4 *current=(const Word4*)x,*past=(const Word4*)y;Word4 *out=(Word4*)z;
    for (unsigned h=0;h<heads;++h)
    for (unsigned r=0;r<rows;++r) {
        unsigned target=(h*rows+r)*8;
        const Word4 *in=(c[r]&0x7fffu)?current+h*8:past+target;
        for (unsigned j=0;j<8;++j) out[target+j]=in[j];
    }
}
