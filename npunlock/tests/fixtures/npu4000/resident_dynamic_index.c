/* Raw gather/scatter/compaction. Only counts and validation flags reach CPU. */
typedef struct { unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location; } MemRef;
static inline __attribute__((always_inline)) unsigned get32(const unsigned char *p){return (unsigned)p[0]|((unsigned)p[1]<<8)|((unsigned)p[2]<<16)|((unsigned)p[3]<<24);}
static inline __attribute__((always_inline)) void put32(unsigned char *p,unsigned v){for(unsigned b=0;b<4;++b)p[b]=(unsigned char)(v>>(8*b));}
void controlled_act(unsigned p){
    if(!p)return;MemRef *m=(MemRef*)p;unsigned words=0;
    for(unsigned t=0;t<3;++t){
        if(!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1)return;
        unsigned n=1;const unsigned *d=(const unsigned*)m[t].dims;
        for(unsigned a=0;a<4;++a){if(!d[a] || d[a]>65536u/n)return;n*=d[a];}
        if(t && n!=words)return;words=n;
    }
    if(words<18)return;
    const unsigned *h=(const unsigned*)m[0].lo;
    unsigned *o=(unsigned*)m[2].lo;for(unsigned i=0;i<words;++i)o[i]=0;
    unsigned count=h[0],op=h[1],width=h[2],base=h[3],span=h[4],total=h[5];
    const unsigned char *a=(const unsigned char*)h+64;
    unsigned char *out=(unsigned char*)o;
#if INDEX_MODE == 1
    if(!count || count>words*4-64 || count>(words-1)/2)return;
    unsigned kept=0;
    for(unsigned i=0;i<count;++i)if(a[i]){o[2*kept]=base+i;o[2*kept+1]=0;++kept;}
    o[words-1]=kept;
#elif INDEX_MODE == 2
    if(!count || count>(words*4-64)/8 || count>(words-1)/2 || !span)return;
    for(unsigned i=0;i<count;++i){
        unsigned lo=get32(a+8*i),hi=get32(a+8*i+4);
        if(hi==0xffffffffu && lo>=0u-span){lo+=span;hi=0;}
        if(hi || lo>=span){o[words-1]=1;return;}
        o[2*i]=lo;o[2*i+1]=0;
    }
#else
    if(!count || !width || width>8 || !span || base>total || span>total-base || count>(words*4-64)/8)return;
    const unsigned char *data=a+count*8,*initial=(const unsigned char*)m[1].lo;
    unsigned bytes=(op==0?count:span)*width;
    if(bytes>words*4-4 || span*width>words*4-64-count*8 && op==0 || count*width>words*4-64-count*8 && op!=0)return;
    for(unsigned i=0;i<bytes;++i)out[i]=initial[i];
    for(unsigned i=0;i<count;++i){
        unsigned idx=get32(a+8*i),hi=get32(a+8*i+4);
        if(hi || idx>=total){o[words-1]=1;return;}
        if(idx<base || idx>=base+span)continue;
        unsigned from=(op==0?idx-base:i)*width,to=(op==0?i:idx-base)*width;
        if(op==2){
            if(h[6]==3)out[to]=(out[to]!=0)||(data[from]!=0);
            else {unsigned carry=0;for(unsigned b=0;b<width;++b){unsigned sum=out[to+b]+data[from+b]+carry;out[to+b]=(unsigned char)sum;carry=sum>>8;}}
        } else for(unsigned b=0;b<width;++b)out[to+b]=data[from+b];
    }
#endif
}
