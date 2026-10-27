/* Complete byte packets. U64 integer arithmetic never passes through FP32. */
static inline __attribute__((always_inline)) unsigned width(unsigned code) {
    return code==0 || code==5 ? 4 : code==1 || code==6 || code==9 ? 2 : code==2 ? 8 : 1;
}
static inline __attribute__((always_inline)) U64 load_u(const unsigned char *p,unsigned n) {
    U64 v=0;for(unsigned b=0;b<n;++b)v|=(U64)p[b]<<(8*b);return v;
}
static inline __attribute__((always_inline)) U64 integer(const unsigned char *p,unsigned code) {
    unsigned n=width(code);U64 v=load_u(p,n);
    if(code!=3 && code!=8 && n<8 && (v&((U64)1<<(8*n-1))))v|=~(((U64)1<<(8*n))-1);
    return v;
}
static inline __attribute__((always_inline)) void store_u(unsigned char *p,U64 v,unsigned n) {
    for(unsigned b=0;b<n;++b)p[b]=(unsigned char)(v>>(8*b));
}
static inline __attribute__((always_inline)) float real(const unsigned char *p,unsigned code) {
    if(code==0)return from_bits((unsigned)load_u(p,4));
    if(code==1){unsigned short h=(unsigned short)load_u(p,2);return load_value(&h,0);}
    if(code==9)return from_bits((unsigned)load_u(p,2)<<16);
    U64 v=integer(p,code);unsigned s=(unsigned)(v>>63);return encode64(s?0-v:v,0,s<<31);
}
static inline __attribute__((always_inline)) U64 trunc_real(float x) {
    unsigned u=bits_of(x),e=(u>>23)&255u;int shift=(int)e-150;
    U64 v=(u&0x7fffffu)|0x800000u;
    if(e==255 || shift>=40)return (U64)1<<63;
    if(shift<=-24)return 0;
    v=shift>=0?v<<shift:v>>-shift;return u>>31?0-v:v;
}
static inline __attribute__((always_inline)) void store_real(unsigned char *p,float v,unsigned code) {
    unsigned u=bits_of(v);
    if(code==0)store_u(p,u,4);
    else if(code==1)store_u(p,half_bits(v),2);
    else if(code==9){
        unsigned a=u&0x7fffffffu;
        store_u(p,a>0x7f800000u?(u>>16)|0x40u:(u+0x7fffu+((u>>16)&1u))>>16,2);
    } else if(code==3)store_u(p,(u&0x7fffffffu)!=0,1);
    else store_u(p,trunc_real(v),width(code));
}
static inline __attribute__((always_inline)) U64 quotient(U64 a,U64 b) {
    U64 q=0,r=0;
    for(int bit=63;bit>=0;--bit){unsigned high=(unsigned)(r>>63);r=(r<<1)|((a>>bit)&1);if(high || r>=b){r-=b;q|=(U64)1<<bit;}}
    return q;
}

void controlled_act(unsigned p) {
    if(!p)return;MemRef *m=(MemRef*)p;unsigned words=0;
    for(unsigned t=0;t<3;++t){
        if(!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1)return;
        unsigned n=1;const unsigned *d=(const unsigned*)m[t].dims;
        for(unsigned a=0;a<4;++a){if(!d[a] || d[a]>65536u/n)return;n*=d[a];}
        if(t && n!=words)return;words=n;
    }
    if(words<16)return;
    const unsigned *h=(const unsigned*)m[0].lo;
    unsigned char *out=(unsigned char*)m[2].lo;
    for(unsigned i=0;i<words*4;++i)out[i]=0;
    unsigned count=h[0],op=h[1],code=h[2],dest=h[3],n=width(code),w=width(dest);
    if(!count || code>9 || dest>9 || count>(words*4-64)/n || count>words*4/w)return;
    const unsigned char *a=(const unsigned char*)h+64,*b=a+count*n;
#if COMPAT_MODE == 0
    for(unsigned i=0;i<count;++i){
        if(code==0 || code==1 || code==9)store_real(out+i*w,real(a+i*n,code),dest);
        else if(dest==0 || dest==1 || dest==9)store_real(out+i*w,real(a+i*n,code),dest);
        else {U64 v=integer(a+i*n,code);store_u(out+i*w,dest==3?v!=0:v,w);}
    }
#elif COMPAT_MODE == 1
    if(code==0 || code==1 || code==9 || count>(words*4-64)/(2*n))return;
    for(unsigned i=0;i<count;++i){
        U64 x=integer(a+i*n,code),y=integer(b+i*n,code),v=0;
        if(op==0)v=x+y;else if(op==1)v=x-y;else if(op==2)v=x*y;
        else if(op==3)v=x&y;else if(op==4)v=x|y;else if(op==5)v=x^y;
        else if(op==6)v=~x;else if(op==7)v=0-x;else if(op==8)v=x>>63?0-x:x;
        else if(op==9 || op==10 || op==11){
            if(!y){((unsigned*)out)[words-1]=1;return;}
            unsigned sx=(unsigned)(x>>63),sy=(unsigned)(y>>63);U64 ax=sx?0-x:x,ay=sy?0-y:y;
            U64 q=quotient(ax,ay),r=ax-q*ay;
            if(op==10)v=r?sy?(U64)0-(sx==sy?r:ay-r):(sx==sy?r:ay-r):0;
            else {v=sx!=sy?0-q:q;if(op==9 && sx!=sy && r)--v;}
        } else if(op>=12 && op<=17){
            unsigned eq=x==y,lt=(x^((U64)1<<63))<(y^((U64)1<<63));
            v=op==12?eq:op==13?!eq:op==14?lt:op==15?lt||eq:op==16?!lt&&!eq:!lt;
        } else if(op==18)v=x?x>>63?(U64)-1:1:0;
        else if(op==19)v=x+y*(((U64)h[5]<<32)|h[4]);
        else if(op==20)v=x-y*(((U64)h[5]<<32)|h[4]);
        else if(op==21){
            if(y>>63)v=x==1?1:x==(U64)-1?(y&1u?(U64)-1:1):0;
            else {v=1;while(y){if(y&1u)v*=x;y>>=1;if(y)x*=x;}}
        }
        store_u(out+i*w,v,w);
    }
#elif COMPAT_MODE == 2
    unsigned rows=h[4],cols=h[5];
    if(!rows || !cols || rows*cols!=count || rows>words*4/w)return;
    for(unsigned r=0;r<rows;++r){U64 total=0;for(unsigned c=0;c<cols;++c)total+=integer(a+(r*cols+c)*n,code);store_u(out+r*w,total,w);}
#endif
}
