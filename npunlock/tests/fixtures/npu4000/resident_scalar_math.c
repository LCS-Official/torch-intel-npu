/* FP32 trig uses a 192-bit 2/pi reduction, including large finite arguments. */
static inline __attribute__((always_inline)) unsigned extract(const unsigned *v,unsigned shift){
    unsigned i=shift>>5,b=shift&31u;
    return i>=7?0:(v[i]>>b)|(b && i+1<7?v[i+1]<<(32-b):0);
}
static inline __attribute__((always_inline)) float trig(float x,unsigned cosine){
    unsigned u=bits_of(x),a=u&0x7fffffffu;
    if(a>=0x7f800000u)return from_bits(a>0x7f800000u?u|0x400000u:0x7fc00000u);
    if(!a)return cosine?1.0f:x;
    if(a<0x39800000u)return cosine?1.0f:x;
    float r=x;unsigned quadrant=0;
    if(a>0x3f490fdbu){
        unsigned mant=(a&0x7fffffu)|0x800000u;
        unsigned constants[6]={0x3c439041u,0xdb629599u,0xf534ddc0u,0xfc2757d1u,0x4e441529u,0xa2f9836eu};
        unsigned product[7];U64 carry=0;
        for(unsigned i=0;i<6;++i){U64 v=(U64)constants[i]*mant+carry;product[i]=(unsigned)v;carry=v>>32;}
        product[6]=(unsigned)carry;
        unsigned shift=215u-((a>>23)-127u),fraction=extract(product,shift-32);
        quadrant=extract(product,shift)&3u;
        float f=encode64(fraction,-32,0);
        if(fraction>=0x80000000u){f=f-1.0f;quadrant=(quadrant+1)&3u;}
        r=f*1.570796251296997f+f*0.000000075497894159f;
        if(u>>31){r=-r;quadrant=(0u-quadrant)&3u;}
    }
    float z=r*r;
    float s=-1.0f/39916800.0f+z*(1.0f/6227020800.0f);
    s=1.0f/362880.0f+z*s;s=-1.0f/5040.0f+z*s;
    s=1.0f/120.0f+z*s;s=-1.0f/6.0f+z*s;s=r*(1.0f+z*s);
    float c=-1.0f/3628800.0f+z*(1.0f/479001600.0f);
    c=1.0f/40320.0f+z*c;c=-1.0f/720.0f+z*c;
    c=1.0f/24.0f+z*c;c=-0.5f+z*c;c=1.0f+z*c;
    quadrant=(quadrant+cosine)&3u;
    return quadrant==0?s:quadrant==1?c:quadrant==2?-s:-c;
}
static inline __attribute__((always_inline)) float round_value(float x,unsigned mode){
    unsigned u=bits_of(x),a=u&0x7fffffffu;int e=(int)(a>>23)-127;
    if(e>=23 || !a)return x;
    if(e<0){
        if(mode==2)return from_bits(u&0x80000000u);
        if(mode==0)return u>>31?-1.0f:0.0f;
        if(mode==1)return u>>31?from_bits(0x80000000u):1.0f;
        return a<=0x3f000000u?from_bits(u&0x80000000u):u>>31?-1.0f:1.0f;
    }
    unsigned mask=(1u<<(23-e))-1u,tail=u&mask;float value=from_bits(u&~mask);
    if(!tail)return x;
    if(mode==0)return u>>31?add32(value,-1.0f):value;
    if(mode==1)return u>>31?value:add32(value,1.0f);
    if(mode==2)return value;
    unsigned half=(mask+1)>>1,odd=(u>>(23-e))&1u;
    return tail>half || (tail==half && odd)?add32(value,u>>31?-1.0f:1.0f):value;
}
static inline __attribute__((always_inline)) float power(float x,float y){
    unsigned ux=bits_of(x),uy=bits_of(y),ax=ux&0x7fffffffu,ay=uy&0x7fffffffu;
    if(!ay || ux==0x3f800000u)return 1.0f;
    if(ax>0x7f800000u || ay>0x7f800000u)return from_bits(0x7fc00000u);
    if(ay==0x7f800000u){if(ax==0x3f800000u)return 1.0f;return (ax>0x3f800000u)^(uy>>31)?from_bits(0x7f800000u):0.0f;}
    int e=(int)(ay>>23)-127;unsigned integer_y=0,odd=0,m=(ay&0x7fffffu)|0x800000u;
    if(e>=0){integer_y=e>=23 || !(m&((1u<<(23-e))-1));odd=integer_y && e<=23?((m>>(23-e))&1u):0;}
    if(ux>>31 && ax && ax!=0x7f800000u && !integer_y)return from_bits(0x7fc00000u);
    unsigned sign=ux>>31 && odd?0x80000000u:0;
    if(!ax)return from_bits(sign|(uy>>31?0x7f800000u:0));
    if(ax==0x7f800000u)return from_bits(sign|(uy>>31?0:0x7f800000u));
    float result;
    if(integer_y && ay<=0x42800000u){
        unsigned n=(unsigned)(uy>>31?-y:y);float value=x;result=1.0f;
        while(n){if(n&1u)result=mul32(result,value);n>>=1;if(n)value=mul32(value,value);}
        return uy>>31?div32(1.0f,result):result;
    }
    result=exp_all(mul32(y,log_all(from_bits(ax))));return from_bits(bits_of(result)|sign);
}
void controlled_act(unsigned p){
    if(!p)return;MemRef *m=(MemRef*)p;unsigned words=0;
    for(unsigned t=0;t<3;++t){if(!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1)return;
        unsigned n=1;const unsigned *d=(const unsigned*)m[t].dims;
        for(unsigned a=0;a<4;++a){if(!d[a] || d[a]>65536u/n)return;n*=d[a];}if(t && n!=words)return;words=n;}
    if(words<18)return;const unsigned *h=(const unsigned*)m[0].lo;unsigned count=h[0],op=SCALAR_MODE;
    if(!count || count>(words-16)/2 || op>10)return;
    const float *a=(const float*)(h+16),*b=a+count;float *out=(float*)m[2].lo;
    for(unsigned i=0;i<words;++i)out[i]=0;
    for(unsigned i=0;i<count;++i){float x=a[i];
        if(op<2)out[i]=trig(x,op);
        else if(op==2)out[i]=power(x,b[i]);
        else if(op==3)out[i]=div32(trig(x,0),trig(x,1));
        else if(op>=4 && op<=7)out[i]=round_value(x,op-4);
        else if(op==8){
            if((bits_of(x)&0x7fffffffu)<0x3c800000u){float q=1.0f/5040.0f;q=1.0f/720.0f+x*q;q=1.0f/120.0f+x*q;q=1.0f/24.0f+x*q;q=1.0f/6.0f+x*q;q=0.5f+x*q;out[i]=mul32(x,add32(1.0f,mul32(x,q)));}
            else out[i]=add32(exp_all(x),-1.0f);
        } else if(op==9){
            if((bits_of(x)&0x7fffffffu)<0x3c800000u){float q=1.0f/7.0f;q=-1.0f/6.0f+x*q;q=1.0f/5.0f+x*q;q=-0.25f+x*q;q=1.0f/3.0f+x*q;q=-0.5f+x*q;out[i]=mul32(x,add32(1.0f,mul32(x,q)));}
            else out[i]=log_all(add32(1.0f,x));
        } else out[i]=from_bits(bits_of(x)&0x7fffffffu);
    }
}
