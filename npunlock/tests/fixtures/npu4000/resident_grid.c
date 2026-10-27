/* Dynamic grid coordinates -> pixel indices, interpolation weights and their
 * derivatives. Model grid values are read only by SHAVE. IEEE FP32 carrier.
 * Header: dimensions,mode,padding,align,points,width,height,depth.
 */
#define INLINE static __attribute__((always_inline)) inline
INLINE int grid_floor(float x){int i=(int)x;return i-(x<(float)i);}
INLINE float grid_abs(float x){return x<0?-x:x;}
INLINE float grid_map(float x,unsigned n,unsigned padding,unsigned align,float *grad){
    if(padding==2){
        float low=align?0.f:-.5f,span=align?(float)(n-1):(float)n;
        if(span==0){*grad=0;return 0;}
        float distance=x-low,sign=distance<0?-1.f:1.f;distance=grid_abs(distance);
        unsigned flips=(unsigned)(distance/span);float extra=distance-(float)flips*span;
        x=(flips&1u)?span-extra+low:extra+low;*grad*=((flips&1u)?-sign:sign);
    }
    if(padding){if(x<=0){*grad=0;x=0;}else if(x>=(float)(n-1)){*grad=0;x=(float)(n-1);}}
    return x;
}
INLINE float grid_cubic(float distance,float *derivative){
    float sign=distance<0?-1.f:1.f,x=grid_abs(distance),a=-.75f;
    if(x<=1){*derivative=sign*((3*(a+2)*x-2*(a+3))*x);return ((a+2)*x-(a+3))*x*x+1;}
    if(x<2){*derivative=sign*((3*a*x-10*a)*x+8*a);return ((a*x-5*a)*x+8*a)*x-4*a;}
    *derivative=0;return 0;
}
void controlled_act(unsigned p){
    if(!p)return;MemRef *m=(MemRef*)p;unsigned size=0;
    for(unsigned t=0;t<3;++t){
        if(!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1)return;
        unsigned count=1;const unsigned *dims=(const unsigned*)m[t].dims;
        for(unsigned a=0;a<4;++a){if(!dims[a] || dims[a]>65536u/count)return;count*=dims[a];}
        if(t && size!=count)return;size=count;
    }
    unsigned *out=(unsigned*)m[2].lo;for(unsigned i=0;i<size;++i)out[i]=0;if(size<16)return;out[size-1]=0x3f800000u;
    const float *h=(const float*)m[0].lo;
    for(unsigned i=0;i<8;++i)if(!(h[i]>=0 && h[i]<=1048576 && (float)(unsigned)h[i]==h[i]))return;
    unsigned dims=(unsigned)h[0],mode=(unsigned)h[1],padding=(unsigned)h[2],align=(unsigned)h[3],points=(unsigned)h[4];
    unsigned extent[3]={(unsigned)h[5],(unsigned)h[6],(unsigned)h[7]};
    unsigned taps=mode==1?1:mode==2?16:1u<<dims,fields=3+dims;
    if((dims!=2 && dims!=3) || mode>2 || (mode==2 && dims!=2) || padding>2 || align>1 || !points ||
       !extent[0] || !extent[1] || !extent[2] || 8+points*dims>size || points>(size-1)/(taps*fields))return;
    for(unsigned point=0;point<points;++point){
        float coordinate[3],factor[3];int base[3];
        for(unsigned axis=0;axis<dims;++axis){
            float u=h[8+point*dims+axis];unsigned bits=bits_of(u)&0x7fffffffu;if(bits>0x7f800000u)u=-1;
            if(!(u>=-1048576 && u<=1048576))return;
            unsigned n=extent[axis];factor[axis]=(float)(align?n-1:n)*.5f;
            float x=align?((u+1)*.5f)*(float)(n-1):((u+1)*(float)n-1)*.5f;
            if(mode!=2)x=grid_map(x,n,padding,align,&factor[axis]);
            if(!(x>=-1048576 && x<=1048576))return;
            coordinate[axis]=x;base[axis]=grid_floor(x);
        }
        for(unsigned tap=0;tap<taps;++tap){
            int index[3];float w[3],dw[3];unsigned valid=1;
            for(unsigned axis=0;axis<dims;++axis){
                unsigned part=mode==2?(tap>>(2*axis))&3u:(tap>>axis)&1u;
                if(mode==1){float f=coordinate[axis]-(float)base[axis];index[axis]=base[axis]+(f>.5f || (f==.5f && (base[axis]&1)));w[axis]=1;dw[axis]=0;}
                else if(mode==2){index[axis]=base[axis]+(int)part-1;w[axis]=grid_cubic(coordinate[axis]-(float)index[axis],&dw[axis]);dw[axis]*=factor[axis];float unused=1;index[axis]=(int)grid_map((float)index[axis],extent[axis],padding,align,&unused);}
                else{index[axis]=base[axis]+part;float f=coordinate[axis]-(float)base[axis];w[axis]=part?f:1-f;dw[axis]=(part?1.f:-1.f)*factor[axis];}
                if(index[axis]<0 || index[axis]>=(int)extent[axis])valid=0;
            }
            unsigned offset=(point*taps+tap)*fields;
            int pixel=index[1]*(int)extent[0]+index[0];if(dims==3)pixel+=index[2]*(int)extent[1]*(int)extent[0];
            float coefficient=1;for(unsigned a=0;a<dims;++a)coefficient*=w[a];
            out[offset]=valid?(unsigned)pixel:0xffffffffu;
            out[offset+1]=valid?0:0xffffffffu;out[offset+2]=bits_of(coefficient);
            for(unsigned axis=0;axis<dims;++axis){float derivative=dw[axis];for(unsigned a=0;a<dims;++a)if(a!=axis)derivative*=w[a];out[offset+3+axis]=bits_of(derivative);}
        }
    }
    out[size-1]=0;
}
