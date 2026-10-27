/* Ordinary MaxPool: generate windows on SHAVE from a bounded source span. */
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m=(MemRef*)p;unsigned size=0;
    for(unsigned t=0;t<3;++t){
        if(!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1)return;
        const unsigned *dims=(const unsigned*)m[t].dims;unsigned count=1;
        for(unsigned a=0;a<4;++a){if(!dims[a] || dims[a]>65536u/count)return;count*=dims[a];}
        if(t && count!=size)return;size=count;
    }
    unsigned *out=(unsigned*)m[2].lo;
    for(unsigned i=0;i<size;++i)out[i]=0;
    if(size<18)return;
    out[size-1]=0x3f800000u;
    const unsigned *h=(const unsigned*)m[0].lo;
    for(unsigned i=0;i<16;++i)if(h[i]>1073741824u)return;
    unsigned count=h[0],start=h[1],ih=h[2],iw=h[3],oh=h[4],ow=h[5];
    unsigned kh=h[6],kw=h[7],sh=h[8],sw=h[9],ph=h[10],pw=h[11],dh=h[12],dw=h[13],base=h[14],span=h[15];
    if(!count || !ih || !iw || !oh || !ow || !kh || !kw || !sh || !sw || !dh || !dw || !span ||
       count>(size-1)/3 || span>size-17 || ih>1073741824u/iw || oh>1073741824u/ow ||
       kh>32765u/kw || ph>kh/2 || pw>kw/2 || base>1073741824u-span ||
       start>1073741824u-count || (oh-1)>1073741824u/sh || (ow-1)>1073741824u/sw ||
       (kh-1)>1073741824u/dh || (kw-1)>1073741824u/dw)return;
    unsigned plane=ih*iw,outputs=oh*ow;
    if(count>32768u/(kh*kw))return;
    if((start+count-1)/outputs>1073741823u/plane)return;
    const unsigned *source=h+16;
    unsigned channel=start/outputs,position=start%outputs,oy=position/ow,ox=position%ow;
    for(unsigned i=0;i<count;++i){
        unsigned kept=0,best=0,index=0;
        for(unsigned y=0;y<kh;++y){
            unsigned iy=oy*sh+y*dh;if(iy<ph || iy-ph>=ih)continue;iy-=ph;
            for(unsigned x=0;x<kw;++x){
                unsigned ix=ox*sw+x*dw;if(ix<pw || ix-pw>=iw)continue;ix-=pw;
                unsigned relative=iy*iw+ix,absolute=channel*plane+relative;
                if(absolute<base || absolute-base>=span)return;
                unsigned value=source[absolute-base];
                if(!kept || selection_better(value,best,1) || (value&0x7fffffffu)>0x7f800000u){
                    kept=1;best=value;index=relative;
                }
            }
        }
        if(!kept)return;
        out[i]=best;out[count+2*i]=index;out[count+2*i+1]=0;
        if(++ox==ow){ox=0;if(++oy==oh){oy=0;++channel;}}
    }
    out[size-1]=0;
}
