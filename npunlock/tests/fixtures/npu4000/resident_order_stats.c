/* Sorted-row statistics. Values and indices stay in device memory.
 * Input 0: mode,width,integer,queries,interpolation; sorted values; queries.
 * Input 1: original INT64 indices. Output: value(2), ids(2+2), weight, reserved.
 */
static unsigned stat_equal(const unsigned *v,unsigned a,unsigned b,unsigned integer) {
    if (integer) return v[2*a]==v[2*b] && v[2*a+1]==v[2*b+1];
    unsigned x=v[a]&0x7fffffffu,y=v[b]&0x7fffffffu;
    return x<=0x7f800000u && y<=0x7f800000u && (v[a]==v[b] || (!x && !y));
}
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m=(MemRef*)p;unsigned size=0;
    for (unsigned t=0;t<3;++t) {
        if (!m[t].isStatic || m[t].rank!=4 || !m[t].dims || !m[t].lo || m[t].hi || m[t].dtype!=1) return;
        unsigned count=1;const unsigned *dims=(const unsigned*)m[t].dims;
        for (unsigned a=0;a<4;++a) {if (!dims[a] || dims[a]>65536u/count) return;count*=dims[a];}
        if (t && size!=count)return;size=count;
    }
    unsigned *out=(unsigned*)m[2].lo;for (unsigned i=0;i<size;++i)out[i]=0;
    if (size<16)return;
    out[size-1]=0x3f800000u;
    const float *h=(const float*)m[0].lo;
    for (unsigned i=0;i<5;++i)if (!(h[i]>=0 && h[i]<=65536 && (float)(unsigned)h[i]==h[i]))return;
    unsigned mode=(unsigned)h[0],width=(unsigned)h[1],integer=(unsigned)h[2],queries=(unsigned)h[3],interp=(unsigned)h[4];
    if (mode>5 || !width || integer>1 || !queries || queries>(size-1)/8 || width>size/2 ||
        5+width*(1+integer)+queries>size || interp>4 || (mode>=3 && mode!=4 && integer))return;
    const unsigned *values=(const unsigned*)(h+5),*ids=(const unsigned*)m[1].lo;
    const float *q=h+5+width*(1+integer);unsigned finite=width;
    if (!integer)while(finite && (values[finite-1]&0x7fffffffu)>0x7f800000u)--finite;
    for (unsigned query=0;query<queries;++query) {
        unsigned a=0,b=0,weight_bits=0;
        if (mode==0) {if (!(q[query]>=0 && q[query]<width))return;a=b=(unsigned)q[query];}
        else if (mode==1 || mode==4) {
            unsigned n=mode==4?finite:width;
            a=b=(!n || (mode==1 && finite<width))?finite<width?finite:0:(n-1)/2;
        } else if (mode==2) {
            unsigned best=0,run=0,best_count=0;
            for (unsigned j=0;j<=width;++j) {
                if (j<width && (j==run || stat_equal(values,run,j,integer)))continue;
                if (j-run>best_count){best=run;best_count=j-run;}run=j;
            }
            a=b=best;
            // Mode permits any tied index. Choose the last original occurrence.
            for (unsigned j=best+1;j<best+best_count;++j)if(ids[2*j]>ids[2*a])a=b=j;
        } else {
            if (!(q[query]>=0 && q[query]<=1))return;
            unsigned n=mode==5?finite:width;
            if (!n || (mode==3 && finite<width))a=b=finite<width?finite:0;
            else {
                float rank=mul32(q[query],(float)(n-1));a=(unsigned)rank;
                // SAU floating comparisons flush tiny ranks. Positive IEEE
                // bit order detects the upper endpoint without losing them.
                b=a+(bits_of(rank)>bits_of((float)a));
                if(a)weight_bits=bits_of(rank-(float)a);
                else {volatile unsigned tiny=bits_of(rank);weight_bits=tiny;}
                if(interp==1)b=a;
                if(interp==2)a=b;
                if(interp==3)weight_bits=a==b?0:0x3f000000u;
                if(interp==4){unsigned up=weight_bits>0x3f000000u || (weight_bits==0x3f000000u && (a&1u));a=b=up?b:a;}
                if(a==b)weight_bits=0;
            }
        }
        unsigned offset=8*query;
        if(integer){out[offset]=values[2*a];out[offset+1]=values[2*a+1];}
        else {
            out[offset]=values[a];out[offset+7]=values[b];
        }
        out[offset+2]=ids[2*a];out[offset+3]=ids[2*a+1];out[offset+4]=ids[2*b];out[offset+5]=ids[2*b+1];out[offset+6]=weight_bits;
    }
    out[size-1]=0;
}
