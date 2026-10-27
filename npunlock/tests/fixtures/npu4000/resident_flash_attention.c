/* Online softmax in bounded CMX packets; no sequence-wide score matrix.
 * Header: Q rows, K rows, D, Dv, Q offset, K offset, causal, sqrt(scale) bits,
 * negative scale, mask kind (0/float/BOOL), first block, reserved, ..., output bytes.
 * Payload: Q, K, V, row state [max, sum, unnormalized output], optional dO, mask.
 */
/* Inline only the normal SAU cases. The shared out-of-line helpers retain
 * gradual underflow, signed zero and nonfinite behavior without code duplication.
 */
static inline __attribute__((always_inline)) float flash_add(float a,float b) {
    unsigned ea=(bits_of(a)>>23)&255u,eb=(bits_of(b)>>23)&255u;
    return ea>25u||eb>25u?a+b:add32(a,b);
}
static inline __attribute__((always_inline)) float flash_mul(float a,float b) {
    unsigned ea=(bits_of(a)>>23)&255u,eb=(bits_of(b)>>23)&255u;
    return ea&&eb&&ea<255u&&eb<255u&&ea+eb>128u?a*b:mul32(a,b);
}
static inline __attribute__((always_inline)) float flash_div(float a,float b) {
    unsigned ea=(bits_of(a)>>23)&255u,eb=(bits_of(b)>>23)&255u;
    return ea&&eb&&ea<255u&&eb<255u&&(int)ea-(int)eb>-125?a/b:div32(a,b);
}
static inline void flash_sum(float x,float *total,float *correction) {
    if((bits_of(x)&0x7fffffffu)>=0x7f800000u||(bits_of(*total)&0x7fffffffu)>=0x7f800000u){
        *total=flash_add(*total,x);*correction=0.0f;return;
    }
    float next=flash_add(*total,x);
    *correction=flash_add(*correction,*total>=x?flash_add(flash_add(*total,-next),x):flash_add(flash_add(x,-next),*total));*total=next;
}
static inline float flash_dot(const float *scaled_q,const float *k,unsigned d,float root) {
    float total=0.0f,correction=0.0f;
    for(unsigned a=0;a<d;++a){
        float x=flash_mul(scaled_q[a],flash_mul(k[a],root));
        flash_sum(x,&total,&correction);
    }
    return flash_add(total,correction);
}

void controlled_act(unsigned p) {
    if(!p)return;MemRef *m=(MemRef*)p;unsigned words=0;
    for(unsigned t=0;t<3;++t){
        if(!m[t].isStatic||m[t].rank!=4||!m[t].dims||!m[t].lo||m[t].hi||m[t].dtype!=1)return;
        unsigned n=1;const unsigned *dims=(const unsigned*)m[t].dims;
        for(unsigned a=0;a<4;++a){if(!dims[a]||dims[a]>65536u/n)return;n*=dims[a];}
        if(t&&n!=words)return;words=n;
    }
    if(words<16)return;
    const unsigned *h=(const unsigned*)m[0].lo;float *out=(float*)m[2].lo;
    for(unsigned a=0;a<words;++a)out[a]=0.0f;
    unsigned nr=h[0],ns=h[1],d=h[2],dv=h[3],mask_kind=h[9];
    if(!nr||nr>32||!ns||ns>128||!d||d>128||!dv||dv>128||h[6]>1||h[8]>1||mask_kind>2||h[10]>1)return;
    unsigned qn=nr*d,kn=ns*d,vn=ns*dv,st=nr*(dv+2);
    unsigned input=qn+kn+vn+st;
#if FLASH_BACKWARD
    input+=nr*dv;
    unsigned result=qn+kn+vn;
#else
    unsigned result=st;
#endif
    unsigned mask_bytes=mask_kind==1?nr*ns*4:mask_kind==2?nr*ns:0;
    if(64u+input*4u+mask_bytes>words*4u||result>=words||h[15]!=result*4u+4u)return;
    const float *q=(const float*)(h+16),*k=q+qn,*v=k+kn,*state=v+vn;
    const float *go=state+st;
    const unsigned char *mask=(const unsigned char*)(q+input);
    float root=from_bits(h[7]);
    if(!(root>=0.0f)||bits_of(root)>=0x7f800000u)return;
    for(unsigned r=0;r<nr;++r){
        float score[128],prob[128],ds[128];float maximum=from_bits(0xff800000u);unsigned nan=0;
        float scaled_q[128];
        for(unsigned a=0;a<d;++a)scaled_q[a]=flash_mul(q[r*d+a],h[8]?-root:root);
        for(unsigned j=0;j<ns;++j){
            float x=flash_dot(scaled_q,k+j*d,d,root);
            if(mask_kind==1)x=flash_add(x,((const float*)mask)[r*ns+j]);
            if(mask_kind==2&&!mask[r*ns+j])x=flash_add(x,from_bits(0xff800000u));
            if(h[6]&&h[5]+j>h[4]+r)x=flash_add(x,from_bits(0xff800000u));
            score[j]=x;if(x>maximum)maximum=x;if((bits_of(x)&0x7fffffffu)>0x7f800000u)nan=1;
        }
        const float *old=state+r*(dv+2);
#if FLASH_BACKWARD
        float delta=0.0f;
        for(unsigned a=0;a<dv;++a)delta=flash_add(delta,flash_mul(go[r*dv+a],flash_div(old[2+a],old[1]==0.0f?1.0f:old[1])));
        for(unsigned j=0;j<ns;++j){
            float pr=old[1]==0.0f?0.0f:flash_div(exp_negative(flash_add(score[j],-old[0])),old[1]);
            prob[j]=pr;float dp=0.0f;
            for(unsigned a=0;a<dv;++a)dp=flash_add(dp,flash_mul(go[r*dv+a],v[j*dv+a]));
            ds[j]=flash_mul(pr,flash_add(dp,-delta));
            for(unsigned a=0;a<dv;++a){unsigned z=qn+kn+j*dv+a;out[z]=flash_add(out[z],flash_mul(pr,go[r*dv+a]));}
            for(unsigned a=0;a<d;++a){unsigned z=qn+j*d+a;out[z]=flash_add(out[z],flash_mul(flash_mul(ds[j],scaled_q[a]),root));}
        }
        for(unsigned a=0;a<d;++a){
            float total=0.0f;
            for(unsigned j=0;j<ns;++j)total=flash_add(total,flash_mul(flash_mul(ds[j],flash_mul(k[j*d+a],root)),h[8]?-root:root));
            out[r*d+a]=total;
        }
#else
        float previous=h[10]?from_bits(0xff800000u):old[0];
        if(previous>maximum)maximum=previous;
        if(nan||(bits_of(previous)&0x7fffffffu)>0x7f800000u)maximum=from_bits(0x7fc00000u);
        float safe=bits_of(maximum)==0xff800000u?0.0f:maximum;
        float alpha=h[10]?0.0f:exp_negative(flash_add(previous,-safe));
        float sum=0.0f,correction=0.0f;
        for(unsigned j=0;j<ns;++j){
            float x=exp_negative(flash_add(score[j],-safe));prob[j]=x;
            float next=flash_add(sum,x);correction=flash_add(correction,sum>=x?flash_add(flash_add(sum,-next),x):flash_add(flash_add(x,-next),sum));sum=next;
        }
        unsigned base=r*(dv+2);out[base]=maximum;
        out[base+1]=flash_add(flash_mul(alpha,h[10]?0.0f:old[1]),flash_add(sum,correction));
        for(unsigned a=0;a<dv;++a){
            float total=0.0f,comp=0.0f;
            for(unsigned j=0;j<ns;++j)flash_sum(flash_mul(prob[j],v[j*dv+a]),&total,&comp);
            out[base+2+a]=flash_add(flash_mul(alpha,h[10]?0.0f:old[2+a]),flash_add(total,comp));
        }
#endif
    }
}
