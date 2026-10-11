/* Fixed width=576, eps=1e-5. FP16 storage, genuine FP32 RMS/rsqrt.
 * Appended to arithmetic.c and resident_activation.c helper prefixes.
 * Preserve the official half rounding before multiplication by gamma. */
static inline float load_half(const unsigned short *p, unsigned i) {
    /* Hardware converts normal halves directly; bit decoding preserves half
     * subnormals and signed zero which hardware conversion can flush. */
    if (p[i]&0x7c00u) return (float)((const __fp16*)p)[i];
    return load_value(p,i);
}

static inline unsigned short round_half(float value) {
    unsigned magnitude=bits_of(value)&0x7fffffffu;
    if (magnitude>=0x38800000u && magnitude<=0x477fe000u) {
        union { __fp16 h; unsigned short u; } result;
        result.h=(__fp16)value;
        return result.u;
    }
    return half_bits(value); /* Subnormals, zero, overflow and special values. */
}


typedef float Float4 __attribute__((ext_vector_type(4)));
typedef unsigned UInt4 __attribute__((ext_vector_type(4)));
typedef unsigned short HalfBits4 __attribute__((ext_vector_type(4)));

/* SIMD adaptation of FP16's IEEE conversion algorithm (MIT).
 * Copyright (c) 2017 Facebook Inc., 2017 Georgia Institute of Technology,
 * Copyright 2019 Google LLC. Full terms: rmsnorm/FP16_LICENSE.
 * https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h
 * Integer extension avoids this toolchain's broken half->float convertvector.
 * The subnormal path uses normal FP32 values, so hardware FTZ cannot erase it. */
static inline Float4 decode_bits4(HalfBits4 raw) {
    UInt4 h=__builtin_convertvector(raw,UInt4),two=h<<17;
    union { UInt4 u;Float4 f; } normal,small,result;
    normal.u=(two>>4)+0x70000000u;
    normal.f=normal.f*0x1.0p-112f;
    small.u=(two>>17)|0x3f000000u;
    small.f=small.f-0.5f;
    UInt4 mask=(UInt4)(two<0x08000000u);
    result.u=((h&0x8000u)<<16)|(small.u&mask)|(normal.u&~mask);
    return result.f;
}
static inline Float4 decode4(const unsigned short *p) {
    return decode_bits4(*(const HalfBits4*)p);
}
static inline HalfBits4 encode4(Float4 value) {
    union { UInt4 u; Float4 f; } bits;
    bits.f=value;
    UInt4 magnitude=bits.u&0x7fffffffu;
    /* Unsigned subtraction combines the normal-value lower/upper tests. */
    UInt4 valid=(UInt4)((magnitude-0x38800000u)<=0x0effe000u)|(UInt4)(magnitude==0);
    if (valid[0]&valid[1]&valid[2]&valid[3]) {
    UInt4 rounded=((magnitude+0xfffu+((magnitude>>13)&1u)-0x38000000u)>>13)
        |((bits.u>>16)&0x8000u);
    rounded=(rounded&(UInt4)(magnitude!=0))|((bits.u>>16)&0x8000u);
        return __builtin_convertvector(rounded,HalfBits4);
    }
    /* Valid shifts are 14..24; clamp inactive lanes before shifting. */
    UInt4 exponent=magnitude>>23;
    UInt4 small_mask=(UInt4)((exponent-102u)<=10u);
    UInt4 shift=((126u-exponent)&small_mask)|(14u&~small_mask);
    UInt4 mantissa=(magnitude&0x7fffffu)|0x800000u;
    UInt4 q=mantissa>>shift,remainder=mantissa&(((UInt4)1u<<shift)-1u);
    UInt4 halfway=(UInt4)1u<<(shift-1u);
    q=q+(((UInt4)(remainder>halfway)|((UInt4)(remainder==halfway)&(0u-(q&1u))))&1u);
    UInt4 normal_mask=(UInt4)(magnitude>=0x38800000u)&(UInt4)(magnitude<0x47800000u);
    UInt4 normal=(magnitude+0xfffu+((magnitude>>13)&1u)-0x38000000u)>>13;
    UInt4 overflow_mask=(UInt4)(magnitude>=0x47800000u);
    UInt4 nan_mask=(UInt4)(magnitude>0x7f800000u);
    UInt4 result=(normal&normal_mask)|(q&small_mask)|(0x7c00u&overflow_mask)
        |(((magnitude>>13)|0x200u)&0x3ffu&nan_mask)|((bits.u>>16)&0x8000u);
    return __builtin_convertvector(result,HalfBits4);
}
typedef float Float16 __attribute__((ext_vector_type(16)));
typedef unsigned UInt16 __attribute__((ext_vector_type(16)));
typedef unsigned short HalfBits16 __attribute__((ext_vector_type(16)));
static inline Float16 decode_bits16(HalfBits16 raw) {
    UInt16 h=__builtin_convertvector(raw,UInt16),two=h<<17;
    union { UInt16 u;Float16 f; } normal,small,result;
    normal.u=(two>>4)+0x70000000u;
    normal.f=normal.f*0x1.0p-112f;
    small.u=(two>>17)|0x3f000000u;
    small.f=small.f-0.5f;
    UInt16 mask=(UInt16)(two<0x08000000u);
    result.u=((h&0x8000u)<<16)|(small.u&mask)|(normal.u&~mask);
    return result.f;
}
void controlled_act(unsigned p) {
    if (!p) return;
    MemRef *m = (MemRef*)p;
    for (unsigned k = 0; k < 3; ++k)
        if (!m[k].lo || m[k].hi || !m[k].isStatic || m[k].dtype != 2 ||
            !m[k].dims || !m[k].strides || !m[k].rank || m[k].rank > 4) return;
    unsigned *dims = (unsigned*)m[0].dims;
    unsigned width_axis=0;
    while (width_axis+1<m[0].rank && dims[width_axis]==1) ++width_axis;
    if (dims[width_axis] != 576 || ((U64*)m[0].strides)[width_axis] != 16) return;
    unsigned n = 1;
    for (unsigned i = 0; i < m[0].rank; ++i) {
        if (!dims[i] || n > (512u*576u)/dims[i]) return;
        n *= dims[i];
    }
    const unsigned short *x = (const unsigned short*)m[0].lo;
    const unsigned short *gamma = (const unsigned short*)m[1].lo;
    unsigned short *out = (unsigned short*)m[2].lo;
    unsigned base=0;
    for (; n>=32u*576u && base+16u*576u<=n; base+=16u*576u) {
        Float16 total=(Float16){0},correction=(Float16){0};
        for (unsigned j=0;j<576;++j) {
            HalfBits16 raw=(HalfBits16){x[base+0+j],x[base+576+j],x[base+1152+j],x[base+1728+j],x[base+2304+j],x[base+2880+j],x[base+3456+j],x[base+4032+j],x[base+4608+j],x[base+5184+j],x[base+5760+j],x[base+6336+j],x[base+6912+j],x[base+7488+j],x[base+8064+j],x[base+8640+j]};
            Float16 value=decode_bits16(raw),square=value*value,next=total+square;
            union { UInt16 u;Float16 f; } total_bits,square_bits;
            total_bits.f=total;square_bits.f=square;
            // Nonnegative IEEE values sort by bits; unordered NaN still chooses false.
            UInt16 choose=(UInt16)(total_bits.u>=square_bits.u)
                &(UInt16)(total_bits.u<=0x7f800000u)
                &(UInt16)(square_bits.u<=0x7f800000u);
            union { UInt16 u;Float16 f; } left,right,error;
            left.f=(total-next)+square;right.f=(square-next)+total;
            error.u=(left.u&choose)|(right.u&~choose);
            correction=correction+error.f;total=next;
        }
        for (unsigned lane=0;lane<16;++lane) {
            float variance=(total[lane]+correction[lane])/576.0f+1e-5f;
            float inverse=1.0f/sqrt32(variance);
            for (unsigned j=0;j<576;j+=4) {
                unsigned i=base+lane*576+j;
                HalfBits4 rounded=encode4(decode4(x+i)*inverse);
                *(HalfBits4*)(out+i)=encode4(decode_bits4(rounded)*decode4(gamma+i));
            }
        }
    }
    for (; base < n; base += 576) {
        float total = 0.0f, correction = 0.0f;
        for (unsigned j=0;j<576;j+=4) {
            Float4 values=decode4(x+base+j),squares=values*values;
            {
                float square=squares[0],next=total+square;
                float error=total>=square ? (total-next)+square : (square-next)+total;
                correction=correction+error;total=next;
            }
            {
                float square=squares[1],next=total+square;
                float error=total>=square ? (total-next)+square : (square-next)+total;
                correction=correction+error;total=next;
            }
            {
                float square=squares[2],next=total+square;
                float error=total>=square ? (total-next)+square : (square-next)+total;
                correction=correction+error;total=next;
            }
            {
                float square=squares[3],next=total+square;
                float error=total>=square ? (total-next)+square : (square-next)+total;
                correction=correction+error;total=next;
            }
        }
        float variance = (total+correction)/576.0f+1e-5f;
        float inverse = 1.0f/sqrt32(variance);
        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;
            HalfBits4 rounded=encode4(decode4(x+i)*inverse);
            *(HalfBits4*)(out+i)=encode4(decode_bits4(rounded)*decode4(gamma+i));
        }
    }
}
