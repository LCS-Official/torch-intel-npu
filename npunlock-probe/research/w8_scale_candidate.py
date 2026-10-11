"""Experimental per-row INT8 least-squares scale; production is unchanged.

The fixed-code optimum is sum(weight*code)/sum(code*code), also used in
llama.cpp's make_qx_quants. Here the candidate must improve error after
FP16 scale serialization and integer recoding; no model calibration.
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
"""
import numpy as np
from torch_npu_lowbit import PackedWeight,quantize_weight


def refine_weight(weight,*,format='int8'):
    if format!='int8':raise ValueError('Candidate only supports per-row INT8')
    original=quantize_weight(weight,format=format)
    values=weight.detach().double().numpy()
    codes=np.frombuffer(original.data,dtype=np.int8).reshape(original.shape).copy()
    scales=np.frombuffer(original.scales,dtype=np.float16).copy()
    error=np.square(values-codes*scales.astype(np.float64)[:,None]).sum(-1)
    for _ in range(4):
        q=codes.astype(np.float64);denominator=np.square(q).sum(-1)
        proposed=np.divide((values*q).sum(-1),denominator,out=scales.astype(np.float64),where=denominator>0)
        proposed=np.clip(proposed,2**-24,np.finfo(np.float16).max).astype(np.float16)
        next_codes=np.clip(np.rint(values/proposed.astype(np.float64)[:,None]),-127,127).astype(np.int8)
        next_error=np.square(values-next_codes*proposed.astype(np.float64)[:,None]).sum(-1)
        improved=next_error<error
        if not improved.any():break
        codes[improved]=next_codes[improved];scales[improved]=proposed[improved];error[improved]=next_error[improved]
    return PackedWeight(original.shape,format,original.group_size,codes.tobytes(),scales.tobytes())
