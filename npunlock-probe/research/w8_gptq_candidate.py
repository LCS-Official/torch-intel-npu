"""CPU W8 GPTQ adaptation of IST-DASLab/gptq (Apache-2.0).

Algorithm reference: https://github.com/IST-DASLab/gptq/blob/main/gptq.py
Paper: https://arxiv.org/abs/2210.17323
Changes: CPU FP64 solver, symmetric -127..127, stored FP16 per-row scale,
stable activation order, original dead-column weights, no groups/CUDA,
and row-wise fallback if actual calibration reconstruction gets worse.
The upstream license and pinned reference are in results/w8-calibration/references.
"""
import numpy as np
import torch
from torch_npu_lowbit import PackedWeight,quantize_weight


def factor_inputs(inputs):
    if (not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or not 1<=inputs.shape[0]<=4096 or not torch.isfinite(inputs).all()):
        raise ValueError('GPTQ requires 1..4096 finite CPU activation rows')
    value=inputs.detach().double();h=value.t()@value/len(value)
    energy=h.diagonal().clone();order=torch.argsort(energy,descending=True,stable=True)
    h=h[order][:,order];dead=energy[order]==0
    h.diagonal()[dead]=1
    h.diagonal().add_(.01*h.diagonal().mean().clamp_min(1e-20))
    upper=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True)
    return upper,order


def fit_gptq(weight,inputs,factor):
    original=quantize_weight(weight,format='int8');upper,order=factor
    if inputs.shape[1]!=weight.shape[1] or upper.shape!=(weight.shape[1],)*2:
        raise ValueError('GPTQ activation/factor width differs')
    values=weight.detach().double()[:,order].clone();scales=torch.from_numpy(np.frombuffer(original.scales,dtype=np.float16).astype(np.float64))
    codes=torch.empty_like(values,dtype=torch.int8);width=weight.shape[1]
    with torch.no_grad():
        for start in range(0,width,128):
            end=min(start+128,width);block=values[:,start:end].clone();residual=torch.zeros_like(block)
            for i in range(end-start):
                column=block[:,i].clone();code=(column/scales).round().clamp(-127,127)
                codes[:,start+i]=code.to(torch.int8)
                error=(column-code*scales)/upper[start+i,start+i]
                block[:,i:].sub_(error[:,None]*upper[start+i,start+i:end][None])
                residual[:,i]=error
            values[:,end:].sub_(residual@upper[start:end,end:])
    restored=codes[:,torch.argsort(order)].contiguous().numpy()
    proposed=PackedWeight(original.shape,'int8',original.group_size,restored.tobytes(),original.scales)
    activation=inputs.detach().double();target=activation@weight.detach().double().t()
    def reconstruction(q):
        code=torch.from_numpy(np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64))
        prediction=(activation@code.t())*scales[None]
        return (prediction-target).square().sum(0).numpy()
    before=reconstruction(original);loss=reconstruction(proposed);improved=loss<before
    selected=np.frombuffer(original.data,dtype=np.int8).reshape(original.shape).copy();selected[improved]=restored[improved]
    result=PackedWeight(original.shape,'int8',original.group_size,selected.tobytes(),original.scales)
    after=reconstruction(result);assert np.all(after<=before+1e-18)
    return result,dict(rows=weight.shape[0],activation_rows=len(inputs),original_output_squared_error=float(before.sum()),
        calibrated_output_squared_error=float(after.sum()),row_selection_counts=dict(original=int((~improved).sum()),gptq=int(improved.sum())),
        original_codes_sha256=__import__('hashlib').sha256(original.data).hexdigest(),
        calibrated_codes_sha256=__import__('hashlib').sha256(result.data).hexdigest())
