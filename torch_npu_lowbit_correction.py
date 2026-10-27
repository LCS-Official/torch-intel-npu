"""CPU preparation of immutable low-rank residuals for frozen W8A16 inference.

Quantization-error low-rank reconstruction follows the LQER idea, with a local
ridge-regularized activation covariance objective; no upstream code is copied.
https://proceedings.mlr.press/v235/zhang24j.html
"""
from dataclasses import dataclass
from math import prod
import numpy as np
import torch
from torch_npu_lowbit import PackedWeight
from npunlock.npu4000 import MAX_TENSOR_ELEMENTS


@dataclass(frozen=True)
class LowRankCorrection:
    shape: tuple
    rank: int
    down: bytes
    up: bytes

    def __post_init__(self):
        if (type(self.shape) is not tuple or len(self.shape)!=2 or any(type(n) is not int or n<=0 for n in self.shape)
            or prod(self.shape)>MAX_TENSOR_ELEMENTS or type(self.rank) is not int or not 1<=self.rank<=min(self.shape)):
            raise ValueError('Correction requires bounded (N,K) and rank in 1..min(N,K)')
        for data,count in ((self.down,self.rank*self.shape[1]),(self.up,self.shape[0]*self.rank)):
            if type(data) is not bytes or len(data)!=count*2 or not np.isfinite(np.frombuffer(data,dtype=np.float16)).all():
                raise ValueError('Correction factors require finite immutable FP16 data of the declared shape')

    def tensors(self,rank=None):
        rank=self.rank if rank is None else rank
        if type(rank) is not int or not 1<=rank<=self.rank:raise ValueError('Invalid correction rank')
        a=torch.from_numpy(np.frombuffer(self.down,dtype=np.float16).reshape(self.rank,self.shape[1]).copy())
        b=torch.from_numpy(np.frombuffer(self.up,dtype=np.float16).reshape(self.shape[0],self.rank).copy())
        return a[:rank].contiguous(),b[:,:rank].contiguous()

    def truncated(self,rank):
        a,b=self.tensors(rank)
        return LowRankCorrection(self.shape,rank,a.numpy().tobytes(),b.numpy().tobytes())


@torch.no_grad()
def fit_lowrank_correction(weight,packed,inputs,*,rank=32):
    """Fit E ~= B A for E=W-W8 using ||E L||_F, LL'=X'X/m+ridge.

    Fixed 1% ridge limits corrections in poorly observed input directions.
    CPU FP64 SVD/solve; balanced factors are stored in FP16. This is preparation,
    not autograd/LoRA training, and does not certify whole-model quality.
    """
    if (not isinstance(packed,PackedWeight) or packed.format!='int8' or packed.group_size!=packed.shape[1]
        or not isinstance(weight,torch.Tensor) or weight.device.type!='cpu' or tuple(weight.shape)!=packed.shape
        or weight.dtype not in (torch.float16,torch.float32) or not torch.isfinite(weight).all()
        or not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or inputs.shape[1]!=packed.shape[1] or not 1<=inputs.shape[0]<=4096
        or max(packed.shape)>4096 or not torch.isfinite(inputs).all()
        or type(rank) is not int or not 1<=rank<=min(packed.shape)):
        raise ValueError('Correction fitting requires finite matching CPU W8A16 weights/activations, rows/K<=4096 and a valid rank')
    codes=torch.from_numpy(np.frombuffer(packed.data,dtype=np.int8).reshape(packed.shape).astype(np.float64))
    scales=torch.from_numpy(np.frombuffer(packed.scales,dtype=np.float16).astype(np.float64))
    error=weight.detach().double()-codes*scales[:,None]
    x=inputs.detach().double()
    h=x.t()@x/len(x)
    h.diagonal().add_(.01*h.diagonal().mean().clamp_min(1e-20))
    lower=torch.linalg.cholesky(h)
    u,s,vh=torch.linalg.svd(error@lower,full_matrices=False)
    root=s[:rank].sqrt()
    up=(u[:,:rank]*root).half().contiguous()
    down=(torch.linalg.solve_triangular(lower.t(),vh[:rank].t(),upper=True).t()*root[:,None]).half().contiguous()
    return LowRankCorrection(packed.shape,rank,down.numpy().tobytes(),up.numpy().tobytes())
