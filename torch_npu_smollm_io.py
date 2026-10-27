"""Explicit FP16 CPU endpoints for SmolLM NPU inference, without a CPU decoder.

Loads the tied embedding/output matrix and final RMSNorm only. No Transformers
or OpenVINO import; tokenization and all decoder execution remain explicit.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


class FrozenSmolIO(torch.nn.Module):
    """Inference-only CPU embedding and norm/output head for the frozen adapter."""
    def __init__(self,config,embedding,norm):
        super().__init__()
        if (config.get('model_type')!='llama' or config.get('hidden_size')!=576
            or config.get('tie_word_embeddings') is not True or config.get('rms_norm_eps')!=1e-5
            or type(config.get('vocab_size')) is not int or config['vocab_size']<=0):
            raise ValueError('Requires tied SmolLM CPU endpoints with D576 and RMSNorm eps=1e-5')
        for name,value,shape in (('embedding',embedding,(config['vocab_size'],576)),('norm',norm,(576,))):
            if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or tuple(value.shape)!=shape
                or value.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(value).all()):
                raise ValueError('Requires finite CPU floating endpoint weights of the expected shape')
            half=value.detach().to(dtype=torch.float16,copy=True,memory_format=torch.contiguous_format)
            if not torch.isfinite(half).all():raise ValueError('Endpoint weight overflows FP16')
            self.register_buffer(name,half)
        self.eval()

    @classmethod
    def from_checkpoint(cls,directory):
        directory=Path(directory);config=json.loads((directory/'config.json').read_text())
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:
            if 'lm_head.weight' in file.keys():
                raise ValueError('Requires the canonical tied checkpoint without a separate lm_head weight')
            return cls(config,file.get_tensor('model.embed_tokens.weight'),file.get_tensor('model.norm.weight'))

    def _cpu_eval(self):
        if self.training or self.embedding.device.type!='cpu' or self.norm.device.type!='cpu':
            raise ValueError('FrozenSmolIO requires eval mode and CPU endpoint buffers')
        if self.embedding.dtype!=torch.float16 or self.norm.dtype!=torch.float16:
            raise ValueError('FrozenSmolIO endpoints require FP16')

    def embed_tokens(self,ids):
        self._cpu_eval()
        if (not isinstance(ids,torch.Tensor) or ids.device.type!='cpu' or ids.dtype!=torch.int64
            or ids.ndim!=2 or any(size<=0 for size in ids.shape)):
            raise ValueError('Embedding requires nonempty CPU int64 (B,S) token IDs')
        return F.embedding(ids,self.embedding)

    def logits(self,hidden):
        self._cpu_eval()
        if (not isinstance(hidden,torch.Tensor) or hidden.device.type!='cpu' or hidden.dtype!=torch.float16
            or hidden.ndim!=3 or hidden.shape[-1]!=576 or any(size<=0 for size in hidden.shape)):
            raise ValueError('Output head requires nonempty CPU FP16 (B,S,576) hidden states')
        # FP32 variance/inverse, then FP16 scaling: preserve the checkpoint's RMSNorm rounding.
        value=hidden.float()
        inverse=torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)
        normalized=(value*inverse).half()*self.norm
        return F.linear(normalized,self.embedding)
