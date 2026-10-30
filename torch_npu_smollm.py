"""Frozen SmolLM2-135M layers: FP16/W8/INT4/NF4, causal, no CPU forward fallback.

CPU weight loading and RoPE constants are explicit preparation. This imports
neither Transformers nor OpenVINO; Transformers is only the demo reference.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from math import prod
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
import npunlock as s
from safetensors import safe_open
from npunlock.npu4000 import MAX_TENSOR_ELEMENTS, compile_rmsnorm, compile_fused_rmsnorm
from torch_npu_lowbit import PackedWeight, quantize_weight, weight_graph
from torch_npu_lowbit_correction import LowRankCorrection
from torch_npu_transformer import FrozenTransformer, _half_bytes
from torch_npu_graph_cache import compile_fused_cached, prefill_cache_graph


_PROJECTIONS=frozenset(('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
    'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight'))
_WEIGHT_FORMATS={'fp16':None,'w8a16':'int8','w4a16':'int4','nf4a16':'nf4'}


def _projection_policy(mode,names):
    if (type(names) is not tuple or any(type(name) is not str for name in names)
        or len(set(names))!=len(names) or not set(names)<=_PROJECTIONS):
        raise ValueError('fp16_projections requires a tuple of distinct SmolLM projection weight names')
    if names and (mode!='w8a16' or len(names)==len(_PROJECTIONS)):
        raise ValueError('FP16 protection requires w8a16 with at least one INT8 projection; use mode=fp16 for all FP16')
    return frozenset(names)


@lru_cache(maxsize=1)
def _rmsnorm_kernel():
    root=Path(__file__).resolve().parent/'npunlock/tests/fixtures/npu4000'
    manifest=json.loads((root/'rmsnorm/manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('RMSNorm source changed; run build_rmsnorm.py')
    kernel=(root/'rmsnorm/rmsnorm-f16.elf').read_bytes()
    if hashlib.sha256(kernel).hexdigest()!=manifest['elf_sha256']:
        raise RuntimeError('RMSNorm kernel differs from its build manifest')
    return kernel


@lru_cache(maxsize=1)
def _kv_select_kernel():
    root=Path(__file__).resolve().parent/'npunlock/tests/fixtures/npu4000'
    manifest=json.loads((root/'kv_select/manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('KV Select source changed; run build_kv_select.py')
    kernel=(root/'kv_select/kv-select-f16.elf').read_bytes()
    if hashlib.sha256(kernel).hexdigest()!=manifest['elf_sha256']:
        raise RuntimeError('KV Select kernel differs from its build manifest')
    return kernel


class FrozenSmolLayer(torch.nn.Module):
    """Explicit single SmolLM2 decoder layer, bounded FP16 NPU input/output."""
    # Reuse the existing frozen-graph execution/cache and profile inspection.
    _run_graph=FrozenTransformer._run_graph
    _compile_graph=FrozenTransformer._compile_graph
    _inspect_blob=FrozenTransformer._inspect_blob
    mode=FrozenTransformer.mode

    def __init__(self,config,weights,*,mode='w8a16',packed_weights=None,fp16_projections=(),corrections=None):
        super().__init__()
        if mode not in _WEIGHT_FORMATS:raise ValueError('Requires fp16, w8a16, w4a16 or nf4a16')
        weight_format=_WEIGHT_FORMATS[mode]
        self._fp16_projections=_projection_policy(mode,fp16_projections)
        required=dict(model_type='llama',hidden_size=576,intermediate_size=1536,num_attention_heads=9,
                      num_key_value_heads=3,hidden_act='silu',attention_bias=False,mlp_bias=False,
                      rms_norm_eps=1e-5,rope_theta=100000,rope_scaling=None,pretraining_tp=1)
        if any(config.get(k)!=v for k,v in required.items()) or config.get('rope_interleaved',False):
            raise ValueError('Requires the verified SmolLM2-135M configuration')
        self._d=576;self._mode=mode;self._keys={};self._evidence={};self._weights={};self._packed={};self._norm_parameters={}
        self._cache_owner=object();self._norm_workspaces={};self._fuse_norm=True;self._whole_layer=True;self._fuse_prefill=True
        shapes={'self_attn.q_proj.weight':(576,576),'self_attn.k_proj.weight':(192,576),
                'self_attn.v_proj.weight':(192,576),'self_attn.o_proj.weight':(576,576),
                'mlp.gate_proj.weight':(1536,576),'mlp.up_proj.weight':(1536,576),'mlp.down_proj.weight':(576,1536),
                'input_layernorm.weight':(576,),'post_attention_layernorm.weight':(576,)}
        if corrections is not None and (type(corrections) is not dict or mode!='w8a16' or packed_weights is None
            or not set(corrections)<=_PROJECTIONS or set(corrections)&self.fp16_projections
            or any(not isinstance(c,LowRankCorrection) or c.shape!=shapes[name] or c.rank>32 for name,c in corrections.items())):
            raise ValueError('Corrections require matching rank<=32 factors, explicit packed W8 weights, and unprotected projections')
        self._corrections={} if corrections is None else dict(corrections)
        if set(weights)!=set(shapes):raise ValueError('Unexpected layer weight names')
        if packed_weights is not None:
            if weight_format is None or type(packed_weights) is not dict or set(packed_weights)!={name for name,shape in shapes.items() if len(shape)==2}:
                raise ValueError('Prepared weights require all seven projections in a matching lowbit mode')
            for name,q in packed_weights.items():
                if (not isinstance(q,PackedWeight) or q.shape!=shapes[name] or q.format!=weight_format
                    or (weight_format=='int8' and q.group_size!=shapes[name][1])):
                    raise ValueError('Prepared weights require matching formats/shapes; W8 uses per-row scales')
        for name,shape in shapes.items():
            w=weights[name]
            if not isinstance(w,torch.Tensor) or w.device.type!='cpu' or w.dtype not in (torch.float16,torch.bfloat16,torch.float32):
                raise ValueError('Preparation requires CPU FP16/BF16/FP32 weights')
            self._weights[name]=(_half_bytes(w.float(),shape),shape)
            if weight_format is not None and len(shape)==2:
                self._packed[name]=(quantize_weight(w.float().half(),format=weight_format,
                    group_size=None if weight_format=='int8' else 32) if packed_weights is None else packed_weights[name])
        # ponytail: fixed 64-position RoPE/cache; extend only with new bounded
        # attention, allocation and numerical checks. No arbitrary masks.
        frequency=1./(100000**(torch.arange(0,64,2).float()/64))
        angles=torch.outer(torch.arange(64).float(),frequency)
        angles=torch.cat((angles,angles),dim=-1)
        self._cos=angles.cos().half().numpy().tobytes();self._sin=angles.sin().half().numpy().tobytes()
        self.eval()

    @classmethod
    def from_checkpoint(cls,directory,*,layer_index=0,mode='w8a16',packed_weights=None,fp16_projections=(),corrections=None):
        _projection_policy(mode,fp16_projections)
        directory=Path(directory);config=json.loads((directory/'config.json').read_text())
        if type(layer_index) is not int or not 0<=layer_index<config.get('num_hidden_layers',0):raise ValueError('Invalid layer index')
        prefix=f'model.layers.{layer_index}.'
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:
            weights={name[len(prefix):]:file.get_tensor(name) for name in file.keys() if name.startswith(prefix)}
        return cls(config,weights,mode=mode,packed_weights=packed_weights,fp16_projections=fp16_projections,corrections=corrections)

    @property
    def fp16_projections(self):return self._fp16_projections

    def _shape(self,shape):
        if (len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape) or shape[2]!=576
            or shape[0]>8 or shape[1]>64 or max(prod(shape),shape[0]*shape[1]*1536,shape[0]*9*shape[1]**2)>MAX_TENSOR_ELEMENTS):
            raise ValueError('Requires bounded FP16 (B,S,576), B<=8, S<=64')
        return tuple(shape)

    def _cpu_weight(self,name):
        data,shape=self._weights[name]
        return torch.from_numpy(np.frombuffer(data,dtype=np.float16).reshape(shape).copy())

    def reference_cpu(self,x,*,quantized=True,segment=None,normalized=None):
        if not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype not in (torch.float16,torch.float32) or not torch.isfinite(x).all():
            raise ValueError('Reference requires finite CPU FP16/FP32 input')
        b,length,d=self._shape(tuple(x.shape));x=x.detach().half();trace={}
        if not torch.isfinite(x).all():raise ValueError('Reference input overflows FP16')
        if segment not in (None,'attention','ffn') or (segment is not None and (not isinstance(normalized,torch.Tensor)
            or normalized.device.type!='cpu' or normalized.dtype!=torch.float16 or normalized.shape!=x.shape)):
            raise ValueError('CPU segment requires matching FP16 normalization')
        def linear(v,name):
            if quantized and self.mode!='fp16' and name not in self.fp16_projections:
                q=self._packed[name]
                if q.format=='int8':
                    codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape)
                    scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)
                    weight=torch.from_numpy(codes.astype(np.float64)*scales[:,None])
                    value=(v.double()@weight.t()).half()
                else:
                    value=F.linear(v.float(),torch.from_numpy(q.dequantize().copy()).float()).half()
                if name in self._corrections:
                    a,b=self._corrections[name].tensors()
                    delta=F.linear(F.linear(v.float(),a.float()).half().float(),b.float()).half()
                    value=(value+delta).half()
                return value
            return F.linear(v.float(),self._cpu_weight(name).float()).half()
        def norm(v,name):
            value=v.float();normalized=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)).half()
            return (normalized*self._cpu_weight(name)).half()
        supplied=normalized
        normalized=supplied if segment=='attention' else norm(x,'input_layernorm.weight');trace['norm1']=normalized
        q=linear(normalized,'self_attn.q_proj.weight').view(b,length,9,64).transpose(1,2)
        k=linear(normalized,'self_attn.k_proj.weight').view(b,length,3,64).transpose(1,2)
        v=linear(normalized,'self_attn.v_proj.weight').view(b,length,3,64).transpose(1,2)
        cos=torch.from_numpy(np.frombuffer(self._cos,dtype=np.float16).reshape(64,64)[:length].copy())[None,None]
        sin=torch.from_numpy(np.frombuffer(self._sin,dtype=np.float16).reshape(64,64)[:length].copy())[None,None]
        def rotate(value):return (value*cos+torch.cat((-value[...,32:],value[...,:32]),-1)*sin).half()
        q=rotate(q);k=rotate(k);trace['keys']=k;trace['values']=v
        k=k.repeat_interleave(3,dim=1);v=v.repeat_interleave(3,dim=1)
        scores=(q.float()@k.float().transpose(-1,-2)).half();scores=(scores*.125).half()
        mask=torch.triu(torch.full((length,length),float('-inf')),diagonal=1)
        probs=(scores+mask).float().softmax(-1).half()
        attended=(probs.float()@v.float()).half().transpose(1,2).reshape(b,length,d);trace['attention']=attended
        delta=linear(attended,'self_attn.o_proj.weight');trace['attention_output']=delta
        value=(x+delta).half();trace['residual1']=value
        if segment=='attention':return value,trace
        if segment=='ffn':value=x
        normalized=supplied if segment=='ffn' else norm(value,'post_attention_layernorm.weight');trace['norm2']=normalized
        gate=linear(normalized,'mlp.gate_proj.weight');up=linear(normalized,'mlp.up_proj.weight')
        activated=F.silu(gate.float()).half();trace['silu']=activated
        hidden=(activated*up).half();trace['gated']=hidden
        delta=linear(hidden,'mlp.down_proj.weight');trace['ffn_output']=delta
        value=(value+delta).half();trace['output']=value
        return value,trace

    def graph(self,shape,*,segment=None,block_index=0,stage=None,kv_cache=None,fused_norm=False):
        if segment not in (None,'attention','ffn','layer') or block_index!=0:raise ValueError('Single frozen SmolLM layer only')
        if kv_cache not in (None,'prefill','decode') or (kv_cache is not None and (segment not in ('attention','layer') or stage is not None)):
            raise ValueError('KV graph requires the attention segment without a diagnostic stage')
        batch,length,d=self._shape(shape);b=batch;rows=b*length;decodes=[];trace={}
        if segment=='layer' and (not fused_norm or stage is not None):raise ValueError('Whole-layer graph requires fused normalization')
        if kv_cache=='decode' and length!=1:raise ValueError('Decode graph requires one token')
        inp=s.input('x',shape=shape,dtype='f16')
        inputs=[inp]
        if fused_norm and segment is None:raise ValueError('Fused norm requires an attention, FFN or whole-layer segment')
        if segment is not None and not fused_norm:
            normalized_input=s.input('normalized',shape=shape,dtype='f16');inputs.append(normalized_input)
        def op(name,*values,shape=None,dtype='f16',**attrs):return s.op(name,*values,_shape=shape or values[0].shape,_dtype=dtype,**attrs)
        def reshape(v,shape):return op('Reshape',v,s.constant(np.array(shape,dtype=np.int64)),shape=shape,dtype=v.dtype,special_zero=False)
        def slice_(v,axis,start,end):
            begin=[0]*len(v.shape);stop=list(v.shape);begin[axis]=start;stop[axis]=end
            shape=list(v.shape);shape[axis]=end-start;mask=[0]*len(shape)
            return op('StridedSlice',v,*(s.constant(np.array(a,dtype=np.int64)) for a in (begin,stop,[1]*len(shape))),shape=tuple(shape),
                begin_mask=mask,end_mask=mask,new_axis_mask=mask,shrink_axis_mask=mask,ellipsis_mask=mask)
        def fused(v,gamma,name):
            norm_input=reshape(v,(rows,576))
            # A second internal row prevents width tiling of the B1 carrier.
            # Slice before projection: public batch, attention and KV stay B1.
            if rows==1:norm_input=op('Concat',norm_input,norm_input,shape=(2,576),axis=0)
            result=op('Power',norm_input,gamma,shape=norm_input.shape,auto_broadcast='numpy',_name=name)
            if rows==1:result=slice_(result,0,0,1)
            return reshape(result,shape)
        if fused_norm:
            gamma=s.input('gamma',shape=(max(2,rows),576),dtype='f16');inputs.append(gamma)
            normalized_input=fused(inp,gamma,'rmsnorm_carrier')
            if segment=='layer':
                gamma_ffn=s.input('ffn_gamma',shape=gamma.shape,dtype='f16');inputs.append(gamma_ffn)
        # Keep N=1 before projection/head formation as well as dynamic MatMul;
        # retaining an intermediate N=batch triggers ConvertBatchedLayerTo1N.
        folded=segment in ('attention','layer') and length==1 and batch>1
        if folded:
            inp=reshape(inp,(1,batch,d));normalized_input=reshape(normalized_input,(1,batch,d))
            shape=(1,batch,d)
        def norm(v,name):
            variance=op('ReduceMean',op('Multiply',v,v,auto_broadcast='numpy'),s.constant(np.array([-1],dtype=np.int64)),shape=(b,length,1),keep_dims=True)
            inverse=op('Power',op('Add',variance,s.constant(np.float16(1e-5)),auto_broadcast='numpy'),s.constant(np.float16(-.5)),auto_broadcast='numpy')
            return op('Multiply',op('Multiply',v,inverse,shape=shape,auto_broadcast='numpy'),s.constant(self._cpu_weight(name).numpy()),auto_broadcast='numpy')
        def linear(v,name):
            v=reshape(v,(rows,v.shape[-1]));data,wshape=self._weights[name]
            if self.mode=='fp16' or name in self.fp16_projections:
                attrs={'name':name.replace('.','_')+'_protected_fp16'} if name in self.fp16_projections else {}
                value=op('MatMul',v,s.constant(np.frombuffer(data,dtype=np.float16).reshape(wshape),**attrs),shape=(rows,wshape[0]),transpose_a=False,transpose_b=True)
            else:
                # Driver 8.3 loses NF4 low-float metadata in the single-row
                # weight-compression route. Pad only the projection, then trim;
                # the public token, attention and KV shapes remain unchanged.
                graph,xml,_=weight_graph(self._packed[name],rows,pad_batch=self.mode=='nf4a16' and rows==1)
                mapped={graph.inputs[0]:v};prefix=name.replace('.','_')+'_'
                for node in graph.nodes:
                    if node.op=='Parameter':continue
                    cloned=replace(node,inputs=tuple(mapped[t] for t in node.inputs),name=prefix+node.name if node.name else None)
                    mapped.update(zip(node.outputs,cloned.outputs))
                decodes.extend(prefix+l.attrib['name'] for l in ET.fromstring(xml).findall('./layers/layer') if l.find("./rt_info/attribute[@name='decompression']") is not None)
                value=mapped[graph.outputs[0]]
                if name in self._corrections:
                    down_weight,up_weight=self._corrections[name].tensors()
                    reduced=op('MatMul',v,s.constant(down_weight.numpy(),name=prefix+'correction_down'),shape=(rows,down_weight.shape[0]),transpose_a=False,transpose_b=True)
                    delta=op('MatMul',reduced,s.constant(up_weight.numpy(),name=prefix+'correction_up'),shape=(rows,wshape[0]),transpose_a=False,transpose_b=True)
                    value=op('Add',value,delta,shape=(rows,wshape[0]),auto_broadcast='numpy')
            return reshape(value,(1,batch,wshape[0]) if folded else (b,length,wshape[0]))
        def heads(v,count):
            return reshape(v,(1,batch*count,1,64)) if folded else op('Transpose',reshape(v,(b,length,count,64)),s.constant(np.array([0,2,1,3],dtype=np.int64)),shape=(b,count,length,64))
        if kv_cache=='decode':
            past_k=s.input('past_k',shape=(batch,3,64,64),dtype='f16')
            past_v=s.input('past_v',shape=(batch,3,64,64),dtype='f16')
            position=s.input('position',shape=(1,),dtype='f32');inputs.extend([past_k,past_v,position])
            if folded:past_k=reshape(past_k,(1,batch*3,64,64));past_v=reshape(past_v,(1,batch*3,64,64))
            indices=op('Convert',position,dtype='i64',destination_type='i64')
            # Share the position lookup for RoPE, mask and exact KV flags.
            mask_table=np.where(np.arange(64)[None,:]>np.arange(64)[:,None],-np.inf,0).astype(np.float16)
            table=s.constant(np.concatenate((np.frombuffer(self._cos,dtype=np.float16).reshape(64,64),
                np.frombuffer(self._sin,dtype=np.float16).reshape(64,64),mask_table,np.eye(64,dtype=np.float16)),axis=1))
            positional=op('Gather',table,indices,s.constant(np.int64(0)),shape=(1,256),batch_dims=0)
            cos=reshape(slice_(positional,1,0,64),(1,1,1,64))
            sin=reshape(slice_(positional,1,64,128),(1,1,1,64))
            table_mask=reshape(slice_(positional,1,128,192),(1,1,1,64))
        else:
            cos=s.constant(np.frombuffer(self._cos,dtype=np.float16).reshape(64,64)[:length][None,None])
            sin=s.constant(np.frombuffer(self._sin,dtype=np.float16).reshape(64,64)[:length][None,None])
        def rotate(v):
            rotated=op('Concat',op('Negative',slice_(v,3,32,64)),slice_(v,3,0,32),shape=v.shape,axis=3)
            return op('Add',op('Multiply',v,cos,auto_broadcast='numpy'),op('Multiply',rotated,sin,auto_broadcast='numpy'),auto_broadcast='numpy')
        def repeat(v):
            return op('Gather',v,s.constant(np.repeat(np.arange(batch*3 if folded else 3),3).astype(np.int64)),s.constant(np.int64(1)),shape=(1,batch*9,v.shape[2],64) if folded else (b,9,v.shape[2],64),batch_dims=0)
        normalized=normalized_input if segment in ('attention','layer') else norm(inp,'input_layernorm.weight');trace['norm1']=normalized
        q=rotate(heads(linear(normalized,'self_attn.q_proj.weight'),9))
        k_cache=rotate(heads(linear(normalized,'self_attn.k_proj.weight'),3))
        v_cache=heads(linear(normalized,'self_attn.v_proj.weight'),3)
        if kv_cache=='decode':
            slots=s.constant(np.arange(64,dtype=np.float16));p=op('Convert',position,destination_type='f16')
            selected=reshape(op('Convert',slice_(positional,1,192,256),dtype='boolean',destination_type='boolean'),(1,1,64,1))
            k_cache=op('Select',selected,k_cache,past_k,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')
            v_cache=op('Select',selected,v_cache,past_v,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')
            mask=table_mask
        else:
            mask=s.constant(np.where(np.arange(length)[None,:]>np.arange(length)[:,None],-np.inf,0).astype(np.float16))
        k=repeat(k_cache);v=repeat(v_cache);keys=64 if kv_cache=='decode' else length
        def attention(q,k,v,batch):
            scores=op('MatMul',q,k,shape=(batch,q.shape[1],length,keys),transpose_a=False,transpose_b=True)
            scores=op('Multiply',scores,s.constant(np.float16(.125)),auto_broadcast='numpy')
            scores=op('Add',scores,mask,auto_broadcast='numpy');probs=op('Softmax',scores,axis=3)
            return op('MatMul',probs,v,shape=(batch,q.shape[1],length,64),transpose_a=False,transpose_b=False)
        attended=attention(q,k,v,1 if folded else b)
        attended=reshape(attended,shape) if folded else reshape(op('Transpose',attended,s.constant(np.array([0,2,1,3],dtype=np.int64)),shape=(b,length,9,64)),shape);trace['attention']=attended
        value=op('Add',inp,linear(attended,'self_attn.o_proj.weight'),auto_broadcast='numpy');trace['residual1']=value
        residual1=value
        if segment=='ffn':value=inp
        normalized=normalized_input if segment=='ffn' else fused(value,gamma_ffn,'rmsnorm_ffn') if segment=='layer' else norm(value,'post_attention_layernorm.weight');trace['norm2']=normalized
        gate=linear(normalized,'mlp.gate_proj.weight');up=linear(normalized,'mlp.up_proj.weight')
        activated=op('Swish',gate);trace['silu']=activated
        hidden=op('Multiply',activated,up,auto_broadcast='numpy');trace['gated']=hidden
        value=op('Add',value,linear(hidden,'mlp.down_proj.weight'),auto_broadcast='numpy');trace['output']=value
        if segment=='attention':value=residual1
        value=trace[stage] if stage is not None else value
        if kv_cache is not None:
            if kv_cache=='prefill' and length<64:
                cache_shape=(1,batch*3,64,64) if folded else (b,3,64,64)
                padding=s.constant(np.zeros((*cache_shape[:2],64-length,64),dtype=np.float16))
                k_cache=op('Concat',k_cache,padding,shape=cache_shape,axis=2)
                v_cache=op('Concat',v_cache,padding,shape=cache_shape,axis=2)
        if folded:
            value=reshape(value,(batch,1,d))
            if kv_cache is not None:
                k_cache=reshape(k_cache,(batch,3,64,64));v_cache=reshape(v_cache,(batch,3,64,64))
        values=[value,k_cache,v_cache] if kv_cache is not None else [value]
        if kv_cache=='decode':
            # The bounded integer positions 1..64 are exact in FP16. This
            # metadata increment does not change any model arithmetic dtype.
            next_position=op('Add',p,s.constant(np.float16(1)),auto_broadcast='numpy')
            values.append(op('Convert',next_position,dtype='f32',destination_type='f32'))
        outputs=[op('Reshape',v,s.constant(np.array(v.shape,dtype=np.int64)),dtype=v.dtype,special_zero=False,_name=name)
                 for v,name in zip(values,['y','cache_k','cache_v','next_position'])]
        graph=s.Graph(inputs,outputs,name='smollm_layer')
        if any(prod(t.shape)>MAX_TENSOR_ELEMENTS for node in graph.nodes for t in node.outputs):raise ValueError('Intermediate exceeds storage bound')
        ir=s.serialize_ir(graph);xml=ET.fromstring(ir.xml)
        for layer in xml.findall('./layers/layer'):
            if layer.attrib['type']=='Swish':layer.set('version','opset4')
            if layer.attrib['type']=='Gather':layer.find('data').attrib.pop('batch_dims',None)
            if layer.attrib['name'] in decodes:
                info=ET.SubElement(layer,'rt_info');ET.SubElement(info,'attribute',name='decompression',version='0')
        return graph,ET.tostring(xml),ir.weights

    def _norm_device_legacy(self,x,name,**kwargs):
        import torch_intel_npu as npu
        if name not in self._norm_parameters:
            self._norm_parameters[name]=self._cpu_weight(name).view(1,1,576).to('intel_npu')
        value=x.float();variance=(value*value).mean(-1,keepdim=True)
        normalized=(value*npu._execute('rsqrt',variance+1e-5)).half()
        return normalized*self._norm_parameters[name]

    def _norm_parameter(self,name,count):
        import torch_intel_npu as npu
        parameter=(name,count)
        if parameter not in self._norm_parameters:
            if name not in self._norm_parameters:
                self._norm_parameters[name]=self._cpu_weight(name).view(1,1,576).to('intel_npu')
            self._norm_parameters[parameter]=npu._broadcast(self._norm_parameters[name].view(1,576),(count,576))
        return self._norm_parameters[parameter]

    def _norm_device(self,x,name,*,parallel=True,reuse=False):
        """Fresh by default; internal synchronous consumers can reuse scratch."""
        import torch_intel_npu as npu
        with npu._lock:
            npu.synchronize();npu._validate(x,dense=True);shape=self._shape(tuple(x.shape))
            if x.dtype!=torch.float16 or name not in ('input_layernorm.weight','post_attention_layernorm.weight'):
                raise ValueError('Fused RMSNorm requires FP16 NPU input and a layer norm weight')
            value=npu._materialize(x);rows=shape[0]*shape[1];kernel=_rmsnorm_kernel();parts=[]
            # ponytail: keep serial <=256 because ReferenceSW splits width at
            # 512 rows. DefaultHW assigns complete row shards across four tiles.
            limit=512 if parallel else 256
            for start in range(0,rows,limit):
                count=min(limit,rows-start);key=('smollm_rmsnorm',count,hashlib.sha256(kernel).digest(),parallel)
                factory=lambda:compile_rmsnorm(count,kernel,libraries=npu._native,parallel=parallel)
                npu._register_graph_factory(key,factory)
                gamma=self._norm_parameter(name,count)
                chunk=value.view(rows,576) if rows<=limit else npu._slice(value,start*576,count*576).view(count,576)
                out=None
                if reuse and rows<=limit:
                    # At most one complete scratch allocation per norm. Public
                    # outputs/KV are fresh graph outputs; chunked norms stay fresh.
                    out=self._norm_workspaces.get(name)
                    if out is None or tuple(out.shape)!=(count,576):
                        out=torch.empty((count,576),dtype=torch.float16,device='intel_npu')
                        self._norm_workspaces[name]=out
                    npu._check_out(out,(count,576),torch.float16,[x,gamma])
                parts.append(npu._run(key,[chunk,gamma],(count,576),torch.float16,out))
            return npu._concat(parts,shape,torch.float16)

    def _use_fused_norm(self,x):
        # ponytail: bounded B1..B8, <=64 positions; complete-row compiler
        # contracts are checked before launching every new fused shape.
        return self._fuse_norm and (x.shape[1]==1 or self._fuse_prefill) and 1<=x.shape[0]<=8

    def _compile_fused(self,shape,segment,kind,prepared=None):
        import torch_intel_npu as npu
        local=(shape,'fused_rmsnorm',segment,kind);key=self._keys[local]
        kernel=_rmsnorm_kernel();selection=_kv_select_kernel() if kind=='decode' else b''
        if prepared is None:
            metadata=prefill_cache_graph(self,shape,kernel,compile_fused_rmsnorm) if key[0]=='smollm_prefill_recipe' else None
            if metadata is None:
                graph,xml,data=self.graph(shape,segment=segment,kv_cache=kind,fused_norm=True);graph_factory=None
            else:
                graph=metadata[0];xml=data=None
                graph_factory=lambda:self.graph(shape,segment=segment,kv_cache=kind,fused_norm=True)
        else:graph,xml,data,graph_factory=prepared
        program=compile_fused_cached(npu,graph,xml,data,kernel,graph_digest=key[1],whole_layer=segment=='layer',
            select_kernel_elf=selection or None,compiler_factory=compile_fused_rmsnorm,graph_factory=graph_factory)
        evidence=self._inspect_blob(program.graph_blob);evidence['rmsnorm']=json.loads(program.patch_reports[0])
        if selection:evidence['kv_select']=json.loads(program.patch_reports[1])
        self._evidence[local]=evidence
        return program

    def _run_fused(self,x,*,segment,cache=None,kv=False,plan=None):
        import torch_intel_npu as npu
        shape=tuple(x.shape);rows=shape[0]*shape[1]
        kind=('prefill' if cache is None else 'decode') if kv else None
        local=(shape,'fused_rmsnorm',segment,kind);compiled_input=None
        if local not in self._keys:
            kernel=_rmsnorm_kernel();selection=_kv_select_kernel() if kind=='decode' else b''
            prepared=prefill_cache_graph(self,shape,kernel,compile_fused_rmsnorm) if segment=='layer' and kind=='prefill' else None
            if prepared is None:
                graph,xml,data=self.graph(shape,segment=segment,kv_cache=kind,fused_norm=True)
                key=('smollm_fused_norm',hashlib.sha256(xml+data+kernel+selection).digest());factory=None
            else:
                graph,digest=prepared;xml=data=None;key=('smollm_prefill_recipe',digest)
                factory=lambda:self.graph(shape,segment=segment,kv_cache=kind,fused_norm=True)
            self._keys[local]=key;compiled_input=(graph,xml,data,factory)
        key=self._keys[local]
        factory=lambda:self._compile_fused(shape,segment,kind,compiled_input)
        npu._register_graph_factory(key,factory)
        if local not in self._evidence and key in npu._sessions:
            program=npu._sessions[key][1]
            evidence=self._inspect_blob(program.graph_blob);evidence['rmsnorm']=json.loads(program.patch_reports[0])
            if kind=='decode':evidence['kv_select']=json.loads(program.patch_reports[1])
            self._evidence[local]=evidence
        name='input_layernorm.weight' if segment=='attention' else 'post_attention_layernorm.weight'
        names=['input_layernorm.weight','post_attention_layernorm.weight'] if segment=='layer' else [name]
        inputs=[x,*[self._norm_parameter(name,max(2,rows)) for name in names]]
        specs=[(shape,torch.float16)]
        if kv:
            specs.extend([((shape[0],3,64,64),torch.float16)]*2)
        if cache is not None:
            inputs.extend([cache.keys,cache.values,cache._position]);specs.append(((1,),torch.float32))
        if plan is not None:
            entry=npu._prepare_many(self._keys[local],inputs,specs)
            plan.append(entry)
            return entry[-1]
        return npu._run_many(self._keys[local],inputs,specs) if len(specs)>1 else (npu._run(self._keys[local],inputs,shape,torch.float16),)

    def forward(self,x):
        import torch_intel_npu as npu
        with npu._lock:
            self._validate_input(x)
            value=npu._materialize(x)
            if self._use_fused_norm(value):
                if self._whole_layer:return self._run_fused(value,segment='layer')[0]
                value=self._run_fused(value,segment='attention')[0]
                return self._run_fused(value,segment='ffn')[0]
            normalized=self._norm_device(value,'input_layernorm.weight',reuse=True)
            value=self._run_graph(value,normalized,segment='attention')
            normalized=self._norm_device(value,'post_attention_layernorm.weight',reuse=True)
            return self._run_graph(value,normalized,segment='ffn')

    def _validate_input(self,x):
        import torch_intel_npu as npu
        if self.training:raise ValueError('FrozenSmolLayer requires eval; training is unsupported')
        npu.synchronize();npu._validate(x,dense=True);self._shape(tuple(x.shape))
        if x.dtype!=torch.float16:raise ValueError('FrozenSmolLayer requires FP16 NPU input')

    def _validate_cache(self,x,cache):
        import torch_intel_npu as npu
        if x.shape[1]!=1 or type(cache) is not SmolKVCache or cache._owner is not self._cache_owner:
            raise ValueError('Decode requires one token and this layer\'s own KV cache')
        if type(cache.length) is not int or not 1<=cache.length<64:
            raise ValueError('KV cache is full or has an invalid length; total tokens must be <=64')
        for tensor,shape,dtype in [(cache.keys,(x.shape[0],3,64,64),torch.float16),
                                    (cache.values,(x.shape[0],3,64,64),torch.float16),(cache._position,(1,),torch.float32)]:
            npu._validate(tensor)
            if tuple(tensor.shape)!=shape or tensor.dtype!=dtype:raise ValueError('KV cache has the wrong batch, layout or dtype')

    def _compile_kv_graph(self,shape,kind,prepared=None):
        import torch_intel_npu as npu
        graph,xml,data=self.graph(shape,segment='attention',kv_cache=kind) if prepared is None else prepared
        result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
        if (result.vendor_id,result.device_id,tuple(result.compiler_version))!=(0x8086,0x643e,(8,3)):
            raise ValueError('Unsupported SmolLM KV profile')
        program=s.load_native(result.graph_blob,graph=graph,libraries=npu._native)
        self._evidence[(shape,'kv_'+kind,0)]=self._inspect_blob(program.graph_blob)
        return program

    def _run_cached_attention(self,x,normalized,cache=None):
        import torch_intel_npu as npu
        shape=tuple(x.shape);kind='prefill' if cache is None else 'decode';local=(shape,'kv_'+kind,0);prepared=None
        if local not in self._keys:
            prepared=self.graph(shape,segment='attention',kv_cache=kind)
            self._keys[local]=('smollm_kv',hashlib.sha256(prepared[1]+prepared[2]).digest())
        key=self._keys[local];factory=lambda:self._compile_kv_graph(shape,kind,prepared)
        npu._register_graph_factory(key,factory)
        if local not in self._evidence and key in npu._sessions:
            self._evidence[local]=self._inspect_blob(npu._sessions[key][1].graph_blob)
        inputs=[x,normalized] if cache is None else [x,normalized,cache.keys,cache.values,cache._position]
        output_shape=shape
        specs=[(target,torch.float16) for target in [output_shape,(output_shape[0],3,64,64),(output_shape[0],3,64,64)]]
        if cache is not None:specs.append(((1,),torch.float32))
        return npu._run_many(self._keys[local],inputs,specs)

    def prefill(self,x):
        """Full prefix; returns output and a read-only, resident KV snapshot."""
        import torch_intel_npu as npu
        with npu._lock:
            self._validate_input(x);value=npu._materialize(x)
            if self._use_fused_norm(value):
                if self._whole_layer:
                    output,keys,values=self._run_fused(value,segment='layer',kv=True)
                    return output,SmolKVCache(self._cache_owner,object(),keys,values,x.shape[1],npu._scalar(x.shape[1],torch.float32).view(1))
                value,keys,values=self._run_fused(value,segment='attention',kv=True)
                output=self._run_fused(value,segment='ffn')[0]
                return output,SmolKVCache(self._cache_owner,object(),keys,values,x.shape[1],npu._scalar(x.shape[1],torch.float32).view(1))
            normalized=self._norm_device(value,'input_layernorm.weight',reuse=True)
            value,keys,values=self._run_cached_attention(value,normalized)
            normalized=self._norm_device(value,'post_attention_layernorm.weight',reuse=True)
            output=self._run_graph(value,normalized,segment='ffn')
            length=x.shape[1];position=npu._scalar(length,torch.float32).view(1)
            return output,SmolKVCache(self._cache_owner,object(),keys,values,length,position)

    def _decode_whole(self,x,cache,plan=None):
        output,keys,values,position=self._run_fused(x,segment='layer',cache=cache,kv=True,plan=plan)
        return output,SmolKVCache(self._cache_owner,cache._sequence,keys,values,cache.length+1,position)

    def decode(self,x,cache):
        """Append exactly one token, without mutating the previous snapshot."""
        import torch_intel_npu as npu
        with npu._lock:
            self._validate_input(x);self._validate_cache(x,cache);value=npu._materialize(x)
            if self._use_fused_norm(value):
                if self._whole_layer:
                    return self._decode_whole(value,cache)
                value,keys,values,position=self._run_fused(value,segment='attention',cache=cache,kv=True)
                output=self._run_fused(value,segment='ffn')[0]
                return output,SmolKVCache(self._cache_owner,cache._sequence,keys,values,cache.length+1,position)
            normalized=self._norm_device(value,'input_layernorm.weight',reuse=True)
            value,keys,values,position=self._run_cached_attention(value,normalized,cache)
            normalized=self._norm_device(value,'post_attention_layernorm.weight',reuse=True)
            output=self._run_graph(value,normalized,segment='ffn')
            return output,SmolKVCache(self._cache_owner,cache._sequence,keys,values,cache.length+1,position)


@dataclass(frozen=True,eq=False)
class SmolKVCache:
    """Fixed 64-slot GQA state. Tensor contents must not be edited by callers."""
    _owner:object
    _sequence:object
    keys:torch.Tensor
    values:torch.Tensor
    length:int
    _position:torch.Tensor


class FrozenSmolStack(torch.nn.Module):
    """Contiguous decoder layers; intermediates remain FP16 NPU tensors.

    Reuses the single-layer execution path, including its shape/precision
    limits. Explicit prefill/decode also keep every layer's KV on the NPU.
    """
    mode=FrozenTransformer.mode

    def __init__(self,directory,*,start_layer=0,num_layers=2,mode='w8a16',packed_weights=None,fp16_projections=(),corrections=None):
        super().__init__()
        self._fp16_projections=_projection_policy(mode,fp16_projections)
        directory=Path(directory);config=json.loads((directory/'config.json').read_text())
        total=config.get('num_hidden_layers',0)
        if (type(start_layer) is not int or type(num_layers) is not int or type(total) is not int
            or start_layer<0 or num_layers<=0 or start_layer+num_layers>total):
            raise ValueError('Requires a nonempty contiguous checkpoint layer range')
        if mode not in _WEIGHT_FORMATS:raise ValueError('Requires fp16, w8a16, w4a16 or nf4a16')
        if packed_weights is not None and (mode=='fp16' or type(packed_weights) is not dict
            or any(type(i) is not int for i in packed_weights) or any(type(value) is not dict for value in packed_weights.values())
            or set(packed_weights)!=set(range(start_layer,start_layer+num_layers))):
            raise ValueError('Prepared stack weights require a complete lowbit map indexed by checkpoint layer')
        if corrections is not None and (mode!='w8a16' or packed_weights is None or type(corrections) is not dict
            or any(type(i) is not int or type(v) is not dict for i,v in corrections.items())
            or set(corrections)!=set(range(start_layer,start_layer+num_layers))):
            raise ValueError('Stack corrections require explicit W8 weights and a complete map indexed by checkpoint layer')
        layers=[]
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:
            names=list(file.keys())
            for index in range(start_layer,start_layer+num_layers):
                prefix=f'model.layers.{index}.'
                weights={name[len(prefix):]:file.get_tensor(name) for name in names if name.startswith(prefix)}
                layers.append(FrozenSmolLayer(config,weights,mode=mode,packed_weights=None if packed_weights is None else packed_weights[index],
                    fp16_projections=fp16_projections,corrections=None if corrections is None else corrections[index]))
        self.layers=torch.nn.ModuleList(layers);self.start_layer=start_layer;self._mode=mode
        self.eval()

    @classmethod
    def from_checkpoint(cls,directory,**options):return cls(directory,**options)

    @property
    def num_layers(self):return len(self.layers)

    @property
    def fp16_projections(self):return self._fp16_projections

    def reference_cpu(self,x,*,quantized=True):
        """Explicit staged CPU reference; never called by forward."""
        traces=[]
        for layer in self.layers:
            x,trace=layer.reference_cpu(x,quantized=quantized);traces.append(trace)
        return x,traces

    def forward(self,x):
        import torch_intel_npu as npu
        with npu._lock:
            if self.training or any(layer.training for layer in self.layers):
                raise ValueError('FrozenSmolStack requires every layer in eval; training is unsupported')
            for layer in self.layers:x=layer(x)
            return x

    def prefill(self,x):
        import torch_intel_npu as npu
        with npu._lock:
            if self.training or any(layer.training for layer in self.layers):
                raise ValueError('FrozenSmolStack requires every layer in eval')
            caches=[];sequence=object()
            for layer in self.layers:
                x,cache=layer.prefill(x);caches.append(replace(cache,_sequence=sequence))
            return x,tuple(caches)

    def decode(self,x,caches,*,group_size=None):
        """Synchronous decode; group_size=1 retains the original per-layer path."""
        import torch_intel_npu as npu
        with npu._lock:
            if group_size is not None and (type(group_size) is not int or group_size not in (1,2,4)):
                raise ValueError('Decode group_size must be 1, 2 or 4')
            if self.training or any(layer.training for layer in self.layers):
                raise ValueError('FrozenSmolStack requires every layer in eval')
            if type(caches) is not tuple or len(caches)!=self.num_layers:
                raise ValueError('Requires one KV cache per consecutive layer')
            self.layers[0]._validate_input(x)
            for layer,cache in zip(self.layers,caches):layer._validate_cache(x,cache)
            if any(cache.length!=caches[0].length or cache._sequence is not caches[0]._sequence for cache in caches):
                raise ValueError('Stack KV caches must belong to the same snapshot and position')
            supported=all(layer._whole_layer and layer._use_fused_norm(x) for layer in self.layers)
            # ponytail: default limited to measured batches; expand after full-stack benchmarks.
            if group_size is None:group_size=4 if supported and self.num_layers>1 and x.shape[0] in (1,4,8) else 1
            if group_size>1 and not supported:
                raise ValueError('Grouped decode requires whole-layer fused RMSNorm on every layer')
            updated=[];sequence=object()
            if group_size>1:
                with npu._smol_graph_plan():
                    plan=[];x=npu._materialize(x)
                    for layer,cache in zip(self.layers,caches):
                        x,cache=layer._decode_whole(x,cache,plan);updated.append(replace(cache,_sequence=sequence))
                    npu._run_chain(plan,group_size)
                    return x,tuple(updated)
            for layer,cache in zip(self.layers,caches):
                x,cache=layer.decode(x,cache);updated.append(replace(cache,_sequence=sequence))
            return x,tuple(updated)


# Cache shortcuts apply only to these unchanged graph-building entry points.
_PREFILL_GRAPH=FrozenSmolLayer.graph
_PREFILL_SERIALIZER=s.serialize_ir
_PREFILL_WEIGHT_GRAPH=weight_graph
