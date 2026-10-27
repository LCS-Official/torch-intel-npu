"""Explicit frozen causal Transformer inference on the validated Intel NPU.

Accepts CPU pre-norm, GELU TransformerEncoderLayer/Encoder; no OpenVINO runtime.
CPU reference/calibration are explicit preparation, never a forward fallback.
"""
from dataclasses import replace
import hashlib
from math import ceil, isfinite, log2, prod
import struct
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
import npunlock as s
from npunlock.npu4000 import Elf, MAX_TENSOR_ELEMENTS
from torch_npu_lowbit import quantize_weight, weight_graph, dpu_modes


def _half_bytes(tensor, shape):
    if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or tuple(tensor.shape)!=shape
        or tensor.dtype not in (torch.float16,torch.float32) or not torch.isfinite(tensor).all()):
        raise ValueError('Preparation requires finite CPU FP16/FP32 parameters of the expected shape')
    value=tensor.detach().half().contiguous()
    if not torch.isfinite(value).all():raise ValueError('Parameter overflows FP16')
    return value.numpy().tobytes()


class FrozenTransformer(torch.nn.Module):
    """1-2 causal pre-norm blocks; FP16, W8A16, or selective W8A8 projections."""
    def __init__(self,source,*,mode='w8a16',calibration_samples=()):
        super().__init__()
        if mode not in ('fp16','w8a16','mixed_int8'):raise ValueError('Requires fp16/w8a16/mixed_int8; full W8A8 Transformer is not verified')
        if isinstance(source,torch.nn.TransformerEncoderLayer):layers=[source]
        elif isinstance(source,torch.nn.TransformerEncoder) and source.norm is None:layers=list(source.layers)
        else:raise ValueError('Requires a CPU TransformerEncoderLayer or Encoder without final norm')
        if source.training or not 1<=len(layers)<=2:raise ValueError('Requires eval mode and 1-2 blocks')
        self._mode=mode;self._blocks=[];self._keys={};self._evidence={};self._norm_parameters={}
        for layer in layers:
            a=layer.self_attn;d=a.embed_dim;ff=layer.linear1.out_features
            # ponytail: verified 32-wide heads, D128/256 and FF=4D only;
            # expand shapes when separate numerical/descriptor checks pass.
            if (not layer.norm_first or not a.batch_first or d not in (128,256) or a.num_heads!=d//32
                or ff!=4*d or layer.activation_relu_or_gelu!=2 or not a._qkv_same_embed_dim
                or a.bias_k is not None or a.bias_v is not None or a.add_zero_attn
                or any(drop.p!=0 for drop in (layer.dropout,layer.dropout1,layer.dropout2)) or a.dropout!=0):
                raise ValueError('Requires pre-norm, batch_first, GELU, dropout=0, D128/256, 32-wide heads, FF=4D')
            if self._blocks and d!=self._d:raise ValueError('Blocks must have the same width')
            self._d=d;self._heads=a.num_heads
            pairs=[(a.in_proj_weight,a.in_proj_bias),(a.out_proj.weight,a.out_proj.bias),
                   (layer.linear1.weight,layer.linear1.bias),(layer.linear2.weight,layer.linear2.bias)]
            block={'linear':tuple((_half_bytes(w,tuple(w.shape)),_half_bytes(b,(w.shape[0],)),tuple(w.shape)) for w,b in pairs),
                   'norm':tuple((_half_bytes(n.weight,(d,)),_half_bytes(n.bias,(d,)),float(n.eps)) for n in (layer.norm1,layer.norm2))}
            if any(not isfinite(v[2]) or v[2]<=0 for v in block['norm']):raise ValueError('Requires finite positive norm eps')
            self._blocks.append(block)
        self._blocks=tuple(self._blocks)
        maxima={f'b{i}_{name}':0. for i in range(len(layers)) for name in ('qkv','out','ff1','ff2')}
        samples=list(calibration_samples)
        if mode=='mixed_int8' and not samples:raise ValueError('Mixed INT8 requires CPU calibration samples')
        for sample in samples:
            _,trace=self.reference_cpu(sample,quantized=False)
            for name in maxima:maxima[name]=max(maxima[name],float(trace[name+'_input'].abs().max()))
        self._scales=tuple(tuple(2.**max(-24,ceil(log2(max(maxima[f'b{i}_{name}']*1.05/127,2.**-24)))) if name in ('qkv','ff1') else None
            for name in ('qkv','out','ff1','ff2')) for i in range(len(layers))) if mode=='mixed_int8' else None
        if self._scales is not None and any(v is not None and v>256 for group in self._scales for v in group):raise ValueError('Calibration range exceeds supported activation scale')
        self._packed=tuple(tuple(quantize_weight(torch.from_numpy(np.frombuffer(w,dtype=np.float16).reshape(shape).copy()),format='int8')
            for w,_,shape in block['linear']) for block in self._blocks) if mode!='fp16' else None
        self._calibration=tuple((name,maxima[name]) for name in maxima)
        self.eval()

    @classmethod
    def from_float(cls,source,**options):return cls(source,**options)

    @property
    def mode(self):return self._mode

    @property
    def activation_scales(self):return self._scales

    @property
    def calibration_maxima(self):return self._calibration

    def _shape(self,shape):
        if (len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape) or shape[2]!=self._d
            or shape[1]>64 or max(prod(shape),shape[0]*shape[1]*4*self._d,
                                 shape[0]*self._heads*shape[1]**2)>MAX_TENSOR_ELEMENTS):
            raise ValueError('Requires bounded (B,S,D) with S<=64 and matching D')
        return tuple(shape)

    def reference_cpu(self,x,*,quantized=True,segment=None,block_index=0,normalized=None):
        """CPU-only staged reference and calibration traces, no device download."""
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype not in (torch.float16,torch.float32)
            or not torch.isfinite(x).all()):raise ValueError('Reference/calibration requires finite CPU FP16/FP32 input')
        b,length,d=self._shape(tuple(x.shape));value=x.detach().half();original=value;trace={}
        if not torch.isfinite(value).all():raise ValueError('Calibration input overflows FP16')
        def norm(v,p):
            w,bias,eps=p
            return F.layer_norm(v.float(),(d,),torch.from_numpy(np.frombuffer(w,dtype=np.float16).copy()).float(),
                torch.from_numpy(np.frombuffer(bias,dtype=np.float16).copy()).float(),eps=eps).half()
        blocks=enumerate(self._blocks) if segment is None else [(block_index,self._blocks[block_index])]
        if segment is not None and (segment not in ('attention','ffn') or normalized is None or normalized.device.type!='cpu'
            or normalized.shape!=x.shape or normalized.dtype!=torch.float16):raise ValueError('CPU segment reference requires matching CPU FP16 normalization')
        for i,block in blocks:
            def linear(v,j,name):
                flat=v.reshape(-1,v.shape[-1]);trace[f'b{i}_{name}_input']=flat
                w,bias,shape=block['linear'][j]
                if quantized and self._mode!='fp16':
                    q=self._packed[i][j];codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape)
                    scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)
                    if self._scales is not None and self._scales[i][j] is not None:
                        step=self._scales[i][j]
                        act=np.clip(np.rint(flat.numpy().astype(np.float64)/step),-128,127).astype(np.int64)
                        accum=np.einsum('mk,nk->mn',act,codes.astype(np.int64),optimize=False)
                        lo=(-128*scales).astype(np.float16).astype(np.float64);hi=(127*scales).astype(np.float16).astype(np.float64)
                        out=torch.from_numpy(accum.astype(np.float64)*step*((hi-lo)/255)).half()
                    else:out=(flat.double()@torch.from_numpy(codes.astype(np.float64)*scales[:,None]).t()).half()
                else:out=(flat.float()@torch.from_numpy(np.frombuffer(w,dtype=np.float16).reshape(shape).copy()).float().t()).half()
                out=(out+torch.from_numpy(np.frombuffer(bias,dtype=np.float16).copy())).half()
                trace[f'b{i}_{name}']=out;return out
            v=normalized if segment=='attention' else norm(value,block['norm'][0]);trace[f'b{i}_norm1']=v
            qkv=linear(v,0,'qkv').view(b,length,3,self._heads,32)
            q,k,v=(qkv[:,:,j].permute(0,2,1,3) for j in range(3))
            scores=(q.float()@k.float().transpose(-1,-2)).half()
            scores=(scores*np.float16(1/32**.5)).half()
            mask=torch.triu(torch.full((length,length),float('-inf')),diagonal=1)
            scores=(scores+mask).half();probs=scores.float().softmax(-1).half()
            attended=(probs.float()@v.float()).half().permute(0,2,1,3).reshape(b,length,d)
            trace[f'b{i}_attention']=attended
            value=(value+linear(attended,1,'out').view(b,length,d)).half();trace[f'b{i}_residual1']=value
            if segment=='attention':break
            if segment=='ffn':value=original
            v=normalized if segment=='ffn' else norm(value,block['norm'][1]);trace[f'b{i}_norm2']=v
            v=linear(v,2,'ff1');v=F.gelu(v.float()).half();trace[f'b{i}_gelu']=v
            value=(value+linear(v,3,'ff2').view(b,length,d)).half();trace[f'b{i}_residual2']=value
        return value,trace

    def graph(self,shape,*,stage=None,segment=None,block_index=0):
        b,length,d=self._shape(shape);rows=b*length;decodes=[];trace={}
        inp=s.input('x',shape=shape,dtype='f16')
        inputs=[inp]
        if segment in ('attention','ffn'):
            normalized=s.input('normalized',shape=shape,dtype='f16');inputs.append(normalized)
        def op(name,*values,shape=None,**attrs):
            return s.op(name,*values,_shape=shape or values[0].shape,_dtype='f16',**attrs)
        def reshape(v,shape):return op('Reshape',v,s.constant(np.array(shape,dtype=np.int64)),shape=shape,special_zero=False)
        def norm(v,p,name):
            w,bias,eps=p
            v=op('MVN',v,s.constant(np.array([-1],dtype=np.int64)),normalize_variance=True,eps=eps,eps_mode='inside_sqrt')
            v=op('Multiply',v,s.constant(np.frombuffer(w,dtype=np.float16)),auto_broadcast='numpy')
            v=op('Add',v,s.constant(np.frombuffer(bias,dtype=np.float16)),auto_broadcast='numpy')
            trace[name]=v;return v
        value=inp
        blocks=enumerate(self._blocks) if segment is None else [(block_index,self._blocks[block_index])]
        for i,block in blocks:
            def linear(v,j,name):
                v=reshape(v,(rows,v.shape[-1]));w,bias,wshape=block['linear'][j]
                if self._mode=='fp16':
                    v=op('MatMul',v,s.constant(np.frombuffer(w,dtype=np.float16).reshape(wshape)),
                         shape=(rows,wshape[0]),transpose_a=False,transpose_b=True)
                    v=op('Add',v,s.constant(np.frombuffer(bias,dtype=np.float16)),auto_broadcast='numpy')
                else:
                    graph,xml,_=weight_graph(self._packed[i][j],rows,bias=bias,
                        activation_scale=self._scales[i][j] if self._scales is not None else None)
                    prefix=name+'_';mapped={graph.inputs[0]:v}
                    for node in graph.nodes:
                        if node.op=='Parameter':continue
                        cloned=replace(node,inputs=tuple(mapped[t] for t in node.inputs),name=prefix+node.name if node.name else None)
                        mapped.update(zip(node.outputs,cloned.outputs))
                    decodes.extend(prefix+l.attrib['name'] for l in ET.fromstring(xml).findall('./layers/layer')
                        if l.find("./rt_info/attribute[@name='decompression']") is not None)
                    v=mapped[graph.outputs[0]]
                trace[name]=v;return v
            v=normalized if segment=='attention' else norm(value,block['norm'][0],f'b{i}_norm1')
            qkv=linear(v,0,f'b{i}_qkv');heads=[]
            for j in range(3):
                args=[s.constant(np.array(a,dtype=np.int64)) for a in ([0,j*d],[rows,(j+1)*d],[1,1])]
                v=op('StridedSlice',qkv,*args,shape=(rows,d),begin_mask=[0,0],end_mask=[0,0],new_axis_mask=[0,0],shrink_axis_mask=[0,0],ellipsis_mask=[0,0])
                v=reshape(v,(b,length,self._heads,32))
                heads.append(op('Transpose',v,s.constant(np.array([0,2,1,3],dtype=np.int64)),shape=(b,self._heads,length,32)))
            q,k,v=heads
            scores=op('MatMul',q,k,shape=(b,self._heads,length,length),transpose_a=False,transpose_b=True)
            scores=op('Multiply',scores,s.constant(np.float16(1/32**.5)),auto_broadcast='numpy')
            mask=np.where(np.arange(length)[None,:]>np.arange(length)[:,None],-np.inf,0).astype(np.float16)
            scores=op('Add',scores,s.constant(mask),auto_broadcast='numpy')
            probs=op('Softmax',scores,axis=3)
            attended=op('MatMul',probs,v,shape=(b,self._heads,length,32),transpose_a=False,transpose_b=False)
            attended=op('Transpose',attended,s.constant(np.array([0,2,1,3],dtype=np.int64)),shape=(b,length,self._heads,32))
            attended=reshape(attended,shape);trace[f'b{i}_attention']=attended
            value=op('Add',value,reshape(linear(attended,1,f'b{i}_out'),shape),auto_broadcast='numpy');trace[f'b{i}_residual1']=value
            if segment=='attention':break
            if segment=='ffn':value=inp
            v=normalized if segment=='ffn' else norm(value,block['norm'][1],f'b{i}_norm2')
            v=linear(v,2,f'b{i}_ff1');v=op('Gelu',v,approximation_mode='erf');trace[f'b{i}_gelu']=v
            value=op('Add',value,reshape(linear(v,3,f'b{i}_ff2'),shape),auto_broadcast='numpy');trace[f'b{i}_residual2']=value
        value=trace[stage] if stage is not None else value
        value=op('Reshape',value,s.constant(np.array(value.shape,dtype=np.int64)),special_zero=False,_name='y')
        graph=s.Graph(inputs,[value],name='frozen_transformer')
        if any(prod(t.shape)>MAX_TENSOR_ELEMENTS for node in graph.nodes for t in node.outputs):raise ValueError('Intermediate exceeds storage bound')
        ir=s.serialize_ir(graph);xml=ET.fromstring(ir.xml)
        for layer in xml.findall('./layers/layer'):
            if layer.attrib['type'] in ('MVN','Gelu'):layer.set('version',{'MVN':'opset6','Gelu':'opset7'}[layer.attrib['type']])
            if layer.attrib['name'] in decodes:
                info=ET.SubElement(layer,'rt_info');ET.SubElement(info,'attribute',name='decompression',version='0')
        return graph,ET.tostring(xml),ir.weights

    def _compile_graph(self,shape,segment,block_index,prepared=None):
        import torch_intel_npu as npu
        graph,xml,data=self.graph(shape,segment=segment,block_index=block_index) if prepared is None else prepared
        result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
        if (result.vendor_id,result.device_id,tuple(result.compiler_version))!=(0x8086,0x643e,(8,3)):raise ValueError('Unsupported Transformer profile')
        program=s.load_native(result.graph_blob,graph=graph,libraries=npu._native)
        self._evidence[(shape,segment,block_index)]=self._inspect_blob(program.graph_blob)
        return program

    def _run_graph(self,x,normalized=None,*,segment=None,block_index=0):
        import torch_intel_npu as npu
        with npu._lock:
            shape=self._shape(tuple(x.shape))
            cache=(shape,segment,block_index);prepared=None
            if cache not in self._keys:
                prepared=self.graph(shape,segment=segment,block_index=block_index)
                self._keys[cache]=('frozen_transformer',hashlib.sha256(prepared[1]+prepared[2]).digest())
            key=self._keys[cache]
            factory=lambda:self._compile_graph(shape,segment,block_index,prepared)
            npu._register_graph_factory(key,factory)
            if cache not in self._evidence and key in npu._sessions:
                self._evidence[cache]=self._inspect_blob(npu._sessions[key][1].graph_blob)
            return npu._run(self._keys[cache],[x] if normalized is None else [x,normalized],shape,torch.float16)

    def _inspect_blob(self,blob):
        elf=Elf(blob);modes=dpu_modes(blob)
        for mode in modes:
            raw=elf.content(elf.named[mode['section']])
            mode['channels']=struct.unpack_from('<I',raw,mode['offset']+0x2c)[0]&16383
        matrices=[m for m in modes if not m['elementwise']]
        # A non-elementwise FP16 task can also transpose/layout tensors.
        # These modes prove INT8 matrix operands, not a mapping of every IR op.
        integer=[m for m in matrices if m['wmode'] in (1,2)]
        if not matrices:raise ValueError('Missing DPU matrix tasks')
        if self.mode!='fp16' and not integer:raise ValueError('Missing INT8 weight DPU tasks')
        if self.mode=='mixed_int8' and (not any(m['amode'] in (1,2) for m in integer)
            or not any(m['amode']==0 for m in integer)):
            raise ValueError(f'Missing mixed W8A8/W8A16 matrix operand proof: {modes}')
        return dict(modes=modes,int8_matrix_tasks=len(integer),blob_sha256=hashlib.sha256(blob).hexdigest(),
            constant_bytes=sum(v['size'] for v in elf.sections if v['name'].startswith('buffer.Constant.')),
            shave_invocations=sum(bool(v['size']) and v['name'].startswith('task.shave.invocation.') for v in elf.sections))

    def _norm_device(self,x,p,block_index,norm_index):
        import torch_intel_npu as npu
        value=x.float();centered=value-value.mean(-1,keepdim=True)
        variance=(centered*centered).mean(-1,keepdim=True)
        normalized=centered*npu._execute('rsqrt',variance+p[2])
        if not (np.all(np.frombuffer(p[0],dtype=np.float16)==1) and np.all(np.frombuffer(p[1],dtype=np.float16)==0)):
            cache=(block_index,norm_index)
            if cache not in self._norm_parameters:
                self._norm_parameters[cache]=tuple(torch.from_numpy(np.frombuffer(data,dtype=np.float16).astype(np.float32)).view(1,1,self._d).to('intel_npu') for data in p[:2])
            gamma,bias=self._norm_parameters[cache]
            normalized=normalized*gamma+bias
        return normalized.half()

    def forward(self,x):
        import torch_intel_npu as npu
        with npu._lock:
            if self.training:raise ValueError('FrozenTransformer is inference-only; use eval()')
            npu.synchronize();npu._validate(x,dense=True);self._shape(tuple(x.shape))
            if x.dtype!=torch.float16:raise ValueError('FrozenTransformer requires FP16 NPU input')
            value=npu._materialize(x)
            if self.mode!='mixed_int8':return self._run_graph(value)
            # The driver MVN's half rounding can flip activation quantization.
            # Use the existing true-FP32 SHAVE path at each norm boundary.
            for i,block in enumerate(self._blocks):
                normalized=self._norm_device(value,block['norm'][0],i,0)
                value=self._run_graph(value,normalized,segment='attention',block_index=i)
                normalized=self._norm_device(value,block['norm'][1],i,1)
                value=self._run_graph(value,normalized,segment='ffn',block_index=i)
            return value
