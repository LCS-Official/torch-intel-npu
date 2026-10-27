"""Experimental immutable INT4/NF4/INT8 weights, FP16 IO, driver DPU.

NF4 codebook: Intel OpenVINO's published nf4.cpp, also used by bitsandbytes.
https://github.com/openvinotoolkit/openvino/blob/master/src/core/src/type/nf4.cpp
No OpenVINO runtime or bitsandbytes dependency; CPU quantization is preparation.
"""
from dataclasses import dataclass
import hashlib
from math import prod, isfinite, frexp
import struct
import xml.etree.ElementTree as ET
import numpy as np
import torch
import npunlock as symbolic
from npunlock.npu4000 import Elf, MAX_TENSOR_ELEMENTS

NF4 = np.array([-1., -.6961928009986877, -.5250730514526367, -.39491748809814453,
    -.28444138169288635, -.18477343022823334, -.09105003625154495, 0.,
    .07958029955625534, .16093020141124725, .24611230194568634, .33791524171829224,
    .44070982933044434, .5626170039176941, .7229568362236023, 1.], dtype=np.float32)
NF4.flags.writeable = False


def pack_codes(codes):
    values = np.asarray(codes)
    if values.dtype != np.uint8 or not values.size or np.any(values > 15):
        raise ValueError('Codes must be a nonempty uint8 array in 0..15')
    flat = values.reshape(-1)
    padded = np.pad(flat, (0, flat.size % 2))
    return (padded[::2] | padded[1::2] << 4).tobytes()


@dataclass(frozen=True)
class PackedWeight:
    shape: tuple
    format: str
    group_size: int
    data: bytes
    scales: bytes

    def __post_init__(self):
        if (type(self.shape) is not tuple or len(self.shape) != 2 or any(type(v) is not int or v <= 0 for v in self.shape)
            or prod(self.shape) > MAX_TENSOR_ELEMENTS or self.format not in ('int4', 'nf4', 'int8')
            or type(self.group_size) is not int or not 1 <= self.group_size <= self.shape[1]
            or self.shape[1] % self.group_size):
            raise ValueError('Weight requires bounded (N,K), int4/nf4/int8, and a group size dividing K')
        if not isinstance(self.data, bytes) or not isinstance(self.scales, bytes):
            raise ValueError('Packed data/scales must be immutable bytes')
        count = prod(self.shape)
        if len(self.data) != (count if self.format=='int8' else (count+1)//2) or (self.format!='int8' and count % 2 and self.data[-1] & 0xf0):
            raise ValueError('Packed weight extent/padding differs')
        if len(self.scales) != 2*count//self.group_size:
            raise ValueError('Scale extent differs')
        scales = np.frombuffer(self.scales, dtype=np.float16)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError('Scales must be finite, positive FP16 values')

    @property
    def dtype(self):
        return {'int4':'i4','nf4':'nf4','int8':'i8'}[self.format]

    @property
    def grouped_shape(self):
        n, k = self.shape
        return n, k//self.group_size, self.group_size

    def dequantize(self):
        """CPU reference only, including both explicit FP16 rounding stages."""
        if self.format=='int8':values=np.frombuffer(self.data,dtype=np.int8).astype(np.float16)
        else:
            raw = np.frombuffer(self.data, dtype=np.uint8)
            codes = np.column_stack((raw & 15, raw >> 4)).reshape(-1)[:prod(self.shape)]
            values = ((codes.astype(np.int8) ^ 8)-8).astype(np.float16) if self.format == 'int4' else NF4[codes].astype(np.float16)
        scales = np.frombuffer(self.scales, dtype=np.float16).reshape(*self.grouped_shape[:2], 1)
        return (values.reshape(self.grouped_shape)*scales).reshape(self.shape)


def quantize_weight(weight, *, format='int4', group_size=None):
    """Prepare symmetric frozen Linear weights on CPU, with FP16 scales."""
    if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2
        or weight.dtype not in (torch.float16,torch.float32)):
        raise ValueError('Quantization preparation requires a CPU 2D Tensor')
    n, k = weight.shape
    size = k if group_size is None else group_size
    if (format not in ('int4', 'nf4', 'int8') or type(size) is not int or not 1 <= size <= k
        or k % size or not 0 < n*k <= MAX_TENSOR_ELEMENTS):
        raise ValueError('Requires int4/nf4/int8 and a group size dividing bounded K')
    values = weight.detach().float().numpy().reshape(n, k//size, size)
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite weights cannot be quantized')
    maximum = np.max(np.abs(values), axis=-1, keepdims=True)
    limit={'int4':7,'nf4':1,'int8':127}[format]
    scale = maximum/limit
    if np.any(scale > np.finfo(np.float16).max):
        raise ValueError('Weight scale overflows FP16')
    scale = np.maximum(scale, np.float32(2**-24)).astype(np.float16)
    normalized = values/scale.astype(np.float32)
    if format in ('int4','int8'):
        codes = np.clip(np.rint(normalized), -limit, limit).astype(np.int8)
        data=codes.tobytes() if format=='int8' else pack_codes((codes & 15).astype(np.uint8))
    else:
        boundaries = (NF4[:-1]+NF4[1:])/2
        codes = np.searchsorted(boundaries, normalized, side='left').astype(np.uint8)
        data=pack_codes(codes)
    return PackedWeight((n,k), format, size, data, scale.tobytes())


@torch.no_grad()
def calibrate_int8(weight, inputs):
    """CPU-only frozen W8 preparation from representative Linear inputs.

    GPTQ adaptation (Apache-2.0, LICENSES/GPTQ.txt):
    https://github.com/IST-DASLab/gptq/blob/main/gptq.py
    Uses CPU FP64, fixed FP16 per-row scales, symmetric -127..127 codes,
    stable activation order and 1% damping. Keeps original dead-column
    weights and falls back per row if calibration reconstruction worsens.
    No grouping, autograd, NPU execution or OpenVINO dependency.
    """
    original=quantize_weight(weight,format='int8')
    if (not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or inputs.shape[1]!=weight.shape[1] or not 1<=inputs.shape[0]<=4096
        or inputs.shape[1]>4096 or max(inputs.numel(),inputs.shape[0]*weight.shape[0])>4*MAX_TENSOR_ELEMENTS
        or not torch.isfinite(inputs).all()):
        raise ValueError('Calibration requires bounded finite CPU FP16/FP32 (rows,K), rows/K<=4096, matching weight K')
    activation=inputs.detach().double();h=activation.t()@activation/len(activation)
    energy=h.diagonal().clone();order=torch.argsort(energy,descending=True,stable=True)
    h=h[order][:,order];h.diagonal()[energy[order]==0]=1
    h.diagonal().add_(.01*h.diagonal().mean().clamp_min(1e-20))
    upper=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True)
    values=weight.detach().double()[:,order].clone()
    scales=torch.from_numpy(np.frombuffer(original.scales,dtype=np.float16).astype(np.float64))
    codes=torch.empty_like(values,dtype=torch.int8);width=weight.shape[1]
    # ponytail: fixed 128-column blocks; extend to grouped formats only after
    # separate preparation and driver numerical checks.
    for start in range(0,width,128):
        end=min(start+128,width);block=values[:,start:end].clone();residual=torch.zeros_like(block)
        for i in range(end-start):
            column=block[:,i].clone();code=(column/scales).round().clamp(-127,127)
            codes[:,start+i]=code.to(torch.int8)
            error=(column-code*scales)/upper[start+i,start+i]
            block[:,i:].sub_(error[:,None]*upper[start+i,start+i:end][None])
            residual[:,i]=error
        values[:,end:].sub_(residual@upper[start:end,end:])
    proposed=codes[:,torch.argsort(order)].contiguous().numpy()
    original_codes=np.frombuffer(original.data,dtype=np.int8).reshape(original.shape).copy()
    target=activation@weight.detach().double().t()
    def reconstruction(c):
        prediction=(activation@torch.from_numpy(c.astype(np.float64)).t())*scales[None]
        return (prediction-target).square().sum(0).numpy()
    improved=reconstruction(proposed)<reconstruction(original_codes)
    original_codes[improved]=proposed[improved]
    return PackedWeight(original.shape,'int8',original.group_size,original_codes.tobytes(),original.scales)


def _activation_scale(weight, scale, compact_groups):
    if scale is None:return None
    if (type(scale) not in (int,float) or not isfinite(scale) or not 2**-24<=scale<=511
        or weight.format!='int8' or weight.group_size!=weight.shape[1] or compact_groups):
        raise ValueError('Activation quantization requires per-channel INT8, no compact groups, and a finite scale in [2^-24,511]')
    # ponytail: exact power-of-two activation grids avoid driver 8.3 rounding
    # drift at FQ thresholds; arbitrary scales need a separately verified kernel.
    if frexp(scale)[0]!=.5:
        raise ValueError('Activation scale must be an exact power of two on driver 8.3')
    if np.any(np.frombuffer(weight.scales,dtype=np.float16)>511):
        raise ValueError('INT8 weight quantization range overflows FP16')
    return float(np.float16(scale))


def dpu_modes(blob):
    """Inspect the validated 8.3 NPU40XX ABI; never edit descriptors.

    Intel npu_compiler: types.td tensor_mode 0x34, firmware VpuDPUInvariant
    size 352, VpuInputTensorDType FP16=0 / U8=1 / I8=2.
    """
    elf=Elf(blob);result=[]
    for section in elf.sections:
        if not section['size'] or not section['name'].startswith('task.dpu.invariant.'):continue
        if section['size']%352:raise ValueError('Unexpected NPU40XX invariant extent')
        raw=elf.content(section)
        for offset in range(0,len(raw),352):
            tensor=struct.unpack_from('<I',raw,offset+0x34)[0]
            result.append(dict(section=section['name'],offset=offset,amode=(tensor>>4)&15,wmode=tensor&15,
                elementwise=bool(struct.unpack_from('<I',raw,offset+0x40)[0]&1),
                mpe_mode=(struct.unpack_from('<I',raw,offset+0x80)[0]>>16)&7,
                palette=(struct.unpack_from('<I',raw,offset+0x24)[0]>>13)&3))
    return result


def weight_graph(weight, batch, *, bias=None, reference_fp16=False, compact_groups=False, pad_batch=False, activation_scale=None):
    """Build compressed-only dequantization + MatMul; all outputs stay FP16."""
    n, k = weight.shape
    if type(batch) is not int or batch <= 0 or max(batch*k,batch*n) > MAX_TENSOR_ELEMENTS:
        raise ValueError('Activation/output exceeds bounded 2D storage')
    if type(compact_groups) is not bool or type(pad_batch) is not bool:
        raise ValueError('Graph options require bool values')
    activation_scale=_activation_scale(weight,activation_scale,compact_groups)
    if activation_scale is not None and reference_fp16:
        raise ValueError('FP16 reference graph does not quantize activations')
    groups=k//weight.group_size
    if compact_groups and groups>32:
        raise ValueError('Compact graph supports at most 32 groups')
    physical=max(batch,32) if pad_batch else batch
    if max(physical*k,physical*n)>MAX_TENSOR_ELEMENTS:
        raise ValueError('Padded activation/output exceeds bounded storage')
    x = symbolic.input('x', shape=(batch,k), dtype='f16')
    value=x
    if physical!=batch:
        value=symbolic.op('Concat',x,symbolic.constant(np.zeros((physical-batch,k),dtype=np.float16)),axis=0,
                          _shape=(physical,k),_dtype='f16')
    decodes=[]
    def quantized(value,scale,name):
        limits=[np.asarray(-128*scale,dtype=np.float16),np.asarray(127*scale,dtype=np.float16)]
        if not all(np.all(np.isfinite(v)) for v in limits):raise ValueError('Quantization range overflows FP16')
        return symbolic.op('FakeQuantize',value,*(symbolic.constant(v) for v in limits*2),levels=256,
                           auto_broadcast='numpy',_shape=value.shape,_dtype='f16',_name=name)
    def compressed(data,shape,scales,name):
        packed=(symbolic.constant(np.frombuffer(data,dtype=np.int8).reshape(shape),name=name) if weight.format=='int8'
                else symbolic.constant_packed(data,shape=shape,dtype=weight.dtype,name=name))
        decodes.append('decode_'+name)
        decoded=symbolic.op('Convert',packed,destination_type='f16',_shape=shape,_dtype='f16',_name=decodes[-1])
        scale=symbolic.constant(scales)
        return symbolic.op('Multiply',decoded,scale,auto_broadcast='numpy',_shape=shape,_dtype='f16')
    scales=np.frombuffer(weight.scales,dtype=np.float16).reshape(n,groups,1)
    if activation_scale is not None:
        value=quantized(value,activation_scale,'quantize_activation')
        packed=symbolic.constant(np.frombuffer(weight.data,dtype=np.int8).reshape(weight.shape),name='weight_int8')
        decoded=symbolic.op('Convert',packed,destination_type='f16',_shape=weight.shape,_dtype='f16')
        channel_scales=scales.reshape(n,1)
        w=symbolic.op('Multiply',decoded,symbolic.constant(channel_scales),auto_broadcast='numpy',_shape=weight.shape,_dtype='f16')
        w=quantized(w,channel_scales,'quantize_weight')
        y=symbolic.op('MatMul',value,w,transpose_a=False,transpose_b=True,_shape=(physical,n),_dtype='f16',_name='y')
    elif reference_fp16:
        w = symbolic.constant(weight.dequantize(), name='weight_fp16')
        y=symbolic.op('MatMul',value,w,transpose_a=False,transpose_b=True,_shape=(physical,n),_dtype='f16',_name='y')
    elif compact_groups and groups>1:
        if weight.format=='int8':codes=np.frombuffer(weight.data,dtype=np.int8).reshape(weight.grouped_shape)
        else:
            raw=np.frombuffer(weight.data,dtype=np.uint8)
            codes=np.column_stack((raw&15,raw>>4)).reshape(-1)[:n*k].reshape(weight.grouped_shape)
        pieces=[]
        for group in range(groups):
            start=group*weight.group_size
            args=[symbolic.constant(np.array(v,dtype=np.int64))
                  for v in ([0,start],[physical,start+weight.group_size],[1,1])]
            part=symbolic.op('StridedSlice',value,*args,begin_mask=[0,0],end_mask=[0,0],new_axis_mask=[0,0],
                             shrink_axis_mask=[0,0],ellipsis_mask=[0,0],_shape=(physical,weight.group_size),_dtype='f16')
            data=codes[:,group,:].tobytes() if weight.format=='int8' else pack_codes(codes[:,group,:])
            w=compressed(data,(n,weight.group_size),scales[:,group,:],f'weight_{group}')
            pieces.append(symbolic.op('MatMul',part,w,transpose_a=False,transpose_b=True,_shape=(physical,n),_dtype='f16'))
        # FP16 partial outputs change rounding. Balanced addition bounds the
        # accumulation depth; opt in only when this memory/accuracy tradeoff fits.
        while len(pieces)>1:
            pieces=[symbolic.op('Add',pieces[i],pieces[i+1],auto_broadcast='numpy',_shape=(physical,n),_dtype='f16')
                    if i+1<len(pieces) else pieces[i] for i in range(0,len(pieces),2)]
        y=symbolic.op('Reshape',pieces[0],symbolic.constant(np.array((physical,n),dtype=np.int64)),
                      special_zero=False,_shape=(physical,n),_dtype='f16',_name='y')
    else:
        scaled=compressed(weight.data,weight.grouped_shape,scales,'weight_packed')
        shape = symbolic.constant(np.array(weight.shape,dtype=np.int64))
        w = symbolic.op('Reshape',scaled,shape,special_zero=False,_shape=weight.shape,_dtype='f16')
        y=symbolic.op('MatMul',value,w,transpose_a=False,transpose_b=True,_shape=(physical,n),_dtype='f16',_name='y')
    if bias is not None:
        b = np.frombuffer(bias,dtype=np.float16)
        if b.shape != (n,) or not np.all(np.isfinite(b)):
            raise ValueError('Bias requires N finite FP16 values')
        y = symbolic.op('Add',y,symbolic.constant(b),auto_broadcast='numpy',_shape=y.shape,_dtype='f16',_name='output')
    if physical!=batch:
        args=[symbolic.constant(np.array(v,dtype=np.int64)) for v in ([0,0],[batch,n],[1,1])]
        y=symbolic.op('StridedSlice',y,*args,begin_mask=[0,0],end_mask=[0,0],new_axis_mask=[0,0],
                      shrink_axis_mask=[0,0],ellipsis_mask=[0,0],_shape=(batch,n),_dtype='f16',_name='trim')
    graph = symbolic.Graph([x],[y],name='lowbit_linear')
    ir = symbolic.serialize_ir(graph)
    xml = ET.fromstring(ir.xml)
    for name in decodes:
        info = ET.SubElement(xml.find(f"./layers/layer[@name='{name}']"),'rt_info')
        ET.SubElement(info,'attribute',name='decompression',version='0')
    return graph, ET.tostring(xml), ir.weights


def compile_weight(weight, batch, native, *, bias=None, reference_fp16=False, compact_groups=False, pad_batch=False, activation_scale=None):
    graph, xml, data = weight_graph(weight,batch,bias=bias,reference_fp16=reference_fp16,compact_groups=compact_groups,pad_batch=pad_batch,activation_scale=activation_scale)
    result = native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
    if (result.vendor_id,result.device_id,tuple(result.compiler_version)) != (0x8086,0x643e,(8,3)):
        raise ValueError('Unsupported lowbit driver/device profile')
    elf = Elf(result.graph_blob)
    if not any(s['size'] and s['name'].startswith('task.dpu.') for s in elf.sections):
        raise ValueError('Lowbit matrix graph has no DPU evidence')
    modes=dpu_modes(result.graph_blob)
    matrix_modes=[v for v in modes if not v['elementwise']]
    if activation_scale is not None and (not matrix_modes or any(v['amode'] not in (1,2) or v['wmode'] not in (1,2) for v in matrix_modes)):
        raise ValueError('Compiler did not produce an 8-bit activation/weight DPU matrix graph')
    program = symbolic.load_native(result.graph_blob,graph=graph,libraries=native)
    return program, dict(xml_sha256=hashlib.sha256(xml).hexdigest(),ir_weight_bytes=len(data),
        blob_bytes=len(result.graph_blob),blob_sha256=hashlib.sha256(result.graph_blob).hexdigest(),
        constant_section_bytes=sum(s['size'] for s in elf.sections if s['name'].startswith('buffer.Constant.')),
        shave_invocation_sections=[s['name'] for s in elf.sections if s['size'] and s['name'].startswith('task.shave.invocation.')],
        dpu_sections=[s['name'] for s in elf.sections if s['size'] and s['name'].startswith('task.dpu.')],dpu_modes=modes)


class QuantizedLinear(torch.nn.Module):
    """Frozen inference with FP16 IO; activation_scale opts INT8 into W8A8."""
    def __init__(self, weight, *, bias=None, compact_groups=False, activation_scale=None):
        super().__init__()
        if not isinstance(weight,PackedWeight):
            raise ValueError('Requires a PackedWeight')
        if type(compact_groups) is not bool or (compact_groups and weight.shape[1]//weight.group_size>32):
            raise ValueError('compact_groups requires bool and at most 32 groups')
        self._weight = weight
        self._compact_groups=compact_groups
        self._activation_scale=_activation_scale(weight,activation_scale,compact_groups)
        if bias is not None:
            if (not isinstance(bias,torch.Tensor) or bias.device.type != 'cpu' or bias.shape != (weight.shape[0],)
                or bias.dtype not in (torch.float16,torch.float32) or not torch.isfinite(bias).all()):
                raise ValueError('Bias preparation requires N finite CPU FP16/FP32 values')
            converted=bias.detach().half()
            if not torch.isfinite(converted).all():raise ValueError('Bias overflows FP16')
            self._bias=converted.contiguous().numpy().tobytes()
        else:self._bias=None
        self._keys = {}

    @property
    def weight(self):return self._weight

    @property
    def bias(self):return self._bias

    @property
    def compact_groups(self):return self._compact_groups

    @property
    def activation_scale(self):return self._activation_scale

    @classmethod
    def from_float(cls, module, *, format='int4', group_size=None, compact_groups=False, activation_scale=None):
        if not isinstance(module,torch.nn.Linear) or module.weight.device.type != 'cpu':
            raise ValueError('Requires a CPU nn.Linear for preparation')
        return cls(quantize_weight(module.weight,format=format,group_size=group_size),bias=module.bias,compact_groups=compact_groups,activation_scale=activation_scale)

    def forward(self,x):
        import torch_intel_npu as npu
        with npu._lock:
            npu.synchronize(); npu._validate(x,dense=True)
            if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != self.weight.shape[1]:
                raise ValueError('QuantizedLinear requires 2D FP16 activations with matching K')
            batch=x.shape[0]
            # Driver 8.3 miscompiles some grouped signed INT4 small-M graphs.
            # Compile padding and trim in one graph. Compact per-group MatMuls
            # use single-axis scales and avoid this grouped compiler route.
            pad_batch=not self.compact_groups and self.weight.format=='int4' and self.weight.group_size<self.weight.shape[1]
            physical=max(batch,32) if pad_batch else batch
            if max(physical*x.shape[1],physical*self.weight.shape[0])>MAX_TENSOR_ELEMENTS:
                raise ValueError('Lowbit physical activation/output exceeds storage bound')
            value = npu._materialize(x)
            if batch not in self._keys:
                _,xml,data = weight_graph(self.weight,batch,bias=self.bias,compact_groups=self.compact_groups,pad_batch=pad_batch,activation_scale=self.activation_scale)
                self._keys[batch] = ('lowbit_linear',hashlib.sha256(xml+data).digest())
            key = self._keys[batch]
            factory=lambda:compile_weight(self.weight,batch,npu._native,bias=self.bias,compact_groups=self.compact_groups,pad_batch=pad_batch,activation_scale=self.activation_scale)[0]
            npu._register_graph_factory(key,factory)
            return npu._run(key,[value],(batch,self.weight.shape[0]),torch.float16)
