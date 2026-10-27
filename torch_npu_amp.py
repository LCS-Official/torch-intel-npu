"""NPU loss scaling: device gradients, CPU scale/growth control metadata."""
from math import isfinite
from collections.abc import Iterable
import torch
from torch.amp.grad_scaler import OptState

_TorchGradScaler = torch.amp.GradScaler
_original_new = _TorchGradScaler.__new__


def _validate_scale(value):
    if type(value) not in (int,float) or not isfinite(value) or value<=0:
        raise ValueError('NPU AMP scale must be a positive finite FP32 value')
    rounded=torch.tensor(value,dtype=torch.float32,device='cpu')
    if not 0<rounded.item()<=torch.finfo(torch.float32).max:
        raise ValueError('NPU AMP scale is outside positive FP32 storage')


def _validate_settings(scale,growth,backoff,interval):
    _validate_scale(scale)
    if (type(growth) not in (int,float) or not isfinite(growth) or not 1<growth<=torch.finfo(torch.float32).max or
            type(backoff) not in (int,float) or not isfinite(backoff) or not 0<backoff<1 or
            type(interval) is not int or not 1<=interval<(1<<31)):
        raise ValueError('NPU AMP requires finite growth>1, 0<backoff<1 and a positive INT32 growth interval')


class GradScaler(_TorchGradScaler):
    """FP32-master loss scaling with CPU control metadata and NPU gradients."""
    def __init__(self,init_scale=2.**16,growth_factor=2.,backoff_factor=.5,growth_interval=2000,enabled=True):
        if enabled:_validate_settings(init_scale,growth_factor,backoff_factor,growth_interval)
        super().__init__('intel_npu',init_scale,growth_factor,backoff_factor,growth_interval,enabled)

    def _lazy_init_scale_growth_tracker(self,dev):
        # These scalars control the algorithm; they are never model tensors.
        super()._lazy_init_scale_growth_tracker(torch.device('cpu'))

    def scale(self,outputs):
        if not self.is_enabled():return outputs
        def apply(value):
            if isinstance(value,torch.Tensor):
                import torch_intel_npu as npu
                with npu._lock,torch.no_grad():npu._validate_float_read(value)
                if self._scale is None:self._lazy_init_scale_growth_tracker(value.device)
                factor=self._scale.item();_validate_scale(factor)
                with torch.autocast('intel_npu',enabled=False):
                    result=value.float()*factor
                    # FP32 opmath preserves zero lanes even when scale exceeds the FP16 range.
                    return result.to(value.dtype) if value.ndim else result
            if isinstance(value,Iterable) and not isinstance(value,(str,bytes)):
                result=map(apply,value)
                return type(value)(result) if isinstance(value,(list,tuple)) else result
            raise ValueError('NPU GradScaler.scale expects a Tensor or iterable of Tensors')
        return apply(outputs)

    @staticmethod
    def _check_optimizer(optimizer):
        if getattr(optimizer,'_step_supports_amp_scaling',False):
            raise ValueError('NPU GradScaler requires an ordinary unfused optimizer')
        for group in optimizer.param_groups:
            if (group.get('foreach') is not False or group.get('fused') is not False or
                    group.get('capturable',False) or group.get('differentiable',False)):
                raise ValueError('NPU GradScaler requires foreach=False, fused=False and noncapturable/nondifferentiable updates')

    def unscale_(self,optimizer):
        if not self.is_enabled():return
        self._check_optimizer(optimizer)
        scale,_=self._check_scale_growth_tracker('unscale_')
        state=self._per_optimizer_states[id(optimizer)]
        if state['stage'] is not OptState.READY:
            raise RuntimeError('NPU unscale_ may run only once before step, between updates')
        # Preserve PyTorch's FP64 reciprocal of the CPU control scalar.
        inverse=scale.double().reciprocal().float().item();_validate_scale(inverse)
        import torch_intel_npu as npu
        with npu._lock,torch.no_grad(),torch.autocast('intel_npu',enabled=False):
            params=[p for group in optimizer.param_groups for p in group['params']]
            gradients=[];seen=set();parameter_storage={p.untyped_storage()._cdata for p in params}
            for param in params:
                npu._validate(param,dense=True)
                if param.dtype!=torch.float32:raise ValueError('NPU AMP requires FP32 master parameters')
                if param.grad is None:continue
                grad=param.grad;npu._validate(grad,dense=True)
                if grad.dtype!=torch.float32:raise ValueError('NPU AMP requires dense FP32 master gradients')
                storage=grad.untyped_storage()._cdata
                if storage in seen or storage in parameter_storage:
                    raise ValueError('NPU AMP rejects aliased gradient/parameter storage')
                seen.add(storage);gradients.append(grad)
            if not gradients:raise RuntimeError('No gradients for NPU AMP unscale_')
            # Preflight the whole optimizer before any gradient is changed.
            finite=None;results=[];limit=torch.finfo(torch.float32).max
            for grad in gradients:
                value=npu._materialize(grad)
                mask=npu._logical(npu._compare(value,-limit,kind='ge'),npu._compare(value,limit,kind='le'),kind='and')
                ok=npu._boolean_reduce(mask,kind='all')
                finite=ok if finite is None else npu._logical(finite,ok,kind='and')
                results.append(npu._binary_op(value,inverse,kind='mul'))
            found_inf=not finite.item()  # One device flag; no gradient payload download.
            for grad,result in zip(gradients,results):npu._copy(grad,result)
        state['found_inf_per_device']={torch.device('intel_npu:0'):torch.tensor(float(found_inf),dtype=torch.float32,device='cpu')}
        state['stage']=OptState.UNSCALED

    def step(self,optimizer,*args,**kwargs):
        if self.is_enabled():self._check_optimizer(optimizer)
        return super().step(optimizer,*args,**kwargs)

    def update(self,new_scale=None):
        if self.is_enabled():
            self._check_scale_growth_tracker('update')
            if new_scale is not None:
                if isinstance(new_scale,torch.Tensor):
                    if (new_scale.device.type!='intel_npu' or new_scale.device.index!=0 or new_scale.dtype!=torch.float32 or
                            new_scale.numel()!=1 or new_scale.requires_grad):
                        raise ValueError('NPU new_scale must be a detached one-element device FP32 Tensor')
                    new_scale=new_scale.item()
                _validate_scale(new_scale);new_scale=float(new_scale)
            _validate_settings(self.get_scale(),self.get_growth_factor(),self.get_backoff_factor(),self.get_growth_interval())
        return super().update(new_scale)

    def load_state_dict(self,state_dict):
        if self.is_enabled() and state_dict:
            _validate_settings(state_dict['scale'],state_dict['growth_factor'],state_dict['backoff_factor'],state_dict['growth_interval'])
            tracker=state_dict['_growth_tracker']
            if type(tracker) is not int or not 0<=tracker<state_dict['growth_interval']:
                raise ValueError('NPU AMP growth tracker is outside its interval')
        return super().load_state_dict(state_dict)


class _StandardGradScaler(GradScaler):
    def __init__(self,device='intel_npu',init_scale=2.**16,growth_factor=2.,backoff_factor=.5,growth_interval=2000,enabled=True):
        if device!='intel_npu':raise ValueError('Expected the intel_npu GradScaler backend')
        super().__init__(init_scale,growth_factor,backoff_factor,growth_interval,enabled)


def _scaler_new(cls,*args,**kwargs):
    if cls is _TorchGradScaler and (args[0] if args else kwargs.get('device','cuda'))=='intel_npu':
        return object.__new__(_StandardGradScaler)
    return object.__new__(cls) if _original_new is object.__new__ else _original_new(cls,*args,**kwargs)


def register_grad_scaler():
    """Preserve class identity and aliases imported before device registration."""
    if _TorchGradScaler.__new__ is _scaler_new:return
    if _TorchGradScaler.__new__ is not _original_new:
        raise RuntimeError('Another extension has changed GradScaler construction')
    # PyTorch has no custom-device scaler factory; scope the constructor adapter to this backend.
    _TorchGradScaler.__new__=staticmethod(_scaler_new)
