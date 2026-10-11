import ctypes as c
from pathlib import Path
import torch

torch.utils.rename_privateuse1_backend('intel_npu')
lib = c.PyDLL(str(Path(torch.__file__).parent / 'lib/torch_cpu.dll'))
ptr = c.c_void_p
i64p = c.POINTER(c.c_int64)
deleter_type = c.CFUNCTYPE(None, ptr, ptr)
@deleter_type
def release(data, ctx):
    print('deleted', data, flush=True)

blob = lib.torch_from_blob
blob.argtypes = [ptr, c.c_int64, i64p, i64p, c.c_int64, c.c_int32,
    c.c_int32, c.c_int32, c.POINTER(ptr), c.c_int32, ptr, c.c_int64,
    deleter_type, ptr]
blob.restype = c.c_int32
wrap = lib.torch_tensor_to_pyobject
wrap.argtypes = [ptr, ptr, c.POINTER(ptr)]
wrap.restype = c.c_int32
delete = lib.aoti_torch_delete_tensor_object
delete.argtypes = [ptr]
delete.restype = c.c_int32
data = (c.c_float * 2)(1, 2)
size = (c.c_int64 * 1)(2)
stride = (c.c_int64 * 1)(1)
handle = ptr()
status = blob(c.cast(data, ptr), 1, size, stride, 0,
    lib.aoti_torch_dtype_float32(), lib.aoti_torch_device_type_privateuse1(),
    0, c.byref(handle), lib.aoti_torch_layout_strided(), None, 0, release, None)
print('blob', status, handle.value, flush=True)
if status:
    error = getattr(lib, '?get_last_error@aot_inductor@torch@@YAPEBDXZ')
    error.restype = c.c_char_p
    print(error().decode(), flush=True)
    raise SystemExit(status)
obj = ptr()
print('wrap', wrap(handle, None, c.byref(obj)), flush=True)
tensor = c.cast(obj, c.py_object).value
decref = c.pythonapi.Py_DecRef
decref.argtypes = [ptr]
decref(obj)
delete(handle)
print(type(tensor), tensor.device, tensor.untyped_storage().device,
      torch._C._dispatch_keys(tensor), tensor.data_ptr(), flush=True)
del tensor
print('done', flush=True)
