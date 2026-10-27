"""Read Windows DXCore counters for Intel 8086:643e, never the Arc GPU."""
import ctypes as C
import json
import os
import time
import uuid

U32, U64, PTR = C.c_uint32, C.c_uint64, C.c_void_p


class EngineInput(C.Structure):
    _fields_ = [('physical', U32), ('engine', U32), ('pid', U32)]


class EngineOutput(C.Structure):
    _fields_ = [('running_us', U64), ('process_succeeded', C.c_bool)]


class MemoryInput(C.Structure):
    _fields_ = [('physical', U32), ('kind', U32), ('pid', U32)]


class MemoryOutput(C.Structure):
    _fields_ = [('committed', U64), ('resident', U64), ('process_succeeded', C.c_bool)]


def guid(value):
    return (C.c_ubyte*16).from_buffer_copy(uuid.UUID(value).bytes_le)


def method(obj, slot, result, *arguments):
    table = C.cast(obj, C.POINTER(C.POINTER(PTR))).contents
    return C.WINFUNCTYPE(result, PTR, *arguments)(table[slot])


def checked(status):
    if status < 0:
        raise OSError(f'DXCore HRESULT 0x{status & 0xffffffff:08x}')


def release(obj):
    if obj.value:
        method(obj, 2, U32)(obj)
        obj.value = None


class NpuCounters:
    """Cumulative scheduled engine time, not SHAVE/DPU arithmetic occupancy."""
    def __init__(self):
        self.adapter = PTR()
        factory, adapters = PTR(), PTR()
        self.dll = C.WinDLL('dxcore.dll')
        create = self.dll.DXCoreCreateAdapterFactory
        create.argtypes, create.restype = [PTR, C.POINTER(PTR)], C.c_int32
        try:
            checked(create(guid('78ee5945-c36e-4b13-a669-005dd11c0f06'), C.byref(factory)))
            attribute = guid('b71b0d41-1088-422f-a27c-0250b7d3a988')
            checked(method(factory, 3, C.c_int32, U32, PTR, PTR, C.POINTER(PTR))(
                factory, 1, attribute, guid('526c7776-40e9-459b-b711-f32ad76dfc28'), C.byref(adapters)))
            inventory = []
            for index in range(method(adapters, 4, U32)(adapters)):
                candidate = PTR()
                try:
                    checked(method(adapters, 3, C.c_int32, U32, PTR, C.POINTER(PTR))(
                        adapters, index, guid('a0783366-cfa3-43be-9d79-55b2da97c63c'), C.byref(candidate)))
                    hardware = self.property(candidate, 3, U32*4)
                    description = self.property(candidate, 2, C.c_char*512).value.decode('utf-8')
                    luid = self.property(candidate, 0, U32*2)
                    row = dict(name=description, vendor=int(hardware[0]), device=int(hardware[1]),
                               luid=f'{luid[1]:08x}:{luid[0]:08x}')
                    inventory.append(row)
                    if tuple(hardware[:2]) == (0x8086, 0x643e):
                        if self.adapter.value:
                            raise RuntimeError('More than one matching NPU')
                        self.adapter, candidate = candidate, PTR()
                        self.identity = row
                finally:
                    release(candidate)
            self.inventory = inventory
            if not self.adapter.value:
                raise RuntimeError(f'8086:643e NPU absent: {inventory}')
            physical = self.property(self.adapter, 15, U32).value
            self.engines = []
            for index in range(physical):
                node, count = U32(index), U32()
                checked(method(self.adapter, 13, C.c_int32, U32, C.c_size_t, PTR, C.c_size_t, PTR)(
                    self.adapter, 16, C.sizeof(node), C.byref(node), C.sizeof(count), C.byref(count)))
                self.engines.extend((index, engine) for engine in range(count.value))
            self.supported = {state: bool(method(self.adapter, 8, C.c_bool, U32)(self.adapter, state))
                              for state in (2, 3, 4, 5, 6, 9)}
        except BaseException:
            self.close()
            raise
        finally:
            release(adapters)
            release(factory)

    @staticmethod
    def property(adapter, prop, kind):
        value = kind()
        checked(method(adapter, 6, C.c_int32, U32, C.c_size_t, PTR)(
            adapter, prop, C.sizeof(value), C.byref(value)))
        return value

    def read(self):
        result = dict(time=time.perf_counter(), engines={})
        for physical, engine in self.engines:
            values = {}
            for state, name in ((4, 'all_us'), (5, 'process_us')):
                if self.supported[state]:
                    argument = EngineInput(physical, engine, os.getpid())
                    output = EngineOutput()
                    checked(method(self.adapter, 9, C.c_int32, U32, C.c_size_t, PTR, C.c_size_t, PTR)(
                        self.adapter, state, C.sizeof(argument), C.byref(argument), C.sizeof(output), C.byref(output)))
                    values[name] = (None if state == 5 and not output.process_succeeded else output.running_us)
            result['engines'][f'{physical}:{engine}'] = values
        return result

    def memory(self):
        result = {}
        for state, scope in ((2, 'adapter'), (3, 'process')):
            if not self.supported[state]:
                continue
            for kind, label in ((0, 'dedicated'), (1, 'shared')):
                argument, output = MemoryInput(0, kind, os.getpid()), MemoryOutput()
                # Adapter-wide query excludes pid and the output success flag.
                checked(method(self.adapter, 9, C.c_int32, U32, C.c_size_t, PTR, C.c_size_t, PTR)(
                    self.adapter, state, 12 if state == 3 else 8, C.byref(argument),
                    24 if state == 3 else 16, C.byref(output)))
                result[f'{scope}_{label}'] = (dict(committed=output.committed, resident=output.resident)
                    if state == 2 or output.process_succeeded else None)
        return result

    def close(self):
        release(self.adapter)


if __name__ == '__main__':
    counter = NpuCounters()
    try:
        print(json.dumps(dict(inventory=counter.inventory, npu=counter.identity,
                              engines=counter.engines, supported=counter.supported,
                              sample=counter.read(), memory=counter.memory()), indent=2))
    finally:
        counter.close()
