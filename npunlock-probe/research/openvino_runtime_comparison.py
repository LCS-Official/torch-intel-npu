"""Alternating runtime comparison using exactly the same compiled ELF blob.

Run in the isolated OpenVINO env with the local npunlock Python path and
NPUNLOCK_NATIVE_DIR pointing to its existing native libraries. No torch import.
"""
import ctypes as ct
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov
from npunlock._native import _InferSessionRunResult
from dpu_diagnostic import OUT, ROOT, STAGES, native_setup, metadata


def execute():
    core = ov.Core()
    native = native_setup(); dll = native.infer
    dll.diag_replay_create.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,
                                      ct.POINTER(ct.c_void_p),ct.POINTER(_InferSessionRunResult)]
    dll.diag_replay_run.argtypes = [ct.c_void_p,ct.c_uint32,ct.POINTER(_InferSessionRunResult)]
    dll.diag_replay_destroy.argtypes = [ct.c_void_p]
    for name in ('diag_replay_create','diag_replay_run','diag_replay_destroy'):
        getattr(dll,name).restype = ct.c_uint32
    report = dict(status='running',openvino_version=ov.__version__,rows=[],
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(__file__),ROOT/'npunlock-probe/research/dpu_diagnostic.py',
                   ROOT/'npunlock-probe/research/dpu_diagnostic_extra.c',OUT/'session_diagnostic.dll')},
                  scope='Same compiled blob per comparison; resident buffers; synchronous public OpenVINO '
                        'infer(share_outputs=True), native standard and native fixed-address replay; '
                        '5 warmups, 31 rotating rounds; no concurrent NPU execution')
    native_calls = ov_calls = 0
    sessions = []; buffers = []; handles = []
    result = _InferSessionRunResult(); result.struct_size = ct.sizeof(result)
    def checked(code):
        message = bytes(result.diagnostic.json.data[:result.diagnostic.json.size]).decode('utf-8',errors='replace') if code else ''
        dll.graphinfer_session_infer_result_release(ct.byref(result))
        assert code == 0,(code,message)
    try:
        for m,k,n in ((1,1024,1024),(128,1024,1024),(1024,1024,1024),(2048,512,256)):
            tag = '-'.join(map(str,(m,k,n)))
            with np.load(OUT/f'fixture-{tag}.npz',allow_pickle=False) as fixture:
                expected = fixture['expected'].copy()
                hosts = {'fp16-fq':fixture['fp16'].copy(),'prequant-u8':fixture['u8'].copy()}
            for path,host in hosts.items():
                xml = OUT/f'{tag}-{path}.xml'
                model = core.read_model(str(xml))
                for compiler in ('DRIVER','PLUGIN'):
                    config = {'PERFORMANCE_HINT':'LATENCY','EXECUTION_MODE_HINT':'ACCURACY','NPU_COMPILER_TYPE':compiler}
                    compiled = core.compile_model(model,'NPU',config)
                    request = compiled.create_infer_request()
                    ov_in = request.get_input_tensor().data
                    ov_out = request.get_output_tensor().data
                    blob = compiled.export_model()
                    if hasattr(blob,'getvalue'): blob = blob.getvalue()
                    session = native.create_inference_session(blob,timeout_ms=10000); sessions.append(session)
                    args = metadata(native,session)
                    assert args[0]['precision'] == (4 if path == 'prequant-u8' else 2)
                    assert args[0]['bytes'] == host.nbytes and args[1]['bytes'] == expected.nbytes
                    ib = session.create_buffer(host.nbytes); ob = session.create_buffer(expected.nbytes)
                    buffers.extend((ib,ob))
                    ni,ni_owners = session._binding_array([(args[0]['name'],ib)])
                    no,no_owners = session._binding_array([(args[1]['name'],ob)])
                    def standard():
                        nonlocal native_calls
                        checked(dll.graphinfer_session_infer(session._result.session,ni,1,no,1,ct.byref(result)))
                        native_calls += 1
                    np.copyto(ov_in,host); ct.memmove(ib.address,host.ctypes.data,host.nbytes)
                    standard()  # Initialize the native graph and queue before recording replay.
                    handle = ct.c_void_p()
                    checked(dll.diag_replay_create(session._result.session,ni,1,no,1,ct.byref(handle),ct.byref(result)))
                    handles.append(handle)
                    def replay():
                        nonlocal native_calls
                        checked(dll.diag_replay_run(handle,1,ct.byref(result))); native_calls += 1
                    def openvino():
                        nonlocal ov_calls
                        request.infer(share_outputs=True); ov_calls += 1
                    runs = {'native-standard':standard,'native-replay':replay,'openvino':openvino}
                    def validate(target):
                        for key,run in runs.items():
                            run()
                            actual = np.empty_like(expected)
                            if key == 'openvino': actual[:] = ov_out
                            else: ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
                            np.testing.assert_array_equal(actual,target)
                    np.copyto(ov_in,host); ct.memmove(ib.address,host.ctypes.data,host.nbytes)
                    validate(expected)
                    zeros = np.full_like(host,128 if path == 'prequant-u8' else 0)
                    np.copyto(ov_in,zeros); ct.memmove(ib.address,zeros.ctypes.data,zeros.nbytes)
                    validate(np.zeros_like(expected))
                    np.copyto(ov_in,host); ct.memmove(ib.address,host.ctypes.data,host.nbytes)
                    validate(expected)
                    samples = {key:[] for key in runs}; stages = {key:[] for key in runs if key != 'openvino'}
                    for run in runs.values():
                        for _ in range(5): run()
                    for round_index in range(31):
                        keys = tuple(runs)
                        order = keys[round_index%3:]+keys[:round_index%3]
                        if round_index%2: order = tuple(reversed(order))
                        for key in order:
                            start = time.perf_counter_ns(); runs[key]()
                            samples[key].append((time.perf_counter_ns()-start)/1e6)
                            if key != 'openvino':
                                times = (ct.c_uint64*10)()
                                assert dll.diag_times(session._result.session,times,10) == 0 and not times[9]
                                stages[key].append({name:times[i]/1e6 for i,name in enumerate(STAGES)})
                    validate(expected)
                    row = dict(shape_mkn=[m,k,n],path=path,compiler=compiler,config=config,
                               compiler_version=str(compiled.get_property('NPU_COMPILER_VERSION')),
                               actual_compiler=str(compiled.get_property('NPU_COMPILER_TYPE')),
                               execution_devices=str(compiled.get_property('EXECUTION_DEVICES')),
                               bit_exact_grid=True,changed_input_and_restore_passed=True,arguments=args,
                               shared_compiled_blob_sha256=hashlib.sha256(blob).hexdigest(),
                               ir_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                               weights_sha256=hashlib.sha256(xml.with_suffix('.bin').read_bytes()).hexdigest(),
                               timings={key:dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),
                                   effective_tops=2*m*k*n/(statistics.median(values)*1e9),samples_ms=values)
                                   for key,values in samples.items()},
                               stage_median_ms={key:{stage:statistics.median(v[stage] for v in values) for stage in STAGES}
                                                for key,values in stages.items()})
                    (OUT/f'paired-{tag}-{path}-{compiler.lower()}.blob').write_bytes(blob)
                    assert dll.diag_replay_destroy(handle) == 0; handles.pop()
                    ib.release(); ob.release(); session.close()
                    report['rows'].append(row)
                    print(json.dumps(dict(shape=[m,k,n],path=path,compiler=compiler,
                         ms={key:value['median_ms'] for key,value in row['timings'].items()},
                         tops={key:value['effective_tops'] for key,value in row['timings'].items()})),flush=True)
                    del request,compiled,ov_in,ov_out
                    gc.collect()
        report['status'] = 'passed'
    finally:
        for handle in handles: dll.diag_replay_destroy(handle)
        for buffer in reversed(buffers): buffer.release()
        for session in reversed(sessions): session.close()
        report.update(successful_native_graph_calls=native_calls,successful_openvino_graph_calls=ov_calls,
                      all_buffers_released=all(v._released for v in buffers),all_sessions_closed=all(v._closed for v in sessions))
        (OUT/'openvino-paired.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(execute())
