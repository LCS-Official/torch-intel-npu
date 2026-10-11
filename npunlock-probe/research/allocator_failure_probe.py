"""Synthetic uncertain completion: verify cached blocks obey quarantine too."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import gc

import torch_npu_warm_profile as common
ROOT=common.ROOT
OUT=common.OUT


def build():
    source=(OUT/'session_profile.c').read_text(encoding='utf-8')
    signature='npunlock_status graphinfer_session_infer(graphinfer_session *session,'
    assert source.count(signature)==1
    source=source.replace(signature,'static npunlock_status warm_original_infer(graphinfer_session *session,')
    source+=r'''
static volatile LONG mock_fail;
__declspec(dllexport) void warm_diag_fail_next(void) { InterlockedExchange(&mock_fail,1); }
npunlock_status graphinfer_session_infer(graphinfer_session *session,
    const graphinfer_shared_tensor *inputs, size_t input_count,
    const graphinfer_shared_tensor *outputs, size_t output_count,
    graphinfer_session_infer_result *result) {
  if (InterlockedExchange(&mock_fail,0)) {
    memset(result,0,sizeof(*result)); result->struct_size=sizeof(*result);
    EnterCriticalSection(session_lock(session));
    session->context_owner->failed=true;
    LeaveCriticalSection(session_lock(session));
    return session_driver_error(&result->diagnostic,"zeFenceHostSynchronize(mock)",1u);
  }
  return warm_original_infer(session,inputs,input_count,outputs,output_count,result);
}
'''
    generated=OUT/'session_failure.c';generated.write_text(source,encoding='utf-8')
    previous=json.loads((OUT/'profile-build.json').read_text(encoding='utf-8'))
    args=[str(generated) if p==str(OUT/'session_profile.c') else str(OUT/'session_failure.dll')
          if p==str(OUT/'session_profile.dll') else p for p in previous['arguments']]
    result=subprocess.run(args,env=dict(os.environ,PATH=str(Path(args[0]).parent)+';'+os.environ['PATH']),
                          capture_output=True,text=True)
    (OUT/'failure-build.log').write_text(result.stdout+result.stderr,encoding='utf-8');result.check_returncode()
    (OUT/'failure-build.json').write_text(json.dumps(dict(arguments=args,
        dll_sha256=common.sha(OUT/'session_failure.dll'),source_sha256=common.sha(generated),
        probe_sha256=common.sha(Path(__file__))),indent=2),encoding='utf-8')
    print('Synthetic failure DLL built; no hardware fault triggered')


def run():
    import torch
    dll=OUT/'session_failure.dll'
    manifest=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    manifest['session_sha256']=common.sha(dll)
    path=OUT/'failure-bound-build.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    common.bind(dll,path)
    import torch_intel_npu as npu
    driver=ct.CDLL(str(dll))
    driver.warm_diag_reset.argtypes=[];driver.warm_diag_reset.restype=None
    driver.warm_diag_fail_next.argtypes=[];driver.warm_diag_fail_next.restype=None
    driver.warm_diag_read.argtypes=[ct.POINTER(ct.c_uint64)]*2;driver.warm_diag_read.restype=None
    with torch.no_grad():
        x=torch.ones(64).to('intel_npu');y=torch.ones(64).to('intel_npu')
        spare=torch.empty(4096,dtype=torch.float16,device='intel_npu');del spare
        before=npu.stats();assert before['cached_buffer_bytes']==8192
        driver.warm_diag_reset();driver.warm_diag_fail_next()
        try:x+y
        except RuntimeError as error:assert 'zeFenceHostSynchronize(mock)' in str(error)
        else:raise AssertionError('Missing synthetic failure')
        failed=npu.stats();assert failed['context_failed'] and failed['npu_calls']==before['npu_calls']
        assert failed['quarantined_buffer_bytes']>=before['live_buffer_bytes']+before['cached_buffer_bytes']
        try:npu.synchronize()
        except RuntimeError:pass
        else:raise AssertionError('Failed context still accessible')
        npu.empty_cache();del x,y;gc.collect();npu.shutdown();final=npu.stats()
        ns=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(ns,counts)
        assert not counts[1] and not counts[3] and not counts[5] and not counts[7] and not counts[8]
        assert final['live_buffers']==final['live_buffer_bytes']==final['cached_buffer_bytes']==final['cached_buffer_blocks']==0
    report=dict(status='passed',synthetic_failure=True,actual_device_submission=False,before=before,failed=failed,
                final_stats=final,driver_free_calls=int(counts[1]),driver_reset_calls=int(counts[7]+counts[8]),
                device_build=manifest,probe_sha256=common.sha(Path(__file__)))
    (OUT/'allocator-failure.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Synthetic failure quarantine includes cached bytes; no driver free/reset: passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','run'),required=True)
    args=parser.parse_args();(build if args.case=='build' else run)()
