"""Isolated allocator traces and paired complete-generation cache controls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time
import types

import numpy as np
from decode_chain_api import provenance, delta, equal
from shave_fp32_pipeline import final
from w8_quality_probe import ROOT, MODEL, sha

OUT=ROOT/'npunlock-probe/research/results/generation-cache'
STAGES=('embedding','upload','decoder','download','norm','head','greedy','bookkeeping')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def build():
    before=OUT/'before';before.mkdir(parents=True,exist_ok=True)
    names=('torch_intel_npu.py','torch_npu_smollm.py','build_torch_npu_device.ps1',
        'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/build.json',
        'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    assert not (OUT/'build.json').exists(),'Keep this evidence; use a new output directory to rebuild'
    for name in names:
        destination=before/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    sources=provenance();sources.update({name:sha(ROOT/name) for name in names});write(before/'sources.json',sources)
    source=(before/'npunlock-probe/torch-device/backend.cpp').read_text()
    source=source.replace('constexpr size_t max_cached_buffers = 128;', '''constexpr size_t max_cached_buffers = 256;
size_t diagnostic_capacity = 128;
bool diagnostic_enabled = false;
struct DiagnosticRow { size_t bytes = 0; int64_t creates = 0, hits = 0, evictions = 0, count_limit = 0, byte_limit = 0; };
std::array<DiagnosticRow, 256> diagnostic_rows{};
size_t diagnostic_used = 0;
int64_t diagnostic_overflow = 0;
void diagnostic_event(size_t bytes, int kind, bool count = false, bool budget = false) {
  if (!diagnostic_enabled) return;
  size_t i = 0;
  for (; i < diagnostic_used && diagnostic_rows[i].bytes != bytes; ++i) {}
  if (i == diagnostic_used) {
    if (i == diagnostic_rows.size()) { diagnostic_overflow++; return; }
    diagnostic_rows[i].bytes = bytes; diagnostic_used++;
  }
  auto& row = diagnostic_rows[i];
  if (kind == 0) row.creates++;
  else if (kind == 1) row.hits++;
  else { row.evictions++; row.count_limit += count; row.byte_limit += budget; }
}''')
    assert source.count('size_t diagnostic_capacity = 128;')==1
    source=source.replace('s.cached.size() >= max_cached_buffers','s.cached.size() >= diagnostic_capacity')
    marker='      auto* oldest = s.cached.front();';assert source.count(marker)==1
    source=source.replace(marker,marker+'''
      diagnostic_event(oldest->size, 2, s.cached.size() >= diagnostic_capacity,
          buffer->size > static_cast<size_t>(max_cached_buffer_bytes - s.cached_bytes));''')
    marker='        s.allocation_reuses++;';assert source.count(marker)==1
    source=source.replace(marker,marker+'\n        diagnostic_event(bytes, 1);')
    marker='      s.driver_allocations++;';assert source.count(marker)==1
    source=source.replace(marker,marker+'\n      diagnostic_event(bytes, 0);')
    marker='  m.attr("max_tensor_elements") = max_tensor_elements;';assert source.count(marker)==1
    source=source.replace(marker,marker+'''
  m.def("_diagnostic_capacity", [](int64_t blocks) {
    std::lock_guard<std::recursive_mutex> lock(state().mutex);
    if (blocks) {
      ready(); TORCH_CHECK(blocks == 128 || blocks == 192 || blocks == 256, "Unsupported diagnostic capacity");
      empty_cache(); diagnostic_capacity = static_cast<size_t>(blocks);
    }
    return diagnostic_capacity;
  }, pybind11::arg("blocks") = 0, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("_diagnostic_enable", [](bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(state().mutex);
    diagnostic_enabled = enabled; diagnostic_rows = {}; diagnostic_used = 0; diagnostic_overflow = 0;
  }, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("_diagnostic_read", []() {
    std::array<DiagnosticRow, 256> rows; size_t used; int64_t overflow;
    {
      pybind11::gil_scoped_release release;
      std::lock_guard<std::recursive_mutex> lock(state().mutex);
      rows = diagnostic_rows; used = diagnostic_used; overflow = diagnostic_overflow;
      diagnostic_rows = {}; diagnostic_used = 0; diagnostic_overflow = 0;
    }
    pybind11::list values;
    for (size_t i = 0; i < used; ++i) {
      const auto& r = rows[i]; pybind11::dict d;
      d["bytes"] = r.bytes; d["creates"] = r.creates; d["hits"] = r.hits;
      d["evictions"] = r.evictions; d["count_limit"] = r.count_limit; d["byte_limit"] = r.byte_limit;
      values.append(d);
    }
    pybind11::dict result; result["rows"] = values; result["overflow"] = overflow; return result;
  });''')
    directory=OUT/'diagnostic';directory.mkdir(exist_ok=True);(directory/'backend.cpp').write_text(source,encoding='utf-8')
    script=(before/'build_torch_npu_device.ps1').read_text(encoding='utf-8-sig')
    script=script.replace('$root = $PSScriptRoot',"$root = '"+str(ROOT).replace("'","''")+"'")
    marker="$output = Join-Path $root 'npunlock-probe\\torch-device'";assert script.count(marker)==1
    script=script.replace(marker,"$output = '"+str(directory).replace("'","''")+"'")
    path=directory/'build.ps1';path.write_text(script,encoding='utf-8')
    with (directory/'build.log').open('w',encoding='utf-8') as log:
        subprocess.run(['pwsh','-NoProfile','-File',str(path),'-Toolchain',str(ROOT/'npunlock-probe/toolchain/msvc')],
            stdout=log,stderr=log,timeout=180,check=True)
    info=read(directory/'build.json');assert info['source_sha256']==sha(directory/'backend.cpp')
    assert info['extension_sha256']==sha(directory/'_torch_intel_npu.pyd')
    write(OUT/'build.json',dict(status='built',builder_sha256=sha(Path(__file__)),diagnostic=info,sources=sources,
        policy='128/192/256 idle blocks; unchanged 16MiB budget; fixed-size diagnostic counters allocate no memory in deleters'))
    print('Isolated diagnostic extension built; production unchanged',flush=True)


def bind():
    directory=OUT/'diagnostic';info=read(OUT/'build.json')
    assert info['diagnostic']['source_sha256']==sha(directory/'backend.cpp')
    source=(OUT/'before/torch_intel_npu.py').read_text()
    marker='_EXT = _ROOT / "npunlock-probe/torch-device/_torch_intel_npu.pyd"';assert source.count(marker)==1
    source=source.replace(marker,f'_EXT = Path({str(directory/"_torch_intel_npu.pyd")!r})')
    source=source.replace('"allocation_cache_limit_blocks": 128','"allocation_cache_limit_blocks": _C._diagnostic_capacity()')
    module=types.ModuleType('torch_intel_npu');module.__file__=str(ROOT/'torch_intel_npu.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def execute(case,mode):
    import torch
    npu=bind()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    from torch_npu_smollm_stack_demo import errors
    torch.set_num_threads(8)
    report=dict(status='running',case=case,mode=mode,sources=provenance(),build=read(OUT/'build.json'),
        source_sha256=sha(Path(__file__)),samples=[],trace=[],pairs=[],decoder_layers=30,new_tokens=32)
    saved={}
    def work():
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),dict(role='user',content='Tell a story about a bird.')],tokenize=False,add_generation_prompt=True)
        ids=tokenizer(text,return_tensors='pt').input_ids;assert ids.shape==(1,24)
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode=mode)
        def run(capacity,trace=False,*,cold=False):
            assert npu._C._diagnostic_capacity()==capacity
            npu._C._diagnostic_enable(trace)
            generated=ids.clone();state=None;logits_rows=[];steps=[];events=[]
            before=npu.stats();start=time.perf_counter_ns()
            def record(step,phase):
                if trace:
                    event=npu._C._diagnostic_read();assert event['overflow']==0
                    stats=npu.stats();events.append(dict(step=step,phase=phase,events=event['rows'],
                        buffers={k:stats[k] for k in ('driver_allocations','cached_buffer_blocks','cached_buffer_bytes','live_buffers','live_buffer_bytes')}))
            for step in range(32):
                marks=[time.perf_counter_ns()]
                host=cpu.model.embed_tokens(ids if step==0 else generated[:,-1:]);marks.append(time.perf_counter_ns())
                x=host.to('intel_npu');marks.append(time.perf_counter_ns());record(step,'upload')
                hidden,state=stack.prefill(x) if state is None else stack.decode(x,state)
                marks.append(time.perf_counter_ns());record(step,'decoder')
                hidden=hidden.cpu();marks.append(time.perf_counter_ns());record(step,'download')
                normalized=cpu.model.norm(hidden);marks.append(time.perf_counter_ns())
                logits=cpu.lm_head(normalized)[:,-1].float();marks.append(time.perf_counter_ns())
                token=logits.argmax(-1,keepdim=True);marks.append(time.perf_counter_ns())
                generated=torch.cat((generated,token),1);logits_rows.append(logits)
                marks.append(time.perf_counter_ns());steps.append(dict(zip(STAGES,[(b-a)/1e6 for a,b in zip(marks,marks[1:])])))
            elapsed=(time.perf_counter_ns()-start)/1e6;change=delta(before,npu.stats())
            assert change['npu_calls']>=960 and change['chain_graphs']==930 and change['chain_groups']==248
            assert change['host_download_copies']==32
            assert change['host_upload_copies']>=32 if cold else change['host_upload_copies']==32
            assert all(c.length==55 for c in state) and all(torch.isfinite(v).all() for v in logits_rows)
            caches=[v.cpu().numpy().copy() for cache in state for v in (cache.keys,cache.values,cache._position)] if trace else []
            record(32,'snapshot')
            return dict(capacity=capacity,elapsed_ms=elapsed,first_token_ms=sum(steps[0].values()),steps=steps,delta=change,events=events),logits_rows,generated,caches
        print(case,mode,'preparing current 30-layer model',flush=True)
        anchor_row,anchor,tokens,_=run(128,cold=True)
        report['input_tokens']=24;report['first_logits_vs_cpu']=errors(anchor[0],cpu(ids,use_cache=False).logits[:,-1].float())
        saved['tokens']=tokens.numpy().copy();saved['logits']=np.stack([v.numpy() for v in anchor])
        report['graphs']=[dict(layer=i,operation=local[-1],sha256=hashlib.sha256(npu._sessions[key][1].graph_blob).hexdigest())
            for i,layer in enumerate(stack.layers) for local,key in layer._keys.items()]
        assert len(report['graphs'])==60
        def checked(capacity,trace=False):
            row,actual,generated,caches=run(capacity,trace);assert row['delta']['session_creations']==0 and row['delta']['npu_calls']==960
            assert torch.equal(tokens,generated);equal([v.numpy() for v in anchor],[v.numpy() for v in actual])
            row['all_32_logits_and_tokens_bit_equal']=True
            if trace:
                event=npu._C._diagnostic_read();assert event['overflow']==0;row['cleanup_events']=event['rows']
                for i,array in enumerate(caches):saved[f'cache-{capacity}/{i}']=array
            npu._C._diagnostic_enable(False)
            stats=npu.stats();assert stats['cached_buffer_blocks']<=capacity and stats['cached_buffer_bytes']<=16*1024*1024
            row['after_cleanup']={k:stats[k] for k in ('cached_buffer_blocks','cached_buffer_bytes','live_buffers','live_buffer_bytes')}
            return row
        def select(capacity):
            npu._C._diagnostic_enable(False);npu._C._diagnostic_capacity(capacity)
            for _ in range(2):checked(capacity)
        if case=='trace':
            for capacity in (128,192,256):
                select(capacity);row=checked(capacity,True);report['trace'].append(row)
                print(capacity,'creates',row['delta']['driver_allocations'],'cached',row['after_cleanup'],flush=True)
            for i in range(90):equal([saved[f'cache-{128}/{i}']],[saved[f'cache-{192}/{i}']]);equal([saved[f'cache-{128}/{i}']],[saved[f'cache-{256}/{i}']])
        else:
            current=None
            for index in range(7):
                pair={128:[],256:[]}
                for capacity in ((128,256,256,128) if index%2==0 else (256,128,128,256)):
                    if capacity!=current:select(capacity);current=capacity
                    row=checked(capacity);report['samples'].append(row);pair[capacity].append(row['elapsed_ms'])
                report['pairs'].append({str(k):st.mean(v) for k,v in pair.items()})
                print(mode,'round',index+1,report['pairs'][-1],flush=True)
            report['summary']={str(capacity):dict(elapsed_ms=st.median(r['elapsed_ms'] for r in report['samples'] if r['capacity']==capacity),
                first_token_ms=st.median(r['first_token_ms'] for r in report['samples'] if r['capacity']==capacity),
                allocations=sorted(set(r['delta']['driver_allocations'] for r in report['samples'] if r['capacity']==capacity))) for capacity in (128,256)}
            report['positive_rounds']=sum(p['256']<p['128'] for p in report['pairs'])
            print(mode,report['summary'],report['positive_rounds'],'/7 faster',flush=True)
    with torch.no_grad():work();final(npu,report)
    path=OUT/f'{case}-{mode}-outputs.npz';np.savez_compressed(path,**saved)
    report.update(status='passed',outputs_sha256=sha(path));write(OUT/f'{case}-{mode}.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','trace','benchmark'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.case=='build':build()
    else:execute(args.case,args.mode)
