"""Audit multi-token fusion, unchanged decode IR and measured prefill gains."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-prefill-fusion'
FIX=ROOT/'npunlock/tests/fixtures/npu4000'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
baseline=json.loads((OUT.parent/'smollm-whole-layer/summary.json').read_text(encoding='utf-8'))
names=['probe','numeric','boundary','guards','hybrid','paired','regression','positions','profile','performance']
cases={name:read(name) for name in names}
extras={name:read(name) for name in ['model-probe-fp16','model-probe-w8a16','ownership','control']}
sources=cases['numeric']['sources'];manifest=cases['numeric']['kernel_manifest']
assert digest(OUT/'candidate.py')==sources['torch_npu_smollm.py']
for name,case in cases.items():
    assert case['sources']==sources and case['kernel_manifest']==manifest and case['checkpoint']==baseline['checkpoint'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
    assert case['final_stats']['hardware_device']=='8086:643e' and case['final_stats']['driver_version']==1005540,name
    for source,value in case['sources'].items():assert digest(ROOT/source)==value,(name,source)
for name,value in sources.items():
    if name not in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:assert value==baseline['sources'][name],name
for name,path in [('torch_npu_smollm.py','torch_npu_smollm.py'),('torch_npu_smollm_kv_demo.py','torch_npu_smollm_kv_demo.py'),('npu4000.py','npunlock/python/npunlock/npu4000.py')]:
    assert digest(OUT/'before'/name)==baseline['sources'][path],name
assert manifest==baseline['kernel_manifest']==json.loads((OUT/'before/manifest.json').read_text())
for name,value in manifest['source_sha256'].items():assert digest(FIX/name)==value
assert digest(FIX/'rmsnorm.c')==digest(OUT/'before/rmsnorm.c')
assert digest(FIX/'rmsnorm/rmsnorm-f16.elf')==manifest['elf_sha256']==digest(OUT/'before/rmsnorm-f16.elf')
assert digest(FIX/'rmsnorm/rmsnorm-f16.s')==manifest['assembly_sha256']
assert digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')==baseline['checkpoint_file_sha256']

from npunlock.npu4000 import inspect,POWER_TEXT_SHA256
layouts=set();probes=[e for name in ['model-probe-fp16','model-probe-w8a16'] for e in extras[name]['checks']]
assert len(probes)==256
for name in ['model-probe-fp16','model-probe-w8a16']:
    assert len(extras[name]['checks'])==128 and not extras[name]['rejected_tags']
    assert extras[name]['candidate_sha256']==sources['torch_npu_smollm.py']
    assert extras[name]['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
for e in probes:
    assert e['status']=='passed' and all(e['bits_equal']) and e['graph_calls']==1 and e['warm_host_copies_and_compiles']==0
    patch=e['patch'];original=(OUT/(e['tag']+'-original.blob')).read_bytes();modified=(OUT/(e['tag']+'-patched.blob')).read_bytes()
    assert hashlib.sha256(original).hexdigest()==patch['original_sha256'] and hashlib.sha256(modified).hexdigest()==patch['patched_sha256']
    assert patch['kernel_elf_sha256']==manifest['elf_sha256'] and patch['accumulation']=='fp32_neumaier'
    start=patch['code_offset'];end=start+patch['code_bytes']
    assert original[:start]==modified[:start] and original[end:]==modified[end:] and len(original)==len(modified)
    assert patch['code_bytes']==5136 and patch['window_bytes']==21216 and patch['shape']==[e['batch']*e['length'],576]
    assert patch['carriers']==['rmsnorm_carrier','rmsnorm_ffn'] and all(c['elements']%576==0 for c in patch['contracts'])
    for task in inspect(original,memref_count=3,kernel_sha256=POWER_TEXT_SHA256)['invocations']:
        for ref in task['memrefs']:
            count=ref['elements']//576
            assert (ref['order'],ref['dims'],ref['strides_bits']) in [
                ('0x4321',(576,count,1,1),(16,9216,9216*count,9216*count)),
                ('0x2431',(1,576,count,1),(16,16,9216,9216*count))]
            layouts.add((ref['order'],count))
integrity=read('integrity');assert integrity['status']=='passed' and len(integrity['checks'])==56
for source,value in integrity['sources'].items():assert digest(ROOT/source)==value
assert all(e['single_token_ir_weights_equal'] for e in integrity['checks'][:48])
assert all(e['rejected'] for e in integrity['checks'][48:])
assert all(e['all_output_cache_bits_equal'] for e in cases['paired']['checks'])
assert all(e['old_new_bits_equal'] for e in cases['regression']['checks'])
assert all(e['generated_tokens_equal'] for e in cases['hybrid']['checks'])
entries=[e for name in ['probe','numeric','boundary','positions'] for e in cases[name]['checks'] if 'execution' in e]
cache_entries=[e for entry in entries for e in entry.get('cache_local_reference',[])]
assert all(e['execution']['relative_l2']<.003 and e['execution']['max_scaled_error']<.01 for e in entries)
assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in cache_entries)
assert all(e['quality_official']['relative_l2']<.03 for e in entries if 'quality_official' in e)
for e in cases['paired']['checks']:
    if e['warmed']:assert e['graph_calls']==e['previous_graph_calls']==e['num_layers']
for e in cases['positions']['checks']:
    if 'positions_checked' in e:assert e['graph_calls_per_decode']==e['num_layers'] and e['new_sessions']==0
assert len(extras['ownership']['checks'])==16
for e in extras['ownership']['checks']:
    if 'graph_calls' in e:
        assert e['graph_calls']==1 and e['warm_allocations']==3 and e['warm_host_copies_and_compiles']==0
        assert e['causal_batch_isolation'] and e['independent_output_kv'] and e['position_and_padding_correct']
        assert e['position_metadata_prewarmed'] and e['initial_position_metadata_uploads'] in [0,1]
    else:assert e['old_snapshots_unchanged'] and e['held_groups']==7 and e['changing_shape_cold_gamma_uploads_after_first']==0
control=extras['control'];assert len(control['checks'])==len(control['timings'])==14
assert control['candidate_sha256']==sources['torch_npu_smollm.py'] and control['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
assert all(e['all_bits_equal'] for e in control['checks'])
for e in control['timings']:
    assert e['rounds']==41 and e['same_model_input'] and e['operation']=='prefill' and e['warm_host_copies_and_compiles']==0
    assert all(len(v)==82 for v in e['samples_ms'].values()) and len(e['round_pairs_ms'])==41
    assert e['graph_calls']==dict(previous=e['num_layers']*4,current=e['num_layers'])
    assert e['allocations_per_prefill']==dict(previous=e['num_layers']*4,current=e['num_layers']*3)
    assert e['latency_reduction_percent']>0
for e in cases['profile']['timings']:assert sum(e['native_calls'].values())==e['num_layers']==e['native_calls']['norm_layer']
for e in cases['performance']['timings']:
    for name,calls in e['graph_calls'].items():
        if not name.startswith('cpu'):assert calls==e['num_layers']

summary=dict(status='passed',date='2026-10-05',sources=sources,kernel_manifest=manifest,packages=cases['numeric']['packages'],
    checkpoint=cases['numeric']['checkpoint'],checkpoint_file_sha256=baseline['checkpoint_file_sha256'],
    multi_token_graph_records=len(probes),unchanged_single_token_ir_records=48,new_graph_reference_guards=8,
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),
    max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    controls=control['timings'],performance=cases['performance']['timings'],profiles=cases['profile']['timings'],observed_row_layouts=sorted(layouts),
    total_npu_calls=sum(c['final_stats']['npu_calls'] for c in {**cases,**extras}.values()),
    cases={name:dict(checks=len(c.get('checks',[])),timings=len(c.get('timings',[])),npu_calls=c['final_stats']['npu_calls']) for name,c in {**cases,**extras}.items()})
changed={'torch_npu_smollm.py':ROOT/'torch_npu_smollm.py','torch_npu_smollm_kv_demo.py':ROOT/'torch_npu_smollm_kv_demo.py',
    'npu4000.py':ROOT/'npunlock/python/npunlock/npu4000.py','rmsnorm.c':FIX/'rmsnorm.c',
    'manifest.json':FIX/'rmsnorm/manifest.json','rmsnorm-f16.elf':FIX/'rmsnorm/rmsnorm-f16.elf',
    'TORCH_NPU_README.md':ROOT/'TORCH_NPU_README.md','NPU4000.md':ROOT/'npunlock/docs/NPU4000.md'}
(OUT/'after').mkdir(exist_ok=True);diff=[]
for name,path in changed.items():
    shutil.copyfile(path,OUT/'after'/name)
    if path.suffix!='.elf':diff.extend(difflib.unified_diff((OUT/'before'/name).read_text(encoding='utf-8').splitlines(True),path.read_text(encoding='utf-8').splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(OUT/'changes.diff').write_text(''.join(diff),encoding='utf-8')
summary['after_sha256']={name:digest(OUT/'after'/name) for name in changed};summary['diff_sha256']=digest(OUT/'changes.diff')
summary['audit_sha256']=digest(Path(__file__))
summary['evidence_sha256']={name:digest(OUT/(name+'.json')) for name in [*cases,*extras,'integrity']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
control_table='\n'.join(f"| {e['num_layers']} / {e['batch']} / {e['prefix_tokens']} | {e['mode']} | {e['median_ms']['previous']:.3f} | {e['median_ms']['current']:.3f} | {e['latency_reduction_percent']:.2f}% | {e['positive_rounds']}/41 |" for e in summary['controls'])
perf_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} / {e['shape'][1]} | {e['median_ms']['fp16_full_prefix']:.3f} | {e['median_ms']['w8a16_full_prefix']:.3f} | {e['median_ms']['cpu_fp16_full_prefix']:.3f} |" for e in summary['performance'])
decode_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} | {e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | {e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in summary['performance'])
check_table='\n'.join(f"| {name} | {c['checks']} | {c['timings']} | {c['npu_calls']} |" for name,c in summary['cases'].items())
report=f'''# SmolLM2：多 token prefill 的整层单图

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

冻结 SmolLM2-135M FP16/W8A16 的多 token forward/prefill 已默认使用整层单图。B≤8、S≤64 的连续 FP16 NPU 输入沿用原 attention、两次 RMSNorm、FFN 与 KV 更新，每层暖 prefill 提交 4→1 次、输出分配 4→3 次。decode 继续沿用上一阶段的一图／四输出分配路径，调用 API 不变。

## 实现与正确性

生产改动只扩展已有图构造器的多 token 入口和融合选择条件，并保留一个用于同实例对照的私有 prefill 开关。沿用相同的 FP32 SHAVE RMSNorm、Neumaier 补偿求和、epsilon=1e-5 和两次 FP16 舍入；矩乘继续走 FP16/W8A16 DPU。没有改 precision helper、布局白名单、CMX／别名／重定位保护，也没有重建内核或 native DLL/PYD。

256 个真实多 token 图（FP16/W8A16 × B1..B8 × S=2,3,7,8,17,32,63,64 × forward/prefill）全部与原四图基线逐位一致，包括 y 与 K/V。每个 blob 仅修改共享 Power 代码窗口中的 5136 字节，窗口总长 21216 字节；长度、其他代码、数据、DMA、barrier 与任务字节不变。45 种观测到的物理行排列／行分片组合均符合上一阶段完整行校验，不需要新的载体布局。

48 个单 token forward/prefill/decode 图的 IR XML 与权重字节完全未变。另有 8 项非法形状／入口检查，包括 S>64、B>8、多 token decode、诊断 stage 和 CPU reference 的内部 layer 参数拒绝。单 token 的数学和设备图保持原样。

## 同实例暖 prefill 对照

同一模型、输入与驻留参数，仅切换私有 prefill 开关。14 组，各 41 轮 ABBA/BAAB，每策略 82 个样本。输出/K/V/位置计数位模式一致，暖显式主机上传、下载与编译为零。

| 层数 / B / token | 模式 | 四图 ms | 整层单图 ms | 中位延迟降幅 | 更快轮数 |
|---|---|---:|---:|---:|---:|
{control_table}

本轮中位延迟下降 10.3%–65.4%。四层 B1、32 token 的 FP16 为 8.47→4.67 ms，W8A16 为 8.53→4.71 ms；B8×64 的收益约 10%–14%。输入越大，减少提交在总时间中的占比越小。本轮没有设备时间戳或带宽计数，不能据此区分其余时间是计算、DMA 还是等待。

计时包含验证、绑定、提交、设备执行和同步 fence，排除初始加载、编译和显式传输。复制计数不含编译器／驱动内部 DMA。桌面负载、频率和内存状态未锁定；以上是本机暖层片段时间，不能换算为完整 LLM 生成速度或功耗。

## 与 CPU 的完整前缀 forward 对照

21 轮轮转各路径，CPU 为 FP16、8 线程。此表比较仅返回 hidden states 的 forward，NPU 每层一图；它与上表返回 K/V 的 prefill 有不同输出及分配范围，不能混用两表的绝对时间。

| 层数 / B / token | NPU FP16 ms | NPU W8A16 ms | CPU FP16 ms |
|---|---:|---:|---:|
{perf_table}

## 原 decode 路径的独立实例对照

decode 的图／权重不变，两版本均每层一图。不同实例具有不同的准备、gamma 和分配状态，下表仍保留更慢的组，不把这些差值归因于图或内核优化。不能从该轮独立实例数据推断未改动的 decode 获得稳定加速。

| 层数 / B | FP16 旧→新 ms | W8A16 旧→新 ms | CPU FP16 ms |
|---|---:|---:|---:|
{decode_table}

## 回归与快照

{summary['numerical_records']} 条真实权重数值记录、{summary['fixed_input_cache_tensor_checks']} 个固定输入 K/V 对照、{summary['continuous_decode_steps']} 次连续位置解码通过。最大执行相对 L2 {summary['max_execution_relative_l2']*100:.4f}%，scaled error {summary['max_execution_scaled_error']*100:.4f}%；原 0.3%／1% 执行门槛及 3% 量化质量门槛未放宽。24 条新旧连续 decode、8 条普通 forward 与 8 组 CPU+NPU 混合生成保持位模式／生成 token 一致。

14 个多 token 快照检查与两组保留检查验证因果性、batch 隔离、独立 y/K/V、未写入位置的零 padding 和正确位置计数。同一实例 B1×2→B3×7→B2×17→B8×63→B1×64→B4×32→B1×2 切换后，所有旧输出／KV／位置快照不变。除首次参数准备外，新 gamma 广播来自 NPU 驻留参数。

首次快照测试错误地将新长度的 FP32 位置元数据上传计入 gamma 上传，被测试断言拒绝；context_failed=False。修正测试为先单独初始化并记录该不可变标量，再检查冷形状的 gamma 广播；未改生产代码或 warm 零复制门槛。失败源码／报告在 failures/cold-position-counter/。新前缀长度首次初始化仍可能上传一个 FP32 标量，warm prefill 没有该开销。

| 成功设备进程 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{check_table}

成功设备进程合计 {summary['total_npu_calls']:,} 次记录内 NPU 调用，全部健康关闭、存活分配归零。通用 torch 后端、precision helper、SHAVE ELF 与 native SHA 均与上一阶段相同；既有 105 项通用精度、16 项 ELF／载体及 17 项载体保护证据在 smollm-whole-layer/，本轮另验证实际多 token 图与新增入口。

范围仍为冻结 SmolLM2-135M FP16/W8A16、B≤8、总位置≤64、相同前缀长度和无 padding mask。没有增加训练、任意模型、跨层并行或完整 LLM 全 NPU 后端。运行仍不依赖 OpenVINO Python/runtime，仍需要 Intel 驱动图编译器。

## 复现

```powershell
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','smollm-prefill-fusion','--case','numeric'
.\\run_torch_npu.ps1 -Script .\\torch_npu_prefill_benchmark.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_prefill_probe.py -ScriptArguments '--mode','fp16'
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_prefill_probe.py -ScriptArguments '--mode','w8a16'
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_prefill_ownership.py
$env:PYTHONPATH="$PWD;$PWD\\npunlock\\python"
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\smollm_prefill_integrity.py
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_smollm_prefill.py
```

完整 sources、kernel manifest、before/after、changes.diff、原始／修改 blob 和证据 SHA 在本目录。初步 32 图报告及旧探测源码保存在 preliminary/；最终审计使用分模式的 256 图证据。模型 safetensors SHA：`{summary['checkpoint_file_sha256']}`。
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print({k:summary[k] for k in ['status','multi_token_graph_records','numerical_records','fixed_input_cache_tensor_checks','continuous_decode_steps','total_npu_calls']})
