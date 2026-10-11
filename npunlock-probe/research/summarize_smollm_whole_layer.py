"""Audit whole-layer numerical, ownership and timing evidence."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-whole-layer'
FIX=ROOT/'npunlock/tests/fixtures/npu4000'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))

baseline=json.loads((OUT.parent/'rmsnorm-b1-fusion/summary.json').read_text())
names=['probe','numeric','boundary','guards','positions','hybrid','paired','regression','profile','performance']
cases={name:read(name) for name in names}
extras={name:read(name) for name in ['model-probe','ownership','control']}
tested_sources=cases['numeric']['sources'];manifest=cases['numeric']['kernel_manifest']
# The sole post-device-test change restores the old CPU-reference whitelist.
# Exact byte comparison proves every device method and graph builder unchanged.
measured=(OUT/'candidate.py').read_bytes()
before=b"if segment not in (None,'attention','ffn','layer') or (segment is not None and (not isinstance(normalized,torch.Tensor)"
after=b"if segment not in (None,'attention','ffn') or (segment is not None and (not isinstance(normalized,torch.Tensor)"
assert measured.count(before)==1 and (ROOT/'torch_npu_smollm.py').read_bytes()==measured.replace(before,after,1)
sources={**tested_sources,'torch_npu_smollm.py':digest(ROOT/'torch_npu_smollm.py')}
for name,case in cases.items():
    assert case['sources']==tested_sources and case['kernel_manifest']==manifest and case['checkpoint']==baseline['checkpoint'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
    assert case['final_stats']['hardware_device']=='8086:643e' and case['final_stats']['driver_version']==1005540,name
    for source,value in case['sources'].items():
        # The same-instance benchmark ran before promoting its candidate.
        target=(OUT/('before/torch_npu_smollm.py' if name=='control' else 'candidate.py')) if source=='torch_npu_smollm.py' else ROOT/source
        assert digest(target)==value,(name,source)
for name,value in sources.items():
    if name not in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py','npunlock/python/npunlock/npu4000.py']:
        assert value==baseline['sources'][name],name
for name,path in [('torch_npu_smollm.py','torch_npu_smollm.py'),('torch_npu_smollm_kv_demo.py','torch_npu_smollm_kv_demo.py'),('npu4000.py','npunlock/python/npunlock/npu4000.py')]:
    assert digest(OUT/'before'/name)==baseline['sources'][path],name
assert digest(OUT/'candidate.py')==tested_sources['torch_npu_smollm.py']==extras['model-probe']['candidate_sha256']==extras['control']['candidate_sha256']
assert extras['model-probe']['baseline_adapter_sha256']==extras['control']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
assert manifest==baseline['kernel_manifest']==json.loads((OUT/'before/manifest.json').read_text())
for name,value in manifest['source_sha256'].items():assert digest(FIX/name)==value
assert digest(FIX/'rmsnorm.c')==digest(OUT/'before/rmsnorm.c')
assert digest(FIX/'rmsnorm/rmsnorm-f16.elf')==manifest['elf_sha256']==digest(OUT/'before/rmsnorm-f16.elf')
assert digest(FIX/'rmsnorm/rmsnorm-f16.s')==manifest['assembly_sha256']
assert digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')==baseline['checkpoint_file_sha256']

from npunlock.npu4000 import inspect,POWER_TEXT_SHA256
layouts=set()
assert len(extras['model-probe']['checks'])==48
for e in extras['model-probe']['checks']:
    assert e['status']=='passed' and all(e['bits_equal']) and e['graph_calls']==1 and e['warm_host_copies_and_compiles']==0
    patch=e['patch'];original=(OUT/(e['tag']+'-original.blob')).read_bytes();modified=(OUT/(e['tag']+'-patched.blob')).read_bytes()
    assert hashlib.sha256(original).hexdigest()==patch['original_sha256']
    assert hashlib.sha256(modified).hexdigest()==patch['patched_sha256']
    assert patch['kernel_elf_sha256']==manifest['elf_sha256'] and patch['accumulation']=='fp32_neumaier'
    start=patch['code_offset'];end=start+patch['code_bytes']
    assert original[:start]==modified[:start] and original[end:]==modified[end:] and len(original)==len(modified)
    assert patch['code_bytes']==5136 and patch['window_bytes']==21216 and patch['shape']==[max(2,e['batch']),576]
    assert patch['carriers']==['rmsnorm_carrier','rmsnorm_ffn']
    assert all(c['elements']%576==0 for c in patch['contracts'])
    for task in inspect(original,memref_count=3,kernel_sha256=POWER_TEXT_SHA256)['invocations']:
        for ref in task['memrefs']:
            count=ref['elements']//576
            assert (ref['order'],ref['dims'],ref['strides_bits']) in [
                ('0x4321',(576,count,1,1),(16,9216,9216*count,9216*count)),
                ('0x2431',(1,576,count,1),(16,16,9216,9216*count))]
            layouts.add((ref['order'],count))
patch_guards=read('patch-guards');assert patch_guards['status']=='passed' and len(patch_guards['checks'])==17
for name,value in patch_guards['sources'].items():assert digest(OUT/'candidate.py' if name=='torch_npu_smollm.py' else ROOT/name)==value
assert patch_guards['candidate_sha256']==tested_sources['torch_npu_smollm.py']
reference_guard=read('reference-guard');assert reference_guard['status']=='passed' and len(reference_guard['checks'])==4
for name,value in reference_guard['sources'].items():assert digest(ROOT/name)==value
assert reference_guard['measured_candidate_sha256']==tested_sources['torch_npu_smollm.py']
assert 'Ran 16 tests' in (OUT/'offline.log').read_text() and '\nOK' in (OUT/'offline.log').read_text()
quick=read('quick');assert quick['status']=='passed' and quick['openvino_installed'] is False
assert len(quick['checks'])==105 and all(e['status']=='passed' for e in quick['checks'])

assert all(e['all_output_cache_bits_equal'] for e in cases['paired']['checks'])
assert all(e['old_new_bits_equal'] for e in cases['regression']['checks'])
assert all(e['generated_tokens_equal'] for e in cases['hybrid']['checks'])
entries=[e for name in ['probe','numeric','boundary','positions'] for e in cases[name]['checks'] if 'execution' in e]
cache_entries=[e for entry in entries for e in entry.get('cache_local_reference',[])]
assert all(e['execution']['relative_l2']<.003 and e['execution']['max_scaled_error']<.01 for e in entries)
assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in cache_entries)
assert all(e['quality_official']['relative_l2']<.03 for e in entries if 'quality_official' in e)
for e in cases['paired']['checks']:
    if e['warmed']:assert e['graph_calls']==e['num_layers'] and e['previous_graph_calls']==e['num_layers']*2
for e in cases['positions']['checks']:
    if 'positions_checked' in e:assert e['graph_calls_per_decode']==e['num_layers'] and e['new_sessions']==0
assert len(extras['ownership']['checks'])==16
for e in extras['ownership']['checks'][:-2]:
    assert e['graph_calls']==1 and e['perturbation_isolated'] and e['old_snapshot_unchanged'] and e['cold_decode_host_copies']==0
    assert e['held_output_unchanged'] and e['warm_host_copies_and_compiles']==0
    assert all(c['relative_l2']<.003 and c['max_scaled_error']<.01 for sample in e['solo_comparisons'] for c in sample)
assert extras['ownership']['checks'][-2]['all_bits_unchanged'] and extras['ownership']['checks'][-2]['gamma_shared_between_b1_b2']
assert extras['ownership']['checks'][-1]['public_outputs_and_kv_independent'] and extras['ownership']['checks'][-1]['graph_calls']==6
assert len(extras['control']['timings'])==len(extras['control']['checks'])==12
assert all(e['all_bits_equal'] for e in extras['control']['checks'])
for e in extras['control']['timings']:
    assert e['rounds']==41 and e['same_model_input_cache'] and e['warm_host_copies_and_compiles']==0
    assert all(len(v)==82 for v in e['samples_ms'].values()) and len(e['round_pairs_ms'])==41
    assert e['graph_calls']==dict(previous=e['num_layers']*2,current=e['num_layers'])
    assert e['allocations_per_decode']==dict(previous=e['num_layers']*5,current=e['num_layers']*4)
    assert e['latency_reduction_percent']>0
for e in cases['profile']['timings']:assert sum(e['native_calls'].values())==e['num_layers']==e['native_calls']['norm_layer']
for e in cases['performance']['timings']:
    for name,calls in e['graph_calls'].items():
        if name.endswith('_decode') and not name.startswith('cpu'):assert calls==e['num_layers']*(2 if name.startswith('previous_') else 1)

summary=dict(status='passed',date='2026-10-05',sources=sources,tested_sources=tested_sources,kernel_manifest=manifest,reference_guard_checks=4,
    checkpoint=cases['numeric']['checkpoint'],checkpoint_file_sha256=baseline['checkpoint_file_sha256'],packages=cases['numeric']['packages'],
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),
    max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    controls=extras['control']['timings'],profiles=cases['profile']['timings'],performance=cases['performance']['timings'],
    observed_row_layouts=sorted(layouts),patch_guards=len(patch_guards['checks']),offline_checks=16,quick_checks=len(quick['checks']),
    total_npu_calls=sum(c['final_stats']['npu_calls'] for c in {**cases,**extras}.values()),
    cases={name:dict(checks=len(c.get('checks',[])),timings=len(c.get('timings',[])),npu_calls=c['final_stats']['npu_calls']) for name,c in {**cases,**extras}.items()},
    quick_npu_calls=quick['execution']['successful_npu_calls'])
changed={'torch_npu_smollm.py':ROOT/'torch_npu_smollm.py','torch_npu_smollm_kv_demo.py':ROOT/'torch_npu_smollm_kv_demo.py',
    'npu4000.py':ROOT/'npunlock/python/npunlock/npu4000.py','rmsnorm.c':FIX/'rmsnorm.c',
    'manifest.json':FIX/'rmsnorm/manifest.json','rmsnorm-f16.elf':FIX/'rmsnorm/rmsnorm-f16.elf',
    'TORCH_NPU_README.md':ROOT/'TORCH_NPU_README.md','NPU4000.md':ROOT/'npunlock/docs/NPU4000.md'}
(OUT/'after').mkdir(exist_ok=True);diff=[]
for name,path in changed.items():
    shutil.copyfile(path,OUT/'after'/name)
    if path.suffix!='.elf':
        diff.extend(difflib.unified_diff((OUT/'before'/name).read_text(encoding='utf-8').splitlines(True),path.read_text(encoding='utf-8').splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(OUT/'changes.diff').write_text(''.join(diff),encoding='utf-8')
summary['after_sha256']={name:digest(OUT/'after'/name) for name in changed};summary['diff_sha256']=digest(OUT/'changes.diff')
summary['audit_sha256']=digest(Path(__file__))
summary['evidence_sha256']={name:digest(OUT/(name+'.json')) for name in [*cases,*extras,'patch-guards','quick','reference-guard']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
control_table='\n'.join(f"| {e['num_layers']} / {e['batch']} / {e['prefix_tokens']} | {e['mode']} | {e['median_ms']['previous']:.3f} | {e['median_ms']['current']:.3f} | {e['latency_reduction_percent']:.2f}% | {e['positive_rounds']}/41 |" for e in summary['controls'])
perf_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} | {e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | {e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in summary['performance'])
profile_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['mean_stage_percent']['norm_layer']:.1f}% | {e['mean_native_total_ms']/e['mean_traced_ms']*100:.1f}% | {sum(e['native_calls'].values())} |" for e in summary['profiles'])
check_table='\n'.join(f"| {name} | {c['checks']} | {c['timings']} | {c['npu_calls']} |" for name,c in summary['cases'].items())
report=f'''# SmolLM2：两图合为整层单图

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

冻结 SmolLM2-135M FP16/W8A16 的 B1..B8 单 token forward/prefill/decode 已默认把两次 RMSNorm、attention、FFN 和 KV 更新放入同一图。每层暖 decode 从 2 次提交／5 次输出分配降到 1 次／4 次，四层从 8 图降到 4 图。移除了 attention→FFN 间 DDR 中间张量的独立分配与绑定；图内具体 DMA 由编译器调度，没有宣称消除所有数据移动。

## 实现和精度

复用已有 SmolLM 图构造器、resident gamma、多输出运行接口与 RMSNorm SHAVE 内核。两个具名 Power 载体分别接在 attention 与 FFN 前；每个图只替换一次共享的 Power 代码窗口，两处调用执行同一 RMSNorm 实现。FP32 Neumaier 补偿求和、epsilon=1e-5、先 normalize→FP16 再乘 gamma→FP16 的舍入顺序未变；DPU 的 FP16/W8A16 矩乘路径未变。没有新增或替换依赖，没有重建 ELF、native DLL/PYD，没有改通用严格 FP32。

B1 的两次 norm 都沿用内部复制完整行再切回第一行的方案，attention/FFN/公开 KV 仍为 B1。第二次 norm 在 B2..B8 可编译为带前导单例维度的布局。whole_layer 模式精确接受 0x4321 `(576,rows,1,1)` 与 0x2431 `(1,576,rows,1)`，并校验相应的完整行密集步长、行数、精度和三 memref 一致性。内核本来就支持跳过前导单例轴；未放宽 single-carrier 模式的布局白名单。

仍要求固定 8086:643e／编译器 8.3、核代码指纹、静态固定 CMX、边界、重定位及无输入输出重叠。48 个真实图（B1..B8 × FP16/W8A16 × forward/prefill/decode）全部与原两图实现逐位一致，包括 y、K/V 与 FP32 位置计数。原始／修改 blob 等长，5136 字节代码区以外逐位不变；代码窗口为 21216 字节，其他 SW/DPU 代码和任务不修改。

## 同实例性能对照

同一模型、输入、gamma 与原始 KV，仅切换私有整层开关。12 组，各 41 轮 ABBA/BAAB、每策略 82 个样本；所有 y/K/V/位置位模式一致。暖显式上传、下载、编译为零。四层两路径分别分配 20／16 次。

| 层数 / B / 前缀 | 模式 | 两图 ms | 整层单图 ms | 中位延迟降幅 | 更快轮数 |
|---|---|---:|---:|---:|---:|
{control_table}

本轮中位延迟下降 20.4%–30.2%，各组 37–41／41 轮更快。计时包括 Python 验证、绑定、提交、设备执行和同步 fence，排除初始加载、编译和显式传输。复制计数不含编译器／驱动内部 DMA。桌面负载、频率和内存状态未锁定；这些是本机暖层片段时间，不代表完整 LLM 速度或功耗。

control.json 在候选版本接入默认路径前生成，其 sources 中的旧 root 适配器 SHA 对应本目录 before 快照；实际计时使用 candidate.py。设备测试之后仅收紧了 CPU reference 的 segment 白名单，拒绝误接受的内部 layer 参数。审计逐字节验证生产文件相对 candidate.py 只有这一处校验差异，所有设备方法、图构造和运行代码完全相同；另有 3 项原 CPU 参考调用结果／trace 一致性和 1 项 layer 拒绝检查。sources 保存最终源码 SHA，tested_sources 保存设备测试时的 SHA，未改写原始数值／计时证据。

## 独立实例与 CPU

每组 21 轮轮转各路径；CPU 为 FP16、8 线程，重复 decode 后恢复原前缀。两图和整层为独立实例。下表保留所有中位时间，不能将独立实例／不同桌面负载的差值与同实例对照混用。

| 层数 / B | FP16 两图→整层 ms | W8A16 两图→整层 ms | CPU FP16 ms |
|---|---:|---:|---:|
{perf_table}

## 执行分解

21 轮 traced 平均比例。归一化、attention、FFN 包含在 norm_layer 阶段，单独旧 bucket 为零不代表其中算子免费。native 包络包含绑定、提交、执行和等待，不能据此称为纯内核时间或断言已被计算／带宽完全限制。

| 层数 / B | 模式 | 整层阶段占比 | native 包络占比 | 原生调用数 |
|---|---|---:|---:|---:|
{profile_table}

## 回归和边界

{summary['numerical_records']} 条真实权重数值记录、{summary['fixed_input_cache_tensor_checks']} 个固定输入 K/V 对照、{summary['continuous_decode_steps']} 次连续位置解码通过。最大执行相对 L2 {summary['max_execution_relative_l2']*100:.4f}%，scaled error {summary['max_execution_scaled_error']*100:.4f}%；保留原 0.3%／1% 执行门槛与 3% 量化质量门槛。24 条新旧 cached decode、8 条普通 forward 与 8 组 CPU+NPU 混合生成保持位模式／生成 token 一致。

14 组 batch 隔离／快照检查，以及一个实例 B1→B2→B1→B8→B3→B1 切换通过；B1/B2 复用两行 gamma。两个调用者对同一缓存执行 6 次 warm decode，共 6 图，y/K/V/position 存储独立，旧输出与旧 KV 不变。执行仍由既有 NPU 锁串行化，此检查没有验证并行加速。

17 项载体保护检查（2 项接受、15 项拒绝）、16 项旧 ELF／载体离线测试与 {summary['quick_checks']} 项通用 FP16/严格 FP32 快速回归通过。首次只接受原布局的探测、随后只接受单例一行的探测分别拒绝 42／24 组；都是执行前布局保护拒绝，context_failed=False。诊断发现两种完整行排列以及 1／2 行分片后补齐精确校验，失败报告／源码在 failures/，没有修改内核或放宽密集步长／CMX 保护。

| 成功设备进程 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{check_table}

以上成功设备进程合计 {summary['total_npu_calls']:,} 次 NPU 调用，全部健康关闭、存活分配归零。通用快速回归另有 {summary['quick_npu_calls']} 次调用。sources、kernel manifest、before/after、changes.diff 与证据 SHA 在 summary.json。

## 当前范围与复现

API 不变，模型／输入仍使用 intel_npu；prefill/decode 返回 NPU KV 快照。仍为冻结 SmolLM2-135M FP16/W8A16、B≤8、相同前缀长度、总位置≤64；多 token 前缀仍用原四图路径。不增加训练、任意模型、padding mask、跨层并行或完整 LLM 全 NPU 后端。运行不依赖 OpenVINO Python/runtime，仍依赖 Intel 驱动图编译器。

```powershell
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','smollm-whole-layer','--case','numeric'
.\\run_torch_npu.ps1 -Script .\\torch_npu_whole_layer_benchmark.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_whole_layer_probe.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_whole_layer_ownership.py
$env:PYTHONPATH="$PWD;$PWD\\npunlock\\python"
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\smollm_whole_layer_guards.py
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\smollm_whole_layer_reference_guard.py
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_smollm_whole_layer.py
```

模型 safetensors SHA：`{summary['checkpoint_file_sha256']}`。
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print({k:summary[k] for k in ['status','numerical_records','fixed_input_cache_tensor_checks','continuous_decode_steps','total_npu_calls']})
