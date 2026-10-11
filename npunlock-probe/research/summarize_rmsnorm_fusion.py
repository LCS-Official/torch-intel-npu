"""Audit final fusion evidence and write the local measurements report."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-fusion'
FIX=ROOT/'npunlock/tests/fixtures/npu4000'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
baseline=json.loads((OUT.parent/'rmsnorm-small/summary.json').read_text())
names=['probe','numeric','boundary','guards','positions','hybrid','paired','regression','profile','performance']
cases={name:read(name) for name in names}
extras={name:read(name) for name in ['layout-probe','model-probe','ownership','control']}
sources=cases['numeric']['sources'];manifest=cases['numeric']['kernel_manifest']
for name,case in cases.items():
    assert case['sources']==sources and case['kernel_manifest']==manifest and case['checkpoint']==baseline['checkpoint'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
    assert case['final_stats']['hardware_device']=='8086:643e' and case['final_stats']['driver_version']==1005540,name
    for source,value in case['sources'].items():assert digest(ROOT/source)==value,(name,source)
for name,value in sources.items():
    if name not in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py','npunlock/python/npunlock/npu4000.py']:
        assert value==baseline['sources'][name],name
for name,path in [('torch_npu_smollm.py','torch_npu_smollm.py'),('torch_npu_smollm_kv_demo.py','torch_npu_smollm_kv_demo.py'),('npu4000.py','npunlock/python/npunlock/npu4000.py')]:
    assert digest(OUT/'before'/name)==baseline['sources'][path],name
assert digest(OUT/'candidate.py')==sources['torch_npu_smollm.py']==extras['model-probe']['candidate_sha256']==extras['control']['candidate_sha256']
assert manifest==baseline['kernel_manifest']==json.loads((OUT/'before/manifest.json').read_text())
for name,value in manifest['source_sha256'].items():assert digest(FIX/name)==value
assert digest(FIX/'rmsnorm.c')==digest(OUT/'before/rmsnorm.c')
assert digest(FIX/'rmsnorm/rmsnorm-f16.elf')==manifest['elf_sha256']==digest(OUT/'before/rmsnorm-f16.elf')
assert digest(FIX/'rmsnorm/rmsnorm-f16.s')==manifest['assembly_sha256']
assert digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')==baseline['checkpoint_file_sha256']

guards=read('patch-guards');assert guards['status']=='passed' and len(guards['checks'])==9
for source,value in guards['sources'].items():assert digest(ROOT/source)==value
assert 'Ran 16 tests' in (OUT/'offline.log').read_text() and 'OK' in (OUT/'offline.log').read_text()
generic=read('generic-quick');generic_provenance=read('generic-quick-provenance')
assert generic['status']=='passed' and len(generic['checks'])==105 and not generic['openvino_installed']
assert digest(OUT/'generic-quick.json')==generic_provenance['result_sha256']
for source,value in generic_provenance['sources'].items():assert digest(ROOT/source)==value
assert generic['execution']['successful_npu_calls']==435

assert len(extras['layout-probe']['checks'])==20 and len(extras['model-probe']['checks'])==42
for e in extras['layout-probe']['checks']:
    assert digest(OUT/(e['tag']+'.blob'))==e['blob_sha256']
    assert e['complete_rows']==(e['rows']>=2)
for e in extras['model-probe']['checks']:
    assert e['status']=='passed' and all(e['bits_equal']) and e['graph_calls']==1 and e['warm_host_copies_and_compiles']==0
    patch=e['patch'];original=(OUT/(e['tag']+'-original.blob')).read_bytes();modified=(OUT/(e['tag']+'-patched.blob')).read_bytes()
    assert hashlib.sha256(original).hexdigest()==patch['original_sha256']
    assert hashlib.sha256(modified).hexdigest()==patch['patched_sha256']
    assert patch['kernel_elf_sha256']==manifest['elf_sha256'] and patch['accumulation']=='fp32_neumaier'
    start=patch['code_offset'];end=start+patch['code_bytes']
    assert original[:start]==modified[:start] and original[end:]==modified[end:] and len(original)==len(modified)
    assert patch['code_bytes']==5136 and patch['window_bytes']==21216
assert all(e['all_output_cache_bits_equal'] for e in cases['paired']['checks'])
assert all(e['old_new_bits_equal'] for e in cases['regression']['checks'])
assert all(e['generated_tokens_equal'] for e in cases['hybrid']['checks'])
entries=[e for name in ['probe','numeric','boundary','positions'] for e in cases[name]['checks'] if 'execution' in e]
cache_entries=[e for entry in entries for e in entry.get('cache_local_reference',[])]
assert all(e['execution']['relative_l2']<.003 and e['execution']['max_scaled_error']<.01 for e in entries)
assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in cache_entries)
assert all(e['quality_official']['relative_l2']<.03 for e in entries if 'quality_official' in e)
for e in cases['paired']['checks']:
    if e['warmed']:
        assert e['graph_calls']==e['num_layers']*(2 if e['batch']>=2 else 4)
        assert e['previous_graph_calls']==e['num_layers']*4
assert len(extras['ownership']['checks'])==17
for e in extras['ownership']['checks'][:-1]:
    assert e['graph_calls']==2 and e['perturbation_isolated'] and e['old_snapshot_unchanged'] and e['cold_decode_host_copies']==0
    assert e['held_output_unchanged'] and e['warm_host_copies_and_compiles']==0
    assert all(c['relative_l2']<.003 and c['max_scaled_error']<.01 for sample in e['solo_comparisons'] for c in sample)
assert extras['ownership']['checks'][-1]['public_outputs_and_kv_independent']
assert len(extras['control']['timings'])==len(extras['control']['checks'])==8
assert extras['control']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
assert all(e['all_bits_equal'] for e in extras['control']['checks'])
for e in extras['control']['timings']:
    assert e['rounds']==41 and e['same_model_input_cache'] and e['warm_host_copies_and_compiles']==0
    assert all(len(v)==82 for v in e['samples_ms'].values()) and len(e['round_pairs_ms'])==41
    assert e['graph_calls']==dict(previous=e['num_layers']*4,current=e['num_layers']*(2 if e['batch']>=2 else 4))
    assert e['allocations_per_decode']==dict(previous=e['num_layers']*5,current=e['num_layers']*5)
for e in cases['profile']['timings']:assert sum(e['native_calls'].values())==e['num_layers']*(2 if e['batch']>=2 else 4)
for e in cases['performance']['timings']:
    for name,calls in e['graph_calls'].items():
        if name.endswith('_decode') and not name.startswith('cpu'):
            assert calls==e['num_layers']*(2 if e['shape'][0]>=2 and not name.startswith('previous_') else 4)

summary=dict(status='passed',date='2026-10-05',sources=sources,kernel_manifest=manifest,
    checkpoint=cases['numeric']['checkpoint'],checkpoint_file_sha256=baseline['checkpoint_file_sha256'],packages=cases['numeric']['packages'],
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),
    max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    controls=extras['control']['timings'],profiles=cases['profile']['timings'],performance=cases['performance']['timings'],
    total_npu_calls=sum(c['final_stats']['npu_calls'] for c in {**cases,**extras}.values()),
    generic_quick=dict(checks=len(generic['checks']),npu_calls=generic['execution']['successful_npu_calls'],rejections=len(generic['rejections'])),
    cases={name:dict(checks=len(c.get('checks',[])),timings=len(c.get('timings',[])),npu_calls=c['final_stats']['npu_calls']) for name,c in {**cases,**extras}.items()})
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
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
control_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['median_ms']['previous']:.3f} | {e['median_ms']['current']:.3f} | {e['latency_reduction_percent']:.2f}% | {e['positive_rounds']}/41 |" for e in summary['controls'])
perf_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} | {e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | {e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in summary['performance'])
profile_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['mean_stage_percent']['norm_attention']:.1f}% | {e['mean_stage_percent']['norm_ffn']:.1f}% | {sum(e['native_calls'].values())} |" for e in summary['profiles'] if e['batch']>=2)
check_table='\n'.join(f"| {name} | {c['checks']} | {c['timings']} | {c['npu_calls']} |" for name,c in summary['cases'].items())
report=f'''# 图内 RMSNorm → attention / FFN 投影融合

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

已接入冻结 SmolLM2 层的单 token B2..B8 路径，包括 forward、单 token prefill 与 decode。每层暖 decode 从 4 图降到 2 图：FP32 SHAVE RMSNorm 与 attention 的 DPU 投影组成一图，第二次 RMSNorm 与 FFN 的 DPU 投影组成另一图。公开调用方法保持不变，暖分配仍为 5 次/层。

归一化保留同一生产内核、FP32 Neumaier 补偿求和、epsilon 与两次 FP16 舍入；DPU 权重支持 FP16/W8A16。没有重建内核、DLL/PYD，也未更改通用严格 FP32 算子。运行不需要 OpenVINO Python/runtime 包，仍使用 Intel 驱动图编译器。

## 融合范围与安全边界

20 组编译布局探测显示：B1 默认四 tile 将 576 元素拆成 80/64 元素片段；一 tile 仍拆成两段 288。RMSNorm 必须看到完整一行，因此 B1 和多 token 前缀继续使用原路径。B2..B8 全图的 attention-prefill、attention-decode、FFN 共 42 组验证通过，全部与分图基线逐位相同。编译器可为不同 DPU tile 复制完整行的归一化任务，不能把任务数量等同于唯一行数。

只替换具有固定 Power 代码指纹的 SHAVE 文本窗口，并验证完整行布局、FP16 静态 CMX memref、输入/输出不重叠、范围引用与代码大小。42 对原始/修改 blob 的代码窗口以外字节完全相同：未修改 DMA、barrier、buffer 或任务调度。该实验接口仅支持已验证的设备/编译器与载体指纹，不是任意图融合器。

首次批量 decode 复用 prefill 已驻留的基础 gamma，在 NPU 上广播新的行数，不重新上传 CPU 权重。首次编译/广播仍有准备成本；预热后显式上传/下载/编译为零。计数不包含编译器内部 DMA，不能称作零 IO。

## 同模型、同输入、同缓存对照

同一实例仅切换融合开关；两种策略共用输入、gamma、权重和原 KV，所有输出/KV 位模式一致。41 轮 ABBA/BAAB，每策略 82 样本，计时排除开关切换与冷准备。

| 层数 / B | 模式 | 分图 ms | 融合 ms | 中位延迟降幅 | 融合更快轮数 |
|---|---|---:|---:|---:|---:|
{control_table}

B1 是未改变执行路径的对照，其差异反映波动。结果为暖模型片段计时，包含绑定、提交、设备执行与 fence 同步；未锁定桌面负载、频率或内存状态。

## 独立实例与 CPU 对照

每组 21 轮轮转各路径；CPU 为 FP16、8 线程。以下均为 decode ms，完整原始样本及 full-prefix 时间在 performance.json。该比较与上面的同实例对照有不同分配/缓存条件，均保留实测结果。

| 层数 / B | FP16 分图→融合 | W8A16 分图→融合 | CPU FP16 |
|---|---:|---:|---:|
{perf_table}

这些冻结层片段时间不能直接换算为完整 LLM 生成速度，也未测功耗。模型支持范围仍是 B≤8、相同前缀长度、总位置≤64；不增加训练、padding mask 或完整模型支持。

## 融合后阶段比例

比例来自 21 轮 traced 平均值。归一化已包含在对应 attention/FFN 阶段中，不能把旧独立 RMSNorm bucket 的零值理解为归一化耗时为零。

| 层数 / B | 模式 | norm + attention | norm + FFN | 原生调用数 |
|---|---|---:|---:|---:|
{profile_table}

## 验证及追溯

{summary['numerical_records']} 条真实权重数值记录、{summary['fixed_input_cache_tensor_checks']} 个固定输入 K/V 对照、{summary['continuous_decode_steps']} 次连续位置解码通过。24 条新旧 cached decode、8 条普通 forward 与 8 组混合生成保留逐位/生成 token 比较。执行最大相对 L2 {summary['max_execution_relative_l2']*100:.4f}%，scaled error {summary['max_execution_scaled_error']*100:.4f}%；原执行门槛 0.3% / 1% 与量化质量门槛 3% 保留。

16 组样本隔离检查覆盖 B2..B8、单 token prefill、不同样本独立 B1 对照、扰动隔离、旧 KV 与公开输出保留、冷解码零主机复制。不同 batch 形状的结果采用原数值门槛，不保证逐位一致。两线程的 6 次 warm decode 共 12 图，输出及 K/V/position 独立，值一致；NPU 锁仍串行执行，该检查不证明并行加速。

9 项离线融合保护检查、原有 16 项 ELF/载体检查与通用 FP16/FP32 快速回归通过；后者为 105 项、435 次 NPU 运算，涵盖特殊值、归约、矩乘、autograd、torch.compile 与 11 类拒绝行为。

| 成功进程/用例 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{check_table}

上述成功设备进程共 {summary['total_npu_calls']:,} 次记录内 NPU 调用，全部健康关闭、存活分配归零；通用快速回归另计 435 次。两次开发阶段失败单独保存在 failures/：静态 invocationCount 为 0 的校验过严，以及冷 decode 重复上传 gamma，均已修复，未放宽数值或无主机复制门槛。初步性能样本保存在 control-preliminary.*，最终结论使用 control.json。

模型 safetensors SHA：`{summary['checkpoint_file_sha256']}`。sources、kernel manifest、before/after/diff、原始/修改 blob 与审计结果保存在本目录。

```powershell
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_fusion_probe.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_fusion_guards.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_fused_model_probe.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_fusion_ownership.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','rmsnorm-fusion','--case','numeric'
.\\run_torch_npu.ps1 -Script .\\torch_npu_norm_fusion_benchmark.py
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_rmsnorm_fusion.py
```
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print({k:summary[k] for k in ['status','numerical_records','fixed_input_cache_tensor_checks','continuous_decode_steps','total_npu_calls']})
