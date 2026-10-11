"""Audit complete-row B1 fusion without changing previous phase evidence."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-b1-fusion'
FIX=ROOT/'npunlock/tests/fixtures/npu4000'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
baseline=json.loads((OUT.parent/'rmsnorm-fusion/summary.json').read_text())
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
    if name not in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:assert value==baseline['sources'][name],name
for name,path in [('torch_npu_smollm.py','torch_npu_smollm.py'),('torch_npu_smollm_kv_demo.py','torch_npu_smollm_kv_demo.py'),('npu4000.py','npunlock/python/npunlock/npu4000.py')]:
    assert digest(OUT/'before'/name)==baseline['sources'][path],name
assert digest(OUT/'candidate.py')==sources['torch_npu_smollm.py']==extras['model-probe']['candidate_sha256']==extras['control']['candidate_sha256']
assert manifest==baseline['kernel_manifest']==json.loads((OUT/'before/manifest.json').read_text())
for name,value in manifest['source_sha256'].items():assert digest(FIX/name)==value
assert digest(FIX/'rmsnorm.c')==digest(OUT/'before/rmsnorm.c')
assert digest(FIX/'rmsnorm/rmsnorm-f16.elf')==manifest['elf_sha256']==digest(OUT/'before/rmsnorm-f16.elf')
assert digest(FIX/'rmsnorm/rmsnorm-f16.s')==manifest['assembly_sha256']
assert digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')==baseline['checkpoint_file_sha256']
parity=read('ir-parity');assert parity['status']=='passed' and len(parity['checks'])==56
assert parity['baseline_sha256']==baseline['sources']['torch_npu_smollm.py']
for source,value in parity['sources'].items():assert digest(ROOT/source)==value
assert all(e['xml_weights_equal'] for e in parity['checks'])

assert len(extras['layout-probe']['checks'])==10 and len(extras['model-probe']['checks'])==8
for e in extras['layout-probe']['checks']:
    assert e['status']=='compiled' and digest(OUT/(e['tag']+'.blob'))==e['blob_sha256']
    assert e['complete_rows']==(e['scheme']!='plain')
for e in extras['model-probe']['checks']:
    assert e['status']=='passed' and all(e['bits_equal']) and e['graph_calls']==1 and e['warm_host_copies_and_compiles']==0
    assert len([m for m in e['baseline_dpu']['modes'] if not m['elementwise']])==len([m for m in e['dpu']['modes'] if not m['elementwise']])
    patch=e['patch'];original=(OUT/(e['tag']+'-original.blob')).read_bytes();modified=(OUT/(e['tag']+'-patched.blob')).read_bytes()
    assert hashlib.sha256(original).hexdigest()==patch['original_sha256']
    assert hashlib.sha256(modified).hexdigest()==patch['patched_sha256']
    assert patch['kernel_elf_sha256']==manifest['elf_sha256'] and patch['accumulation']=='fp32_neumaier'
    start=patch['code_offset'];end=start+patch['code_bytes']
    assert original[:start]==modified[:start] and original[end:]==modified[end:] and len(original)==len(modified)
    assert patch['code_bytes']==5136 and patch['window_bytes']==21216 and patch['shape']==[2,576]
    assert all(c['elements']%576==0 for c in patch['contracts'])
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
        assert e['graph_calls']==e['num_layers']*2
        assert e['previous_graph_calls']==e['num_layers']*(2 if e['batch']>=2 else 4)
assert len(extras['ownership']['checks'])==16
for e in extras['ownership']['checks'][:-2]:
    assert e['graph_calls']==2 and e['perturbation_isolated'] and e['old_snapshot_unchanged'] and e['cold_decode_host_copies']==0
    assert e['held_output_unchanged'] and e['warm_host_copies_and_compiles']==0
    assert all(c['relative_l2']<.003 and c['max_scaled_error']<.01 for sample in e['solo_comparisons'] for c in sample)
assert extras['ownership']['checks'][-2]['batch_sequence']==[1,2,1,8,3,1]
assert extras['ownership']['checks'][-2]['all_bits_unchanged'] and extras['ownership']['checks'][-2]['gamma_shared_between_b1_b2']
assert extras['ownership']['checks'][-1]['public_outputs_and_kv_independent'] and extras['ownership']['checks'][-1]['graph_calls']==12
assert len(extras['control']['timings'])==len(extras['control']['checks'])==8
assert extras['control']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
assert all(e['all_bits_equal'] for e in extras['control']['checks'])
for e in extras['control']['timings']:
    assert e['batch']==1 and e['rounds']==41 and e['same_model_input_cache'] and e['warm_host_copies_and_compiles']==0
    assert all(len(v)==82 for v in e['samples_ms'].values()) and len(e['round_pairs_ms'])==41
    assert e['graph_calls']==dict(previous=e['num_layers']*4,current=e['num_layers']*2)
    assert e['allocations_per_decode']==dict(previous=e['num_layers']*5,current=e['num_layers']*5)
for e in cases['profile']['timings']:assert sum(e['native_calls'].values())==e['num_layers']*2
for e in cases['performance']['timings']:
    for name,calls in e['graph_calls'].items():
        if name.endswith('_decode') and not name.startswith('cpu'):
            assert calls==e['num_layers']*(4 if e['shape'][0]==1 and name.startswith('previous_') else 2)

summary=dict(status='passed',date='2026-10-05',sources=sources,kernel_manifest=manifest,
    checkpoint=cases['numeric']['checkpoint'],checkpoint_file_sha256=baseline['checkpoint_file_sha256'],packages=cases['numeric']['packages'],
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),
    max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    controls=extras['control']['timings'],profiles=cases['profile']['timings'],performance=cases['performance']['timings'],
    unchanged_batch_ir_records=len(parity['checks']),ir_parity_sha256=digest(OUT/'ir-parity.json'),
    total_npu_calls=sum(c['final_stats']['npu_calls'] for c in {**cases,**extras}.values()),
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
control_table='\n'.join(f"| {e['num_layers']} / {e['prefix_tokens']} | {e['mode']} | {e['median_ms']['previous']:.3f} | {e['median_ms']['current']:.3f} | {e['latency_reduction_percent']:.2f}% | {e['positive_rounds']}/41 |" for e in summary['controls'])
perf_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} | {e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | {e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in summary['performance'])
profile_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['mean_stage_percent']['norm_attention']:.1f}% | {e['mean_stage_percent']['norm_ffn']:.1f}% | {sum(e['native_calls'].values())} |" for e in summary['profiles'])
check_table='\n'.join(f"| {name} | {c['checks']} | {c['timings']} | {c['npu_calls']} |" for name,c in summary['cases'].items())
report=f'''# B1 单序列：完整行归一化与投影融合

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

冻结 SmolLM2 层的单 token B1 forward/prefill/decode 已默认融合，每层暖 decode 从 4 次提交降到 2 次；四层从 16 次降到 8 次。B2..B8 延续上一阶段的融合，多 token 前缀保留原路径。每层暖分配仍为 5 次，公开 API、输入输出 batch 与 KV 大小保持原样。

## 实现与布局验证

仅在图内归一化入口将 1×576 输入复制为 2×576，执行原有 FP32 SHAVE RMSNorm，再切回第一行，送入原有 B1 投影。gamma 使用已有的两行驻留广播参数。第二行增加归一化与相应调度/数据移动工作，不执行第二份模型 attention/FFN 或 KV 更新，也不暴露额外输出；这不是多用户 batch。

10 组 compile-only 探测对照了原始行、只复制归一化输入、保留 norm 输出、复制完整投影及保留投影输出，并比较默认四 tile 与一 tile。原始 B1 分别拆成 80/64 或 288 元素片段；只复制归一化输入并切回一行，就能保留完整 576 元素，无需辅助输出或完整模型复制。

真实 FP16/W8A16 的 attention-forward、attention-prefill、attention-decode、FFN 共 8 组全部逐位一致；每组非 elementwise DPU invariant 数量与原 B1 分图相同。公开图中投影与动态注意力保持 B1，代码窗口以外的 blob 字节逐位不变；完整行、CMX 边界、精度和别名校验继续使用上一阶段的接口。

没有重建 SHAVE ELF 或 native DLL/PYD，未更改 FP32 Neumaier 补偿归约、epsilon、舍入顺序或通用严格 FP32 算子。运行仍无需 OpenVINO Python/runtime，仍使用 Intel 驱动图编译器。

## 同实例 B1 性能对照

同一模型、输入、gamma 与原 KV，仅切换 B1 融合开关。41 轮 ABBA/BAAB，每策略 82 样本，所有输出/KV 位模式一致；暖显式上传/下载/编译为零，两路径分配均为 5 次/层。

| 层数 / 前缀位置 | 模式 | 原分图 ms | B1 融合 ms | 中位延迟降幅 | 更快轮数 |
|---|---|---:|---:|---:|---:|
{control_table}

计时包含绑定、提交、设备工作及 fence，排除初始准备、编译和显式传输；内部 DMA 未计入复制计数。未锁定桌面负载、频率或内存状态，初步样本保存在 control-preliminary.*，最终结论使用 control.json。

## 独立实例及 CPU 对照

每组 21 轮轮转各路径，CPU 为 FP16、8 线程；下表为 decode ms。B4 的旧/新版本均执行上一阶段的批量融合，是未改变该路径的对照。B2..B8 的 56 组 FP16/W8A16 attention-forward/prefill/decode/FFN IR XML 与权重字节完全相同；部分未改变路径的计时仍出现变慢，说明不能据此归因于 B1 修改。独立实例与同实例具有不同的分配/缓存条件，保留全部测量结果。

| 层数 / B | FP16 旧→新 | W8A16 旧→新 | CPU FP16 |
|---|---:|---:|---:|
{perf_table}

以上为冻结层片段时间，不能直接换算为完整 LLM 生成速度，也未测功耗。支持范围仍是 B≤8、相同前缀长度、总位置≤64；不增加训练、padding mask 或完整模型支持。

## 阶段比例

21 轮 traced 平均值，归一化包含在 attention/FFN 阶段；旧独立 RMSNorm bucket 为零不表示归一化没有成本。

| 层数 / B | 模式 | norm + attention | norm + FFN | 原生调用数 |
|---|---|---:|---:|---:|
{profile_table}

## 正确性、快照与追溯

{summary['numerical_records']} 条真实权重数值记录、{summary['fixed_input_cache_tensor_checks']} 个固定输入 K/V 对照、{summary['continuous_decode_steps']} 次连续位置解码通过。24 条新旧 cached decode、8 条普通 forward 与 8 组混合生成通过逐位/token 检查。最大执行相对 L2 {summary['max_execution_relative_l2']*100:.4f}%，scaled error {summary['max_execution_scaled_error']*100:.4f}%；原门槛 0.3% / 1% 与量化质量门槛 3% 保留。

14 组隔离检查覆盖 B1/B2/B3/B8、前缀 1/7/32/63、冷 decode 零主机复制、输入扰动、旧 KV 与保留输出。一个实例按 B1→B2→B1→B8→B3→B1 切换，B1/B2 共用 gamma 地址，6 组输出/KV 快照保持不变。两个调用者对同一 B1 缓存执行 6 次 warm decode，共 12 图，公开输出及 K/V/position 各自独立、位模式一致；执行仍由 NPU 锁串行化。

未修改底层载体/精度 helper，上一阶段的 9 项融合保护、16 项 ELF/载体与通用快速精度回归证据保留在 rmsnorm-fusion/。本轮设备回归不放宽输入保护、数值或无主机复制门槛。

| 成功进程/用例 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{check_table}

成功设备进程共 {summary['total_npu_calls']:,} 次记录内 NPU 调用，全部健康关闭、存活分配归零。首次探测脚本误用一行 gamma 绑定两行图参数，被 native 缓冲区尺寸校验拒绝；修正脚本绑定后通过，失败记录在 failures/probe-gamma-binding/，未放宽保护。

模型 safetensors SHA：`{summary['checkpoint_file_sha256']}`。完整 sources、kernel manifest、before/after/diff 与原始/修改 blob 在本目录。

```powershell
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_b1_layout_probe.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_b1_model_probe.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_b1_ownership.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','rmsnorm-b1-fusion','--case','numeric'
.\\run_torch_npu.ps1 -Script .\\torch_npu_b1_fusion_benchmark.py
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_rmsnorm_b1.py
```
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print({k:summary[k] for k in ['status','numerical_records','fixed_input_cache_tensor_checks','continuous_decode_steps','total_npu_calls']})
