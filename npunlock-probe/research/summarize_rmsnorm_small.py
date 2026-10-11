"""Audit the RMSNorm kernel, scratch ownership and controlled timings."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-small'
FIX=ROOT/'npunlock/tests/fixtures/npu4000'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['probe','numeric','boundary','guards','positions','hybrid','paired','regression','profile','performance']
cases={name:json.loads((OUT/f'{name}.json').read_text()) for name in names}
extras={name:json.loads((OUT/f'{name}.json').read_text()) for name in ['diagnostic','workspace','control','control-initial','restoration']}
baseline=json.loads((OUT.parent/'smollm-kv-batch/summary.json').read_text())
sources=cases['numeric']['sources'];manifest=cases['numeric']['kernel_manifest']
for name,case in cases.items():
    assert case['sources']==sources and case['kernel_manifest']==manifest and case['checkpoint']==baseline['checkpoint'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
    assert case['final_stats']['hardware_device']=='8086:643e' and case['final_stats']['driver_version']==1005540,name
for name,value in sources.items():
    assert digest(ROOT/name)==value,name
    if name not in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:assert value==baseline['sources'][name],name
for name in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:
    assert digest(OUT/'before'/name)==baseline['sources'][name]
for name,case in extras.items():
    for source,value in case.get('sources',{}).items():
        path=OUT/'control-initial.py' if name=='control-initial' and source=='torch_npu_rmsnorm_benchmark.py' else ROOT/source
        assert digest(path)==value,(name,source)
build=json.loads((OUT/'build-manifest.json').read_text())
assert digest(ROOT/'npunlock-probe/research/build_rmsnorm_small.py')==build['source_sha256']
assert build['baseline_manifest']==baseline['kernel_manifest']==json.loads((OUT/'before/manifest.json').read_text())
assert digest(OUT/'before/rmsnorm-f16.elf')==baseline['kernel_manifest']['elf_sha256']
assert digest(OUT/'before/rmsnorm.c')==baseline['kernel_manifest']['source_sha256']['rmsnorm.c']
for name,value in manifest['source_sha256'].items():assert digest(FIX/name)==value
prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
generated=prefix+b'\n'.join((FIX/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
    if name!='rmsnorm.c' else (FIX/name).read_bytes() for name in ['arithmetic.c','resident_activation.c','rmsnorm.c'])
assert hashlib.sha256(generated).hexdigest()==manifest['generated_source_sha256']
assert digest(FIX/'rmsnorm/rmsnorm-f16.elf')==manifest['elf_sha256']==build['kernels']['reduce4']['elf_sha256']
assert digest(FIX/'rmsnorm/rmsnorm-f16.s')==manifest['assembly_sha256']
assert digest(FIX/'rmsnorm.c')==digest(OUT/'reduce4.c')
for name in ['reduce4','copy']:
    for suffix,key in [('c','source_sha256'),('s','assembly_sha256'),('elf','elf_sha256')]:assert digest(OUT/f'{name}.{suffix}')==build['kernels'][name][key]
assert len(extras['diagnostic']['checks'])==39 and all(e['old_new_bit_differences']==0 and e['copy_bits_equal'] for e in extras['diagnostic']['checks'])
for rows,carriers in extras['diagnostic']['carriers'].items():
    assert len({e['layout_sha256'] for e in carriers})==1
    for e in carriers:assert digest(OUT/f"carrier-{rows}-{e['kernel']}.blob")==e['patched_sha256']
assert all(e['all_output_cache_bits_equal'] for e in cases['paired']['checks'])
assert all(e['old_new_bits_equal'] for e in cases['regression']['checks'])
assert all(e['generated_tokens_equal'] for e in cases['hybrid']['checks'])
entries=[e for name in ['probe','numeric','boundary','positions'] for e in cases[name]['checks'] if 'execution' in e]
cache_entries=[e for entry in entries for e in entry.get('cache_local_reference',[])]
assert all(e['execution']['relative_l2']<.003 and e['execution']['max_scaled_error']<.01 for e in entries)
assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in cache_entries)
assert all(e['quality_official']['relative_l2']<.03 for e in entries if 'quality_official' in e)
for e in cases['paired']['checks']:
    if e['warmed']:assert e['graph_calls']==e['previous_graph_calls']==e['num_layers']*4
assert len(extras['control']['timings'])==len(extras['control']['checks'])==14
for name in ['control','control-initial']:
    assert extras[name]['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
    assert all(e['all_bits_equal'] for e in extras[name]['checks'])
    for e in extras[name]['timings']:
        assert e['rounds']==41 and e['same_model_input_cache'] and e['warm_host_copies_and_compiles']==0
        assert all(len(values)==82 for values in e['samples_ms'].values()) and len(e['round_pairs_ms'])==41
        assert set(e['graph_calls'].values())=={e['num_layers']*4}
        assert e['allocations_per_decode']==dict(previous=e['num_layers']*7,kernel_only=e['num_layers']*7,current=e['num_layers']*5)
for e in cases['profile']['timings']:assert sum(e['native_calls'].values())==e['num_layers']*4
for e in cases['performance']['timings']:
    for name,calls in e['graph_calls'].items():
        if name.endswith('_decode') and not name.startswith('cpu'):assert calls==e['num_layers']*4
assert digest(OUT.parent/'smollm-kv-batch/probe.json')==extras['restoration']['restored_probe_sha256']
assert digest(OUT/'replay-batch-probe.py')==extras['restoration']['replay_script_sha256']
assert digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')==baseline['checkpoint_file_sha256']
summary=dict(status='passed',date='2026-10-05',sources=sources,kernel_manifest=manifest,checkpoint=cases['numeric']['checkpoint'],
    checkpoint_file_sha256=baseline['checkpoint_file_sha256'],build=build,packages=cases['numeric']['packages'],
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),
    max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    controls=extras['control']['timings'],initial_controls=extras['control-initial']['timings'],profiles=cases['profile']['timings'],
    performance=cases['performance']['timings'],diagnostic=extras['diagnostic']['timings'],workspace=extras['workspace']['checks'],
    total_npu_calls=sum(c['final_stats']['npu_calls'] for c in {**cases,**extras}.values()),
    cases={name:dict(checks=len(c.get('checks',[])),timings=len(c.get('timings',[])),npu_calls=c['final_stats']['npu_calls']) for name,c in {**cases,**extras}.items()})
changed={'torch_npu_smollm.py':ROOT/'torch_npu_smollm.py','torch_npu_smollm_kv_demo.py':ROOT/'torch_npu_smollm_kv_demo.py',
    'rmsnorm.c':FIX/'rmsnorm.c','manifest.json':FIX/'rmsnorm/manifest.json','rmsnorm-f16.elf':FIX/'rmsnorm/rmsnorm-f16.elf',
    'rmsnorm-f16.s':FIX/'rmsnorm/rmsnorm-f16.s','TORCH_NPU_README.md':ROOT/'TORCH_NPU_README.md','NPU4000.md':ROOT/'npunlock/docs/NPU4000.md'}
(OUT/'after').mkdir(exist_ok=True);diff=[]
for name,path in changed.items():
    shutil.copyfile(path,OUT/'after'/name)
    if path.suffix not in ['.elf']:
        diff.extend(difflib.unified_diff((OUT/'before'/name).read_text(encoding='utf-8').splitlines(True),path.read_text(encoding='utf-8').splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(OUT/'changes.diff').write_text(''.join(diff),encoding='utf-8')
summary['after_sha256']={name:digest(OUT/'after'/name) for name in changed};summary['diff_sha256']=digest(OUT/'changes.diff')
summary['audit_sha256']=digest(Path(__file__))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
norm_table='\n'.join(f"| {e['rows']} | {e['median_ms']['baseline-fresh']:.3f} | {e['median_ms']['reduce4-fresh']:.3f} | {e['median_ms']['reduce4-reuse']:.3f} | {e['median_ms']['copy-reuse']:.3f} |" for e in summary['diagnostic'])
control_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['median_ms']['previous']:.3f} | {e['median_ms']['kernel_only']:.3f} | {e['median_ms']['current']:.3f} | {e['latency_reduction_percent']:.2f}% | {e['positive_rounds']}/41 |" for e in summary['controls'])
profile_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['mean_stage_percent']['rmsnorm1']+e['mean_stage_percent']['rmsnorm2']:.1f}% | {e['mean_stage_percent']['attention']:.1f}% | {e['mean_stage_percent']['ffn']:.1f}% | {e['mean_native_total_ms']/e['mean_traced_ms']*100:.1f}% |" for e in summary['profiles'])
perf_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} | {e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | {e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in summary['performance'])
check_table='\n'.join(f"| {name} | {c['checks']} | {c['timings']} | {c['npu_calls']} |" for name,c in summary['cases'].items())
report=f'''# 小行数 RMSNorm：向量归约与内部工作区复用

2026-10-05，本机 Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

已接入每次解码/forward/prefill 的内部 RMSNorm 工作区复用，并将 FP16 解码、FP32 平方改为每次 4 元素。保持原有 Neumaier FP32 补偿求和顺序、epsilon、两次 FP16 舍入及特殊值转换路径；关闭 FMA。生产 ELF 与诊断候选一致，代码段由 5312 降到 5136 字节，未改载体的 DMA、barrier 或任务分配。

每个 layer norm 名称最多保留一块工作区，行数变化时替换，最多合计 1.125 MiB/层。公开输出与 KV 由后续图独立分配，仍是不可变快照；私有 `_norm_device` 默认返回新输出。串行 512 行的分块路径保持独立输出，不复用单块工作区。别名拒绝与设备健康检查保留；非连续别名可能先执行只读 materialize DMA，再拒绝计算。

每层暖 decode 仍提交 4 图，分配由 7 次降到 5 次。未修改 native DLL/PYD、通用 FP32 算子、矩乘或精度策略；没有添加 OpenVINO Python/runtime 依赖，仍使用 Intel 驱动图编译器。模型范围仍为冻结的 SmolLM2-135M FP16/W8A16 层片段，B≤8、相同前缀长度、总位置≤64；不增加训练或完整 LLM 支持。

## 同载体诊断

同一编译载体只替换代码窗口，3 种内核窗口以外的字节 SHA 完全相同；保留原始 blob。复制内核保持相同输入/输出绑定与任务布局，作为固定成本对照。每组轮转 6 条件、41 轮，零暖上传/下载/编译。

| 行数 | 旧内核/新分配 ms | 新内核/新分配 ms | 新内核/复用 ms | 复制/复用 ms |
|---|---:|---:|---:|---:|
{norm_table}

小行数仍有较大的绑定、提交、DMA、设备执行及 fence 综合成本；复制对照也包含这些环节和 SHAVE 复制工作，不能等同于纯 IO 或空提交。8 行起的时间跳变来自既有 4 tile / 8 SHAVE 任务调度，不能按元素量线性外推。

## 同模型、同输入、同缓存对照（主要性能证据）

在同一个模型上仅切换归一化策略：旧内核新分配／新内核新分配／新内核复用。旧 ELF 预先读入内存；所有策略共用 attention/FFN、输入、参数与 KV。41 轮采用轮转后的 ABC/CBA 顺序，每策略 82 样本；按每轮两样本平均计算胜出轮数。计时区间内不包含策略切换，零显式上传/下载/编译。

| 层数 / B | 模式 | 旧 ms | 只换内核 ms | 最终 ms | 最终延迟降幅 | 最终更快轮数 |
|---|---|---:|---:|---:|---:|---:|
{control_table}

单独换内核的微小收益会波动，并非每组都改善。最初 6 组的有效对照下降 4.5%–8.1%；其代码与样本保存在 `control-initial.*`。随后因独立模型对照出现变慢，扩大到上表 14 组；所有组保留逐位输出/KV 比较、原始样本和分配计数。

## 分别创建模型与 CPU 对照

每组 21 轮与 CPU FP16（8 线程）交错，以下均为 decode ms。部分独立实例的 FP16 对照变慢，说明结果仍受运行条件、实例/缓存与分配状态影响；本轮不承诺普遍加速。所有这些小模型片段中 CPU 仍更快。

| 层数 / B | FP16 旧→新 | W8A16 旧→新 | CPU FP16 |
|---|---:|---:|---:|
{perf_table}

两类性能测试均为暖片段时间，排除初始编译/准备和片段显式传输，包含提交、隐式 DMA、设备执行及同步；未锁定桌面负载、频率或内存状态。不代表全 LLM 端到端速度、节能收益或长期稳定性。

## 当前阶段比例

21 轮 plain/traced 交错；比例来自 traced 平均值，原生执行包络嵌套在各阶段中，不能与阶段比例相加。

| 层数 / B | 模式 | 两次 RMSNorm | attention | FFN | 原生执行包络 |
|---|---|---:|---:|---:|---:|
{profile_table}

RMSNorm 仍占约 34%–44%，其余主要为 attention 与 FFN。原生执行包络包含绑定/提交/设备/fence，不是独立内核时间。下一步值得先做图内 RMSNorm→投影融合可行性探测，尝试减少两次提交和中间 DMA，同时保留 FP32 归约精度与内存保护。

## 验证及追溯

39 组内核检查与旧版本逐位相同，覆盖全部 63,488 个有限 FP16 编码的混合行、零/非规格化数/最大有限数、随机与真实 embedding、两个真实 gamma。工作区检查 24 项涵盖变化输入/形状、512 行串行分块、别名拒绝与恢复、保留 4 个公开输出/3 个旧 KV，以及两个调用者的 6 次交错 decode。

{summary['numerical_records']} 条真实模型数值记录、{summary['fixed_input_cache_tensor_checks']} 个固定输入 K/V 对照、{summary['continuous_decode_steps']} 次连续位置解码、24 条新旧 cached decode 逐位比较、8 条普通 forward 逐位比较、8 组混合生成全部通过。执行最大相对 L2 {summary['max_execution_relative_l2']*100:.4f}%，scaled error {summary['max_execution_scaled_error']*100:.4f}%；原门槛 0.3% / 1% 保留，量化质量门槛 3% 保留。

| 进程/用例 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{check_table}

共 {summary['total_npu_calls']:,} 次记录内 NPU 调用，所有进程健康关闭，存活分配归零。一次错误输出目录覆盖上一轮 probe 数值文件，已用 SHA 验证的旧代码/旧 ELF 重跑恢复（`restoration.json`）；上一轮性能数据未改动。模型 safetensors SHA 为 `{summary['checkpoint_file_sha256']}`；完整 sources、kernel manifest、原始结果及 before/after/diff 见本目录。审计工具验证现有证据，不重跑硬件。

```powershell
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\build_rmsnorm.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_small_probe.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\rmsnorm_workspace_probe.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','rmsnorm-small','--case','numeric'
.\\run_torch_npu.ps1 -Script .\\torch_npu_rmsnorm_benchmark.py
.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_rmsnorm_small.py
```
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
print({key:summary[key] for key in ['status','numerical_records','fixed_input_cache_tensor_checks','continuous_decode_steps','total_npu_calls']})
