"""Audit folded-batch decode sources, correctness and paired timing evidence."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

names=['probe','numeric','boundary','guards','positions','hybrid','paired','regression','profile_before','profile','performance']
cases={name:json.loads((OUT/f'{name}.json').read_text(encoding='utf-8')) for name in names}
extras={name:json.loads((OUT/f'{name}.json').read_text(encoding='utf-8')) for name in ['isolation','control','layout-probe','lowering','legacy']}
baseline=json.loads((OUT.parent/'smollm-kv-profile/summary.json').read_text(encoding='utf-8'))
sources=cases['regression']['sources'];changed=['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']
for name,case in cases.items():
    assert case['sources']==sources and case['kernel_manifest']==baseline['kernel_manifest'] and case['checkpoint']==baseline['checkpoint'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
for name,value in sources.items():
    assert digest(ROOT/name)==value,name
    if name not in changed:assert value==baseline['sources'][name],name
for name in changed:assert digest(OUT/'before'/Path(name).name)==baseline['sources'][name],name
for name in ['control','isolation','legacy']:
    for source,value in extras[name]['sources'].items():assert digest(ROOT/source)==value,source
assert extras['control']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
assert extras['layout-probe']['candidate_sha256']==digest(OUT/'candidate.py')
assert sum(e.get('status')=='compiled' for e in extras['layout-probe']['checks'])==12
assert sum(e['stage']=='execute' for e in extras['layout-probe']['checks'])==6
assert extras['lowering']['rejected_source_sha256']==baseline['compiler_failures']['batch-head-fold']['source_sha256']
failures={}
for name in ['single-token-prefill','odd-batch-bit-parity']:
    path=OUT/'failures'/name;failure=json.loads((path/'isolation.json').read_text(encoding='utf-8'))
    assert failure['status']=='failed' and not failure['failure_stats']['context_failed']
    assert digest(path/'smollm_batch_isolation.py')==failure['sources']['npunlock-probe/research/smollm_batch_isolation.py']
    adapter=path/'torch_npu_smollm.py' if name=='single-token-prefill' else ROOT/'torch_npu_smollm.py'
    assert digest(adapter)==failure['sources']['torch_npu_smollm.py']
    failures[name]=dict(error=failure['error'],sources=failure['sources'])
assert digest(OUT.parent/'smollm-kv-profile/after/torch_npu_smollm.py')==baseline['sources']['torch_npu_smollm.py']
assert digest(OUT/'before/torch_npu_assembly_benchmark.py')==baseline['assembly_sources']['torch_npu_assembly_benchmark.py']
assert extras['legacy']['checks']==[dict(batch=2,graph_calls=[8,6],all_bits_equal=True)]
build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
assert build==baseline['extension_build'] and digest(ROOT/'npunlock-probe/torch-device/backend.cpp')==build['source_sha256']
model_hash=digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors');assert model_hash==baseline['checkpoint_file_sha256']
entries=[e for name in ['probe','numeric','boundary','positions'] for e in cases[name]['checks'] if 'execution' in e]
cache_entries=[e for entry in entries for e in entry.get('cache_local_reference',[])]
assert all(e['all_output_cache_bits_equal'] for e in cases['paired']['checks'])
assert all(e['generated_tokens_equal'] for e in cases['hybrid']['checks'])
calls={}
for e in cases['paired']['checks']:
    if e['warmed']:
        value=dict(previous=e['previous_graph_calls']//e['num_layers'],current=e['graph_calls']//e['num_layers'])
        assert calls.setdefault(e['batch'],value)==value
assert calls=={1:dict(previous=4,current=4),2:dict(previous=6,current=4),4:dict(previous=8,current=4),8:dict(previous=12,current=4)}
assert len(extras['isolation']['checks'])==16
assert all(e['graph_calls']==4 and e['perturbation_isolated'] and len(e['solo_bits_equal_samples'])==e['batch'] for e in extras['isolation']['checks'])
for name in ['profile_before','profile']:
    for e in cases[name]['timings']:
        assert e['rounds']==21 and e['warm_host_copies_and_compiles']==0
        assert sum(e['native_calls'].values())==e['num_layers']*calls[e['batch']]['previous' if name=='profile_before' else 'current']
assert cases['profile_before']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
controls=[]
assert len(extras['control']['timings'])==len(extras['control']['checks'])==6
assert all(e['all_bits_equal'] for e in extras['control']['checks'])
for e in extras['control']['timings']:
    assert e['rounds']==41 and e['same_model_input_cache'] and e['warm_host_copies_and_compiles']==0
    assert all(len(v)==82 for v in e['samples_ms'].values()) and len(e['round_pairs_ms'])==41
    for version in ['previous','current']:assert e['graph_calls'][version]==e['num_layers']*calls[e['batch']][version]
    controls.append({k:v for k,v in e.items() if k not in ['samples_ms','round_pairs_ms']})
performance=[]
for e in cases['performance']['timings']:
    for mode in ['fp16','w8a16']:
        assert e['graph_calls'][mode+'_decode']==e['num_layers']*4
        assert e['graph_calls']['previous_'+mode+'_decode']==e['num_layers']*calls[e['shape'][0]]['previous']
    performance.append({k:v for k,v in e.items() if k!='samples_ms'})
summary=dict(status='passed',date='2026-10-05',sources=sources,extension_build=build,kernel_manifest=cases['probe']['kernel_manifest'],
    checkpoint=cases['probe']['checkpoint'],checkpoint_file_sha256=model_hash,packages=cases['probe']['packages'],
    total_npu_calls=sum(c['final_stats']['npu_calls'] for c in {**cases,**extras}.values()),
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),paired_bits_equal_records=len(cases['paired']['checks']),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),
    max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    warm_calls_per_layer=calls,profiles={name:cases[name]['timings'] for name in ['profile_before','profile']},
    controls=controls,performance=performance,isolation=extras['isolation']['checks'],lowering=extras['lowering']['checks'],failures=failures,
    solo_max_relative_l2=max(e['relative_l2'] for g in extras['isolation']['checks'] for r in g['solo_comparisons'] for e in r),
    solo_max_scaled_error=max(e['max_scaled_error'] for g in extras['isolation']['checks'] for r in g['solo_comparisons'] for e in r),
    cases={name:dict(checks=len(c['checks']),timings=len(c.get('timings',[])),npu_calls=c['final_stats']['npu_calls'],final_live_buffers=0)
        for name,c in {**cases,**extras}.items()})
checks='\n'.join(f"| {name} | {v['checks']} | {v['timings']} | {v['npu_calls']} |" for name,v in summary['cases'].items())
control_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['median_ms']['previous']:.3f} → {e['median_ms']['current']:.3f} | "
    f"{e['latency_reduction_percent']:.2f}% | {e['median_paired_delta_ms']:.3f} | {e['positive_rounds']} / 41 | "
    f"{e['graph_calls']['previous']} → {e['graph_calls']['current']} |" for e in controls)
profile_table=[]
for name in ['profile_before','profile']:
    for e in summary['profiles'][name]:
        p=e['mean_stage_percent']
        profile_table.append(f"| {'旧' if name=='profile_before' else '新'} | {e['mode']} | {e['num_layers']} / {e['batch']} | {e['median_plain_ms']:.3f} | "
            f"{p['rmsnorm1']+p['rmsnorm2']:.1f}% | {p['attention']:.1f}% | {p['assembly']:.1f}% | {p['ffn']:.1f}% | "
            f"{e['mean_native_total_ms']/e['mean_traced_ms']*100:.1f}% |")
perf_table='\n'.join(f"| {e['num_layers']} / {e['shape'][0]} | {e['prefix_tokens']}→{e['shape'][1]} | {e['median_ms']['fp16_decode']:.3f} | "
    f"{e['median_ms']['w8a16_decode']:.3f} | {e['median_ms']['w8a16_full_prefix']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in performance)
lowering_table='\n'.join(f"| {e['stage']} | {e['output_shapes']} | {e['status']} |" for e in summary['lowering'])
report=f'''# SmolLM2 整批 NPU decode：保持 N=1 的内部布局

2026-10-05，Lunar Lake 8086:643e，驱动 {cases['probe']['final_stats']['driver_version']}，编译器 8.3。
在 FP16／W8A16 冻结 SmolLM2-135M 片段中，用一次图提交计算整批注意力，替代逐样本提交及拼接。

## 编译与实现

上轮仅在注意力 MatMul 前折叠 batch/head 的方案失败，QKV/head/RoPE/cache 中间仍出现 N=B。
本轮在形成 heads 之前就保持 N=1：投影／残差 `(1,B,576)`，Q `(1,B*9,1,64)`，
K/V cache `(1,B*3,64,64)`。GQA 用全局头索引逐个重复三次，每个样本的三组 KV 只服务自己的九组 Q。
Softmax 沿最后的 64 个槽计算；输出再恢复原 `(B,1,576)`、`(B,3,64,64)`。
这是布局改写，不把不同样本当成互相可见的 token，也不改变 RoPE、mask、算术精度或 KV 存储约定。

初始候选 FP16/W8A16 的 B2/B4/B8 静态投影和整批 decode 共 12 次编译通过，六组随机输入执行通过。
旧失败候选的 FP16 B2 图逐段截断、独立编译结果如下；编译失败仍是 `ConvertBatchedLayerTo1N`／Level Zero 0x78000007，
诊断 JSON/log 与 XML/bin 留存。截断会改变融合机会，只能缩小触发范围，不能证明驱动内部具体哪个变换有 bug。

| 截断阶段 | 输出形状 | 编译结果 |
|---|---|---|
{lowering_table}

静态投影及 heads 截断能编译，加入 RoPE 的截断就失败；因此错误不需要动态注意力 MatMul 即可触发。
保持 N=1 的前移布局通过该组合，未修改驱动／编译器本身。

正式路径删除动态 sample 标量、逐样本 Python 循环和最终三输出拼接；直接以五输入／四输出图运行整批注意力。
单 token 批量 prefill／普通 forward 的注意力也采用相同布局，修复此前会触发相同编译错误的短前缀。
其余 prefill／普通 forward 和 batch=1 保留原计算路径；prefill 不再预上传无用 sample 标量。
暖 decode 每层 B1/B2/B4/B8 为 4/6/8/12 → 4/4/4/4：两次 SHAVE RMSNorm、一次 DPU/SHAVE 注意力、一次 FFN。
首次 decode 还可能有两个 gamma 广播，不计入暖调用。编译器安排内部 DPU/SHAVE/DMA 任务，
单图整批支持不等于已测得所有样本在硬件上同时计算，也不等于多层之间可以并行。

## 正确性与生命周期

{len(summary['cases'])} 个串行进程完成，累计 {summary['total_npu_calls']:,} 次 NPU 调用。
所有最终进程 shutdown 后 live_buffers=0、context_failed=false；lowering 的预期编译失败未发生设备推理异常。

| case | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{checks}

{summary['numerical_records']} 条数值、{summary['fixed_input_cache_tensor_checks']} 项固定真实输入 K/V、{summary['continuous_decode_steps']} 步连续解码通过。
24 组新旧输出／缓存／位置逐位一致；8 组普通 forward 与前轮逐位一致；8 组混合生成 token 与 CPU 一致。
22 项模型缓存错误在执行前拒绝，inference_mode、快照复用／不可变、零空槽及 63→64／完整 64 prefill 通过。
附加 B2..B8 × 两种精度、前缀 1/7/32/63 的 16 组隔离测试：整批与独立 B1 的输出／已用 KV 均满足原执行门槛，
并记录逐位比较和每个样本的误差；不保证与单独 B1 的全部输出逐位相同。prefill 与普通 forward 逐位相同并通过 CPU 数值门槛。
这 86 个独立样本比较的已用 K/V 全部逐位相同；输出存在小差异，最大独立 B1 对照 L2 {summary['solo_max_relative_l2']*100:.4f}%，
最大归一化误差 {summary['solo_max_scaled_error']*100:.4f}%。这是不同 batch 形状的观测，不据此承诺通用逐位等价。
只改一个样本的解码输入，其他样本的输出和 KV 保持逐位不变。暖调用无显式主机复制／新增编译。

执行门槛仍为 L2<0.3%、`abs(error)/(abs(reference)+RMS)<1%`；最大执行 L2 {summary['max_execution_relative_l2']*100:.4f}%，
最大归一化误差 {summary['max_execution_scaled_error']*100:.4f}%；官方 CPU 浮点参考质量 L2 最大 {summary['max_quality_relative_l2_official']*100:.4f}%，仍满足 3% 门槛。
量化误差与执行误差分开记录；保留旧固定 atol 不匹配数，不据其放宽新门槛。

## 同模型配对性能

复用同一模型、输入、初始 KV 快照、norm 参数及原生 runtime；只临时切换图构造／执行方法。
旧／新各自缓存编译键，预热结束后零新增编译。41 轮交替 ABBA/BAAB，每轮各两次，共每种策略 82 次。
切换 Python 方法／字典发生在计时外。六组输出／缓存／位置逐位一致后才计时。
排除初始编译、准备、上传下载，包含 Python、分配／释放、设备工作、提交及 fence 等待。
负载／频率未锁定，B1 两组是无计算变化的控制；逐轮差值中位数和胜出轮数一并保留。

| 层数 / B | 精度 | 旧→新中位 ms | 延迟减少 | 每轮旧减新中位 ms | 新更快轮数 | 整段提交旧→新 |
|---|---|---:|---:|---:|---:|---:|
{control_table}

四层 B4 的两种精度延迟都减少约 54%，单层 B8 减少约 74%；四个批量组新路径均在 41/41 轮更快。
B1 无计算变化，对照分别约 +1.7%／-0.3%，不视为 B1 加速证据；尚未控制桌面负载／频率。

另有 21 轮旧／新／重算前缀／官方 CPU FP16 交错计时；以下只展示当前路径和 CPU。
CPU 为八线程官方 eager 实现，包含重复 decode 后 DynamicCache crop，未代表最优 CPU 或全模型端到端性能。

| 层数 / B | 前缀→总长 | 当前 FP16 ms | 当前 W8A16 ms | W8 重算前缀 ms | CPU FP16 decode ms |
|---|---|---:|---:|---:|---:|
{perf_table}

## 分段计时与范围

各组原始／带包装调用交替 21 轮，沿用前轮计时器；旧／新 profile 分进程，绝对时间只作诊断。
阶段占比使用带包装均值，毫秒列使用不带包装中位。原生包络包含绑定、提交、设备执行及 fence，
与阶段时间重叠，不能再次相加或解释为纯计算／IO；健康状态查询不是额外硬件 fence。

| 版本 | 精度 | 层数 / B | 无包装中位 ms | 两次 RMSNorm | 注意力 | 拼接 | FFN | 原生包络占总时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(profile_table)}

本轮 W8A16 分段观测中，四层 B4 两次 RMSNorm 已占约 44%，注意力约 28%，独立拼接为零；
接下来应优先处理小行数 RMSNorm 和提交／分配成本。该判断基于阶段总耗时，仍不能拆分纯内核与 IO。

仍限冻结 FP16/W8A16、B<=8、相同零起点前缀、总长<=64、一步 decode；不支持 padding／样本各自不同位置。
本轮没有新增 W4A16、训练、通用 generate、异步队列或 GPU/NPU 并行，也未测功耗／长上下文／完整 30 层端到端速度。
没有新增依赖／安装导入 OpenVINO runtime，仍依赖驱动图编译器。
native C++/DLL/PYD、SHAVE RMSNorm、基础真正 FP32 算法、权重／量化保持原版本；只修改 SmolLM 图布局和测试工具。
历史 `torch_npu_assembly_benchmark.py` 改为加载它所属阶段的 after/ 快照，避免旧 sample 方法与当前整批图混用；
单独 B2 冒烟验证旧／新拼接暖调用 8/6、输出／KV 逐位相同，旧报告未被覆盖。

## 来源与复现

权重 revision 12fd25f77366fa6b3b4b768ec3050bf629380bac，safetensors SHA256：{model_hash}。
before/after、diff、每个 JSON/log/NPZ、候选图 XML/bin 与 summary.json 留存。
`layout-probe` 使用保留的初始 candidate.py；正式版本进一步移除了 sample 接口／预上传，并修复单 token 批量 prefill。
`failures/single-token-prefill/` 保存新增边界首次编译失败；`failures/odd-batch-bit-parity/` 保存附加 B1 逐位断言失败。
后者改用已有执行门槛并保留逐位结果及全部误差，未放宽原数值门槛，不能据此声称独立 B1 的全部输出逐位等价。

```powershell
Set-Location E:\\Pytorch_xfire_NGPU
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','paired'
# 依次运行 probe / numeric / boundary / guards / positions / regression / hybrid / profile / profile_before / performance。
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_batch_isolation.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_batch_lowering.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_batch_benchmark.py
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_legacy_assembly_probe.py
# 设备测试必须串行；before/ 为配对旧版本。
& .\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_smollm_batch.py
```
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8');(OUT/'after').mkdir(exist_ok=True);summary['source_snapshots']={}
for name in [*changed,'TORCH_NPU_README.md','npunlock/docs/NPU4000.md','torch_npu_batch_benchmark.py','torch_npu_assembly_benchmark.py',
    'npunlock-probe/research/summarize_smollm_batch.py','npunlock-probe/research/smollm_batch_probe.py',
    'npunlock-probe/research/smollm_batch_isolation.py','npunlock-probe/research/smollm_batch_lowering.py',
    'npunlock-probe/research/smollm_legacy_assembly_probe.py','npunlock-probe/torch-device/build.json']:
    target=OUT/'after'/Path(name).name;shutil.copyfile(ROOT/name,target)
    summary['source_snapshots'][name]=dict(snapshot=str(target.relative_to(OUT)),sha256=digest(target))
for name in [*changed,'TORCH_NPU_README.md','npunlock/docs/NPU4000.md','torch_npu_assembly_benchmark.py']:
    target=Path(name).name
    (OUT/f'{target}.diff').write_text(''.join(difflib.unified_diff((OUT/'before'/target).read_text(encoding='utf-8').splitlines(True),
        (ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='before/'+target,tofile='after/'+target)),encoding='utf-8')
summary['artifact_sha256']={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='summary.json'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','total_npu_calls','warm_calls_per_layer','controls']},indent=2))
