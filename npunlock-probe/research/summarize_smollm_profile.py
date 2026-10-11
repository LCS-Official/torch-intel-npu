"""Audit decode stage profiles, compiler failures and DMA assembly comparison."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-profile'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['probe','numeric','boundary','guards','positions','hybrid','paired','regression','profile_before','profile','performance']
cases={name:json.loads((OUT/f'{name}.json').read_text(encoding='utf-8')) for name in names}
extras={name:json.loads((OUT/f'{name}.json').read_text(encoding='utf-8')) for name in ['concat','device','assembly']}
baseline=json.loads((OUT.parent/'smollm-kv-fused/summary.json').read_text(encoding='utf-8'))
sources=cases['regression']['sources'];changed=['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py','torch_intel_npu.py','npunlock/python/npunlock/npu4000.py']
for name,case in cases.items():
    assert case['sources']==sources and case['kernel_manifest']==baseline['kernel_manifest'] and case['checkpoint']==baseline['checkpoint'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
for name,value in sources.items():
    assert digest(ROOT/name)==value,name
    if name not in changed:assert value==baseline['sources'][name],name
for name in changed:assert digest(OUT/'before'/Path(name).name)==baseline['sources'][name],name
for name,value in extras['concat']['sources'].items():assert digest(ROOT/name)==value,name
for name,value in extras['assembly']['sources'].items():assert digest(ROOT/name)==value,name
assert extras['assembly']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
assert build==baseline['extension_build'] and digest(ROOT/'npunlock-probe/torch-device/backend.cpp')==build['source_sha256']
assert extras['device']['extension_sha256']==build['extension_sha256'] and extras['device']['session_dll_sha256']==build['session_sha256']
model_hash=digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors');assert model_hash==baseline['checkpoint_file_sha256']
initial=json.loads((OUT/'initial-profile/profile.json').read_text(encoding='utf-8'))
assert initial['status']=='passed' and initial['final_stats']['live_buffers']==0 and not initial['final_stats']['context_failed']
assert initial['sources']['torch_npu_smollm_kv_demo.py']==digest(OUT/'initial-profile/torch_npu_smollm_kv_demo.py')
for name,value in initial['sources'].items():
    if name!='torch_npu_smollm_kv_demo.py':assert value==baseline['sources'][name],name
failures={}
for name in ['batch-head-fold','batch-projection']:
    path=OUT/'failures'/name;failure=json.loads((path/'paired.json').read_text(encoding='utf-8'))
    assert failure['status']=='failed' and not failure['failure_stats']['context_failed']
    for source in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:assert digest(path/source)==failure['sources'][source],(name,source)
    failures[name]=dict(error=failure['error'],source_sha256=failure['sources']['torch_npu_smollm.py'])
entries=[entry for name in ['probe','numeric','boundary','positions'] for entry in cases[name]['checks'] if 'execution' in entry]
cache_entries=[item for entry in entries for item in entry.get('cache_local_reference',[])]
paired=cases['paired']['checks'];assert all(e['all_output_cache_bits_equal'] for e in paired)
assert all(e['generated_tokens_equal'] for e in cases['hybrid']['checks'])
warm_calls={}
for entry in paired:
    if entry['warmed']:
        batch=entry['batch'];count=entry['num_layers'];value=dict(previous=entry['previous_graph_calls']//count,current=entry['graph_calls']//count)
        assert warm_calls.setdefault(batch,value)==value
assert warm_calls=={1:dict(previous=4,current=4),2:dict(previous=8,current=6),4:dict(previous=10,current=8),8:dict(previous=14,current=12)}
assert cases['profile_before']['baseline_adapter_sha256']==baseline['sources']['torch_npu_smollm.py']
profiles={}
for name in ['profile_before','profile']:
    profiles[name]=cases[name]['timings']
    for entry in profiles[name]:
        assert entry['rounds']==21 and entry['warm_host_copies_and_compiles']==0
        expected=entry['num_layers']*warm_calls[entry['batch']]['previous' if name=='profile_before' else 'current']
        assert sum(entry['native_calls'].values())==expected
performance=[]
for entry in cases['performance']['timings']:
    value={k:v for k,v in entry.items() if k!='samples_ms'};med=entry['median_ms']
    value['latency_reduction_percent']={mode:100*(1-med[mode+'_decode']/med['previous_'+mode+'_decode']) for mode in ['fp16','w8a16']}
    for mode in ['fp16','w8a16']:
        assert entry['graph_calls'][mode+'_decode']==entry['num_layers']*warm_calls[entry['shape'][0]]['current']
        assert entry['graph_calls']['previous_'+mode+'_decode']==entry['num_layers']*warm_calls[entry['shape'][0]]['previous']
    performance.append(value)
assembly=[]
assert len(extras['assembly']['checks'])==len(extras['assembly']['timings'])==5
assert all(e['all_bits_equal'] for e in extras['assembly']['checks'])
for entry in extras['assembly']['timings']:
    assert entry['rounds']==41 and entry['same_model_input_cache'] and entry['warm_host_copies_and_compiles']==0
    assert all(len(values)==82 for values in entry['samples_ms'].values()) and len(entry['round_pairs_ms'])==41
    for version in ['previous','current']:
        assert entry['graph_calls'][version]==entry['num_layers']*warm_calls[entry['batch']][version]
    assembly.append({k:v for k,v in entry.items() if k not in ['samples_ms','round_pairs_ms']})
summary=dict(status='passed',date='2026-10-05',sources=sources,extension_build=build,checkpoint=cases['probe']['checkpoint'],
    checkpoint_file_sha256=model_hash,kernel_manifest=cases['probe']['kernel_manifest'],packages=cases['probe']['packages'],
    total_npu_calls=sum(case['final_stats']['npu_calls'] for case in {**cases,**extras}.values()),
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),paired_bits_equal_records=len(paired),
    continuous_decode_steps=sum(e.get('positions_checked',0) for e in cases['positions']['checks']),
    max_execution_relative_l2=max(e['execution']['relative_l2'] for e in entries),max_execution_scaled_error=max(e['execution']['max_scaled_error'] for e in entries),
    max_quality_relative_l2_official=max(e['quality_official']['relative_l2'] for e in entries if 'quality_official' in e),
    all_hybrid_tokens_equal=True,warm_calls_per_layer=warm_calls,profiles=profiles,performance=performance,assembly=assembly,
    assembly_sources=extras['assembly']['sources'],compiler_failures=failures,
    initial_profile_sources=initial['sources'],
    cases={name:dict(checks=len(case['checks']),timings=len(case.get('timings',[])),npu_calls=case['final_stats']['npu_calls'],final_live_buffers=0)
        for name,case in {**cases,**extras}.items()})
table='\n'.join(f"| {name} | {v['checks']} | {v['timings']} | {v['npu_calls']} |" for name,v in summary['cases'].items())
profile_table=[]
for name in ['profile_before','profile']:
    for e in profiles[name]:
        p=e['mean_stage_percent'];norm=p['rmsnorm1']+p['rmsnorm2'];total=e['mean_traced_ms']
        profile_table.append(f"| {'旧' if name=='profile_before' else '新'} | {e['mode']} | {e['num_layers']} / {e['batch']} | "
            f"{e['median_plain_ms']:.3f} | {norm:.1f}% | {p['attention']:.1f}% | {p['assembly']:.1f}% | {p['ffn']:.1f}% | "
            f"{e['mean_other_ms']/total*100:.1f}% | {e['mean_native_total_ms']/total*100:.1f}% |")
timings='\n'.join(f"| {e['shape'][0]} / {e['prefix_tokens']}→{e['shape'][1]} | {e['num_layers']} | "
    f"{e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | "
    f"{e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | "
    f"{e['latency_reduction_percent']['w8a16']:.1f}% | {e['median_ms']['w8a16_full_prefix']:.3f} | {e['median_ms']['cpu_fp16_decode']:.3f} |" for e in performance)
assembly_table='\n'.join(f"| {e['num_layers']} / {e['batch']} | {e['mode']} | {e['median_ms']['previous']:.3f} → "
    f"{e['median_ms']['current']:.3f} | {e['latency_reduction_percent']:.2f}% | {e['median_paired_delta_ms']:.3f} | "
    f"{e['positive_rounds']} / {e['rounds']} | {e['graph_calls']['previous']} → {e['graph_calls']['current']} |" for e in assembly)
report=f'''# NPU decode 分段诊断与 DMA 拼接合并

2026-10-05，Lunar Lake 8086:643e、驱动 {cases['probe']['final_stats']['driver_version']}、编译器 8.3。
本轮测量各阶段及原生调用耗时，试验批量注意力／投影布局，最终合并三个独立 DMA 拼接。
`prefill`／`decode`、模型算术、精度和不可变快照约定不变；无需新依赖或原生重建。

## 分段计时的口径

21 轮原始／带计时调用交替执行。计时器仅在 demo 中使用 stdlib mock 临时包装方法，结束后恢复。
RMSNorm、注意力、拼接、FFN 为阶段总时间，递归父子调用去重，拼接从注意力总时间扣除。
原生执行列是 `_C.run/run_many` 的嵌套耗时，包含参数绑定、提交、设备执行及 fence 等待，
不能分离纯内核与 fence，也不是额外阶段。阶段外还含 Python、Tensor 分配／释放和 C++ 检查；
“其他”是总时间减互不重叠的阶段时间。健康检查计时与上述时间重叠，不可再次相加。
Python `synchronize()` 仅查询状态；每次图提交本身同步等 fence。初始准备／编译／传输均排除，
暖区间零显式 Tensor 主机复制／新增编译。计时包装有开销，各组保存未包装／包装样本以便评估。

下面毫秒列为无包装 decode 中位数，百分比用带包装均值计算。旧／新 profile 分进程运行，
桌面负载／频率未控制，跨进程绝对时间仅作诊断。最终收益以同模型、同输入／缓存的 ABBA 对照为主要依据。

| 版本 | 精度 | 层数 / B | 无包装中位 ms | 两次 RMSNorm | 注意力 | 拼接 | FFN | 其他 | 原生执行占总时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(profile_table)}

第一轮、生产未改变时的 profile 另存 `initial-profile/`；当时单样本 RMSNorm 合计约 38%、
注意力约 30%；B4 注意力约 44%、B8 约 60%，拼接约 10–13%。它只包裹注意力叶调用，
父调度计入“其他”；最终 profile 统一包裹父子并去重，故两版阶段百分比不可直接混比。
状态查询只占很小比例，不能把原生执行包络都解释成 Python 检查或都解释成计算／IO。

## 试验与最终实现

折叠 batch/head 布局，以及整批 QKV／缓存更新后逐样本 context 的方案，都在 B2 首次编译失败：
`IE.Convolution` 无法通过 `ConvertBatchedLayerTo1N`，Level Zero `0x78000007`。
失败发生在编译，不是数值门槛失败；没有发生推理 device-lost，失败状态记录 context_failed=false。
源码、JSON/log 保存于 `failures/batch-head-fold/` 和 `failures/batch-projection/`；两条路径已撤回。
这不证明所有批量 QKV 布局都不可行，也不影响已有静态 FFN 批量支持。

最终复用原生多输出能力，将注意力输出、K、V 三个 Concat 放在同一张字节图中。
`compile_concat_many` 强制编译结果只有 DMA、无 DPU／SHAVE 计算任务；`_concat_many` 检查整块
连续输入、dtype、形状、存储和绑定上限，复用 `_run_many` 的输出／别名／寿命检查。
每个输出独立分配，F16/F32 只按字节搬运。B1 不改变；B>1 仍顺序执行 B 次注意力，最后一次提交输出三组拼接。
每层暖提交 B1/B2/B4/B8：4/8/10/14 → 4/6/8/12。两次 RMSNorm 和 FFN 仍各自执行。

## 验证与同进程性能

14 个最终进程串行通过，累计 {summary['total_npu_calls']:,} 次 NPU 调用；shutdown 后每个 live_buffers=0、context_failed=false。
最初 profile 和两次被撤回的编译试验不计入这个统计。

| case | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{table}

{summary['numerical_records']} 条数值、{summary['fixed_input_cache_tensor_checks']} 项固定实际输入 K/V、{summary['continuous_decode_steps']} 步连续解码通过。
24 组旧／新输出、K/V、位置逐位一致；8 组普通 forward 逐位一致；8 组混合生成 token 与 CPU 一致。
22 项模型缓存错误在任何层执行前拒绝，inference_mode 通过；快照不变、空槽为零、63→64／完整 64 prefill 均通过。
执行门槛仍 L2<0.3%、`abs(error)/(abs(reference)+RMS)<1%`，最大执行 L2 {summary['max_execution_relative_l2']*100:.4f}%，
最大归一化误差 {summary['max_execution_scaled_error']*100:.4f}%；官方 CPU 浮点参考质量 L2 最大 {summary['max_quality_relative_l2_official']*100:.4f}%，仍用 3% 门槛。
通用拼接检查包括两种精度的 Inf、NaN payload、次正规数、负零、1 ULP 等位模式，50 次复用各自输出
零新增分配／主机复制／编译，以及 B8 三输出与错误用法拒绝后恢复；旧单输出真正 FP32 add 与设备 quick suite 通过。

以下为同进程 21 轮交错、轮换顺序的中位 ms。旧适配器从 before/ 加载，双方用同一个原生扩展和当前 Python runtime。
排除初始准备／编译／预热／上传下载及 CPU mask/RoPE；包含 Python、分配、DMA、提交／fence。
CPU FP16 为官方参考，包含重复 decode 的 DynamicCache crop，不代表最优 CPU 实现。

| B / 前缀→总长 | 层数 | FP16 旧→新 | W8A16 旧→新 | W8 延迟减少 | W8 重算前缀 | CPU FP16 decode |
|---|---:|---:|---:|---:|---:|---:|
{timings}

这轮 W8A16 批量延迟下降 5.2–10.7%，但执行未改变的单层 B1 FP16 对照也出现约 21% 差异，
说明不同模型实例／测量顺序和桌面负载可能干扰计时，不能将所有差异都归因于修改。
因此追加下面的控制对照：复用同一模型、输入和缓存，仅临时切换旧／新拼接方法，图编译缓存共享。
41 轮交替 ABBA／BAAB，每轮旧／新各两次；切换方法发生在计时外，每种策略共 82 次计时。
预热后零显式主机复制／新增编译，五组输出／缓存／位置逐位一致；每层调用数与上述结论一致。

| 层数 / B | 精度 | 旧→新中位 ms | 延迟减少 | 每轮旧减新中位 ms | 新更快轮数 | 整段暖提交旧→新 |
|---|---|---:|---:|---:|---:|---:|
{assembly_table}

B1 两组总体中位差异均小于 1%，逐轮胜出只有 18/41，不应声称 B1 有加速。
B4 FP16／W8A16 分别下降约 7.4%／5.9%，33/41 与 36/41 轮新路径更快；B8 单层 W8A16 下降约 3.2%，
27/41 轮更快，收益较弱。整体支持三次拼接合并一次的批量收益，仍是本机一轮暖延迟观测，未控制频率／后台负载。
不同采样口径及进程的绝对时间不可相减作为加速，当前四层 B4 对照仍慢于前一轮 CPU FP16 decode 的观测值。
真正的整批注意力布局仍未跑通，主瓶颈未消除。

原始样本和调用计数在 performance.json／assembly.json，阶段样本在 profile_before.json／profile.json。
统计不证明端到端全 LLM 或长上下文速度优势；复制计数排除驱动 DMA，显式内存峰值排除编译常量／CMX／驱动／CPU 模型。
仍限冻结 FP16/W8A16、B<=8、总长<=64；未增加 W4A16、训练、padding、通用 generate、异步执行、GPU/NPU 并行。
当前结果支持继续处理批量注意力编译布局和小阶段同步成本；进一步硬件时间分解需设备时间戳／更底层计数器。

## 来源与复现

模型 revision：12fd25f77366fa6b3b4b768ec3050bf629380bac；safetensors SHA256：{model_hash}。
native DLL/PYD/C++、SHAVE RMSNorm、权重、量化哈希与前轮一致；基础 FP32 算法未改动，回归通过，只新增 DMA 编译／调度助手。
离线运行，不安装／导入 OpenVINO runtime，仍依赖驱动图编译器。before/after、diff、JSON/log/NPZ 与 summary.json 保留来源。

```powershell
Set-Location E:\\Pytorch_xfire_NGPU
.\\run_torch_npu.ps1 -Script .\\torch_npu_concat_many_demo.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_assembly_benchmark.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','profile'
# 依次运行 profile_before / probe / numeric / boundary / guards / positions / hybrid / paired / regression / performance。
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_profile_device_probe.py
# NPU 测试必须串行；before/ 用于旧版对照。
& .\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_smollm_profile.py
```
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8');(OUT/'after').mkdir(exist_ok=True);summary['source_snapshots']={}
for name in [*changed,'TORCH_NPU_README.md','npunlock/docs/NPU4000.md','torch_npu_concat_many_demo.py',
    'torch_npu_assembly_benchmark.py','npunlock-probe/research/summarize_smollm_profile.py',
    'npunlock-probe/research/smollm_profile_device_probe.py','npunlock-probe/torch-device/build.json']:
    target=OUT/'after'/Path(name).name;shutil.copyfile(ROOT/name,target)
    summary['source_snapshots'][name]=dict(snapshot=str(target.relative_to(OUT)),sha256=digest(target))
for name in [*changed,'TORCH_NPU_README.md','npunlock/docs/NPU4000.md']:
    target=Path(name).name
    (OUT/f'{target}.diff').write_text(''.join(difflib.unified_diff((OUT/'before'/target).read_text(encoding='utf-8').splitlines(True),
        (ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='before/'+target,tofile='after/'+target)),encoding='utf-8')
summary['artifact_sha256']={str(path.relative_to(OUT)):digest(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name!='summary.json'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({key:summary[key] for key in ['status','total_npu_calls','numerical_records','fixed_input_cache_tensor_checks','continuous_decode_steps','warm_calls_per_layer','assembly']},indent=2))
