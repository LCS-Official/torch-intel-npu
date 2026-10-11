"""Audit fused decode sources, isolated correctness runs and paired timings."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-fused'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['probe','numeric','boundary','guards','positions','hybrid','paired','regression','performance']
cases={name:json.loads((OUT/f'{name}.json').read_text()) for name in names}
baseline=json.loads((OUT.parent/'smollm-kv-multi/summary.json').read_text())
sources=cases['regression']['sources']
for name,case in cases.items():
    assert case['status']=='passed' and case['sources']==sources,name
    assert case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
    assert case['kernel_manifest']==baseline['kernel_manifest'] and case['checkpoint']==baseline['checkpoint'],name
for name,value in sources.items():
    assert digest(ROOT/name)==value,name
    if name not in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:
        assert value==baseline['sources'][name],name
for name in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py']:
    assert digest(OUT/'before'/name)==baseline['sources'][name],name
build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
assert build==baseline['extension_build']==json.loads((OUT/'before/build.json').read_text(encoding='utf-8-sig'))
assert digest(ROOT/'npunlock-probe/torch-device/backend.cpp')==build['source_sha256']
model_hash=digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')
assert model_hash==baseline['checkpoint_file_sha256']
entries=[entry for name in ['probe','numeric','boundary','positions'] for entry in cases[name]['checks'] if 'execution' in entry]
cache_entries=[item for entry in entries for item in entry.get('cache_local_reference',[])]
paired=cases['paired']['checks'];assert all(e['all_output_cache_bits_equal'] for e in paired)
hybrid=cases['hybrid']['checks'];assert all(e['generated_tokens_equal'] for e in hybrid)
warm_calls={}
for entry in paired:
    if entry['warmed']:
        batch=entry['batch'];count=entry['num_layers']
        value=dict(previous=entry['previous_graph_calls']//count,fused=entry['graph_calls']//count)
        assert warm_calls.setdefault(batch,value)==value
assert warm_calls=={1:dict(previous=5,fused=4),2:dict(previous=17,fused=8),4:dict(previous=27,fused=10),8:dict(previous=47,fused=14)}
performance=[]
for entry in cases['performance']['timings']:
    value={k:v for k,v in entry.items() if k!='samples_ms'};med=entry['median_ms']
    value['latency_reduction_percent']={mode:100*(1-med[mode+'_decode']/med['previous_'+mode+'_decode']) for mode in ['fp16','w8a16']}
    value['full_prefix_over_decode']={mode:med[mode+'_full_prefix']/med[mode+'_decode'] for mode in ['fp16','w8a16','cpu_fp16']}
    batch=entry['shape'][0];count=entry['num_layers']
    for mode in ['fp16','w8a16']:
        assert entry['graph_calls'][mode+'_decode']==count*warm_calls[batch]['fused']
        assert entry['graph_calls']['previous_'+mode+'_decode']==count*warm_calls[batch]['previous']
    performance.append(value)
summary=dict(status='passed',date='2026-10-05',sources=sources,extension_build=build,
    hardware_device=cases['probe']['final_stats']['hardware_device'],driver_version=cases['probe']['final_stats']['driver_version'],
    checkpoint=cases['probe']['checkpoint'],checkpoint_file_sha256=model_hash,kernel_manifest=cases['probe']['kernel_manifest'],
    packages=cases['probe']['packages'],warm_calls_per_layer=warm_calls,
    cases={name:dict(checks=len(case['checks']),timings=len(case.get('timings',[])),npu_calls=case['final_stats']['npu_calls'],
        final_live_buffers=0,peak_explicit_tensor_bytes=case['final_stats']['peak_buffer_bytes']) for name,case in cases.items()},
    total_npu_calls=sum(case['final_stats']['npu_calls'] for case in cases.values()),
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(entry.get('positions_checked',0) for entry in cases['positions']['checks']),
    paired_bits_equal_records=len(paired),all_hybrid_tokens_equal=True,
    max_execution_relative_l2=max(entry['execution']['relative_l2'] for entry in entries),
    max_execution_scaled_error=max(entry['execution']['max_scaled_error'] for entry in entries),
    max_quality_relative_l2_official=max(entry['quality_official']['relative_l2'] for entry in entries if 'quality_official' in entry),
    max_local_cache_relative_l2=max(entry['relative_l2'] for entry in cache_entries),
    max_local_cache_scaled_error=max(entry['max_scaled_error'] for entry in cache_entries),
    max_hybrid_logits_relative_l2=max(entry['logits_relative_l2'] for entry in hybrid),performance=performance,
    limits=['Frozen FP16/W8A16; B<=8; total length<=64; zero-based causal prefill; one token per decode.',
        'B>1 attention still runs sequentially; assembly, two RMSNorm and FFN are separate synchronized graphs.',
        'Positions 1..64 increment exactly in FP16 and store as FP32; FP32 sample indices 0..7 convert to int64.',
        'No generic Transformers generate/.to integration, W4A16, training, padding or long-context validation.',
        'Warm timings exclude preparation/compilation/segment transfers; include allocation, DMA/submission/fences.',
        'CPU FP16 reference includes DynamicCache crop; desktop load/frequency uncontrolled.',
        'Explicit copy counters exclude driver DMA; memory peaks exclude constants/CMX/driver/CPU models.'])
table='\n'.join(f"| {name} | {v['checks']} | {v['timings']} | {v['npu_calls']} |" for name,v in summary['cases'].items())
call_table='\n'.join(f"| B{batch} | {v['previous']} | {v['fused']} |" for batch,v in sorted(warm_calls.items()))
timings='\n'.join(f"| {e['shape'][0]} / {e['prefix_tokens']}→{e['shape'][1]} | {e['num_layers']} | "
    f"{e['median_ms']['previous_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | "
    f"{e['median_ms']['previous_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | "
    f"{e['latency_reduction_percent']['w8a16']:.1f}% | {e['median_ms']['w8a16_full_prefix']:.3f} | "
    f"{e['median_ms']['cpu_fp16_decode']:.3f} |" for e in performance)
reductions={mode:[e['latency_reduction_percent'][mode] for e in performance] for mode in ['fp16','w8a16']}
prefix_reductions=[100*(1-e['median_ms']['w8a16_decode']/e['median_ms']['w8a16_full_prefix']) for e in performance]
assert all(e['median_ms']['w8a16_decode']<e['median_ms']['w8a16_full_prefix'] for e in performance)
assert all(e['median_ms'][mode+'_decode']>e['median_ms']['cpu_fp16_decode'] for e in performance for mode in ['fp16','w8a16'])
report=f'''# NPU KV decode：位置更新与样本选取合并

2026-10-05，Lunar Lake 8086:643e、驱动 {summary['driver_version']}、编译器 8.3。
本轮仅改变冻结 SmolLM2 的图和 Python 调度，复用上一轮原生多输出扩展。
原生 session DLL、C++/PYD、SHAVE RMSNorm 内核、量化格式、模型权重和普通 forward 算术保持原版本。

## 实现与调用

`prefill`／`decode` API 和不可变快照约定不变。注意力图现在输出第四项 FP32 位置，
消除独立位置加法提交。图内使用已有 FP16 位置加一再转回 FP32；1..64 均可精确表示，
这是有界整数元数据更新，不更改模型精度或基础真正 FP32 算术保证。
B>1 时整批输入、归一化结果和 K/V 直接绑定，设备标量 sample 经 Convert/Gather 选取一个样本，
消除每个样本外部四次 DMA 切片提交。同一层／精度／批量形状的样本／位置复用一张图。
sample 常量在 prefill 准备并复用；暖 decode 不上传／下载 Tensor 或按样本／位置新建图。

编译器 8.3 的批量动态权重矩乘限制仍存在；注意力依次执行 B 次，随后在设备上拼接三个输出。
两次 RMSNorm 和 FFN 仍各自提交。B1 为四次提交，B>1 为 B+6。

| 每层暖 decode | 上一轮多输出版 | 本轮 |
|---|---:|---:|
{call_table}

前轮文档的 B8=49 来自首次 decode，包含两次 gamma 广播准备；本轮旧／新重复调用确认
真正暖 B8 为 47→14。首次 decode 的准备和编译不包含在暖性能测量中。

## 硬件验证

9 个独立进程串行通过，累计 {summary['total_npu_calls']:,} 次 NPU 调用；每个进程
shutdown 后 live_buffers=0、context_failed=false。底层扩展及基础算术源码哈希与前轮一致，
本轮没有重跑未改动的通用设备全套；改动覆盖下列模型测试。

| case | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
{table}

{summary['numerical_records']} 条数值记录、{summary['fixed_input_cache_tensor_checks']} 项固定实际输入 K/V 检查。
覆盖起点 0/10/26、1/2/4 层、不同样本 B2/B8、63→64 与完整 64-token prefill。
单层／四层 × FP16/W8A16 共连续 {summary['continuous_decode_steps']} 步，固定图复用且无暖主机复制／新增编译。
旧快照与已有 K/V 逐位不变，空槽保持零，重复解码一致。22 项错误模型缓存用法在任何层执行前拒绝，
inference_mode 通过。8 组普通 forward 与旧版逐位一致；24 组连续旧／新 decode 的输出、K/V 和位置也逐位一致，
涵盖 B1/B2/B4/B8、单层／四层和 1→4、32→35、61→64 位置。

执行门槛保持 L2<0.3%、`abs(error)/(abs(reference)+RMS)<1%`；输出 RMS 按 token，K/V 按 head。
最大执行 L2 {summary['max_execution_relative_l2']*100:.4f}%，最大归一化误差 {summary['max_execution_scaled_error']*100:.4f}%；
相对官方 CPU 浮点层最大 L2 {summary['max_quality_relative_l2_official']*100:.4f}%，保持已有 3% 质量门槛。
固定输入 K/V 最大 L2 {summary['max_local_cache_relative_l2']*100:.4f}%，归一化误差 {summary['max_local_cache_scaled_error']*100:.4f}%。
原固定 atol 失败计数保留。8 组前 2／4 层 NPU、其余 CPU 的混合生成 token 与 CPU 全部一致，
首步 logits 最大 L2 {summary['max_hybrid_logits_relative_l2']*100:.4f}%；属于短生成 smoke，不代表全 30 层／广泛质量。

## 同进程性能

CPU 8 线程、21 轮交错测量、轮换顺序。旧适配器从 before/ 原样加载，仅重定位内核路径；
双方复用同一个原生扩展和权重。排除初始准备、编译、预热、输入上传／输出下载和 CPU mask/RoPE 准备。
时间包含 Python、分配、设备 DMA、提交和 fence；CPU FP16 decode 包含 DynamicCache crop，非最优 CPU 基准。
桌面负载和频率未控制，跨轮历史绝对时间不作配对比较。

单位 ms，中位数。旧→新为上一轮多输出适配器→本轮位置／样本选取合并。

| B / 前缀→总长 | 层数 | FP16 旧→新 | W8A16 旧→新 | W8 延迟减少 | W8 重算前缀 | CPU FP16 decode |
|---|---:|---:|---:|---:|---:|---:|
{timings}

W8A16 相比上一轮延迟减少 {min(reductions['w8a16']):.1f}–{max(reductions['w8a16']):.1f}%，
FP16 为 {min(reductions['fp16']):.1f}–{max(reductions['fp16']):.1f}%。本轮六组 W8A16 缓存 decode
均快于重算短前缀，延迟低 {min(prefix_reductions):.1f}–{max(prefix_reductions):.1f}%；
所有缓存路径仍慢于对应 CPU FP16 decode。这是本轮暖层段结果，不代表端到端 LLM 或更长上下文优势。

重算前缀／CPU 对照与全部原始 21 样本保存于 performance.json；速度结论以同进程配对结果为准。
Tensor 存储上限 1,048,576、计算分块 65,536、缓存 512 图未放宽。显式复制计数不包含驱动 DMA；
显式峰值内存不包含编译常量、CMX、驱动内部和 CPU 模型。保留多个快照会额外占用内存。
仍限冻结 FP16/W8A16、B<=8、总长<=64；尚未增加 W4A16、训练、padding、通用 generate 或更长上下文。
混合生成保留 CPU embedding/final norm/logits、剩余层及段间传输，未实现 GPU/NPU 并行。

## 复现与来源

模型 revision：12fd25f77366fa6b3b4b768ec3050bf629380bac；safetensors SHA256：{model_hash}。
离线加载，不执行远程模型代码，不导入／安装 OpenVINO runtime；仍需驱动图编译器。
无需重建原生扩展或 SHAVE 内核。本轮 before/after、源码 diff、JSON/log/NPZ、summary.json 保留来源及原始数据。

```powershell
Set-Location E:\\Pytorch_xfire_NGPU
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','paired'
# 依次运行 probe / numeric / boundary / guards / positions / hybrid / regression / performance。
# 所有 NPU 测试必须串行；before/ 用于旧版对照。
& .\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_smollm_fused.py
```
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8');(OUT/'after').mkdir(exist_ok=True)
summary['source_snapshots']={}
for name in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py','TORCH_NPU_README.md','npunlock/docs/NPU4000.md',
             'npunlock-probe/research/summarize_smollm_fused.py','npunlock-probe/torch-device/build.json']:
    target=OUT/'after'/Path(name).name;shutil.copyfile(ROOT/name,target)
    summary['source_snapshots'][name]=dict(snapshot=str(target.relative_to(OUT)),sha256=digest(target))
for name in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py','TORCH_NPU_README.md','npunlock/docs/NPU4000.md']:
    target=Path(name).name
    (OUT/f'{target}.diff').write_text(''.join(difflib.unified_diff((OUT/'before'/target).read_text(encoding='utf-8').splitlines(True),
        (ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='before/'+target,tofile='after/'+target)),encoding='utf-8')
summary['artifact_sha256']={str(path.relative_to(OUT)):digest(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name!='summary.json'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({key:summary[key] for key in ['status','total_npu_calls','numerical_records','fixed_input_cache_tensor_checks',
    'continuous_decode_steps','paired_bits_equal_records','warm_calls_per_layer','performance']},indent=2))
