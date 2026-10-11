"""Audit and aggregate native multi-output KV checks and paired timings."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-multi'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['probe','numeric','boundary','guards','positions','hybrid','performance','regression']
cases={name:json.loads((OUT/f'{name}.json').read_text()) for name in names}
extras={name:json.loads((OUT/f'{name}.json').read_text()) for name in ['bindings','device']}
baseline=json.loads((OUT.parent/'smollm-kv/summary.json').read_text())
sources=cases['regression']['sources'];runner='torch_npu_smollm_kv_demo.py'
runner_versions={digest(ROOT/runner),digest(OUT/'failures/packed-cache-type'/runner)}
for name,case in cases.items():
    assert case['status']=='passed',name
    assert {k:v for k,v in case['sources'].items() if k!=runner}=={k:v for k,v in sources.items() if k!=runner},name
    assert case['sources'][runner] in runner_versions and case['kernel_manifest']==baseline['kernel_manifest'],name
for name,case in {**cases,**extras}.items():
    assert case['status']=='passed' and case['final_stats']['live_buffers']==0 and not case['final_stats']['context_failed'],name
for path,value in sources.items():assert digest(ROOT/path)==value,path
changed={'torch_npu_smollm.py','torch_intel_npu.py',runner,'npunlock-probe/torch-device/_torch_intel_npu.pyd'}
for path,value in baseline['sources'].items():
    if path not in changed:assert digest(ROOT/path)==value,path
for path,value in extras['bindings']['sources'].items():assert digest(ROOT/path)==value,path
build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
for field,path in [('source_sha256','npunlock-probe/torch-device/backend.cpp'),
                   ('extension_sha256','npunlock-probe/torch-device/_torch_intel_npu.pyd'),
                   ('session_sha256','npunlock-probe/session-runtime/npunlock_session.dll')]:assert build[field]==digest(ROOT/path)
assert extras['device']['extension_sha256']==build['extension_sha256'] and extras['device']['session_dll_sha256']==build['session_sha256']
model_hash=digest(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors');assert model_hash==baseline['checkpoint_file_sha256']
entries=[entry for name in ['probe','numeric','boundary','positions'] for entry in cases[name]['checks'] if 'execution' in entry]
cache_entries=[item for entry in entries for item in entry.get('cache_local_reference',[])]
hybrid=cases['hybrid']['checks'];performance=[]
for entry in cases['performance']['timings']:
    item={key:value for key,value in entry.items() if key!='samples_ms'};med=entry['median_ms']
    item['packed_over_multi']={mode:med['packed_'+mode+'_decode']/med[mode+'_decode'] for mode in ['fp16','w8a16']}
    item['latency_reduction_percent']={mode:100*(1-med[mode+'_decode']/med['packed_'+mode+'_decode']) for mode in ['fp16','w8a16']}
    item['full_prefix_over_decode']={mode:med[mode+'_full_prefix']/med[mode+'_decode'] for mode in ['fp16','w8a16','cpu_fp16']}
    performance.append(item)
summary=dict(status='passed',date='2026-10-05',hardware_device=cases['probe']['final_stats']['hardware_device'],
    driver_version=cases['probe']['final_stats']['driver_version'],
    cases={name:dict(checks=len(case['checks']),timings=len(case.get('timings',[])),npu_calls=case['final_stats']['npu_calls'],
                    final_live_buffers=0,peak_explicit_tensor_bytes=case['final_stats']['peak_buffer_bytes'])
           for name,case in {**cases,**extras}.items()},
    total_npu_calls=sum(case['final_stats']['npu_calls'] for case in {**cases,**extras}.values()),
    numerical_records=len(entries),fixed_input_cache_tensor_checks=len(cache_entries),
    continuous_decode_steps=sum(entry.get('positions_checked',0) for entry in cases['positions']['checks']),
    max_execution_relative_l2=max(entry['execution']['relative_l2'] for entry in entries),
    max_execution_scaled_error=max(entry['execution']['max_scaled_error'] for entry in entries),
    max_quality_relative_l2_official=max(entry['quality_official']['relative_l2'] for entry in entries if 'quality_official' in entry),
    max_local_cache_relative_l2=max(entry['relative_l2'] for entry in cache_entries),
    max_local_cache_scaled_error=max(entry['max_scaled_error'] for entry in cache_entries),
    all_hybrid_tokens_equal=all(entry['generated_tokens_equal'] for entry in hybrid),
    max_hybrid_logits_relative_l2=max(entry['logits_relative_l2'] for entry in hybrid),
    max_hybrid_next_token_kl=max(entry['next_token_kl'] for entry in hybrid),performance=performance,
    sources=sources,runner_versions_by_case={name:case['sources'][runner] for name,case in cases.items()},
    packages=cases['probe']['packages'],checkpoint=cases['probe']['checkpoint'],kernel_manifest=cases['probe']['kernel_manifest'],
    checkpoint_file_sha256=model_hash,extension_build=build,
    limits=['Frozen SmolLM2 FP16/W8A16; B<=8; total length<=64; zero-based causal positions; one token per decode.',
            'B>1 still slices samples and assembles on NPU; synchronous fences; no batch parallelism.',
            'Independent complete output allocations; state tensors read-only; no input/output or output/output alias.',
            'Warm timings exclude preparation/compile/transfers; CPU FP16 decode includes crop; desktop load/frequency uncontrolled.',
            'Paired old/new adapters use the same new extension; isolates graph packing/splitting, not binary speed.',
            'Copy counters exclude driver DMA/compiler; peaks exclude constants/CMX/driver/CPU model allocations.',
            'Hybrid retains CPU embedding/final norm/logits/remaining layers; no W4A16, general generate, training or longer contexts.'])
table='\n'.join(f"| {name} | {v['checks']} | {v['timings']} | {v['npu_calls']} | 0 |" for name,v in summary['cases'].items())
timings='\n'.join(f"| {e['shape'][0]} / {e['prefix_tokens']}→{e['shape'][1]} | {e['num_layers']} | "
    f"{e['median_ms']['packed_fp16_decode']:.3f} → {e['median_ms']['fp16_decode']:.3f} | "
    f"{e['median_ms']['packed_w8a16_decode']:.3f} → {e['median_ms']['w8a16_decode']:.3f} | "
    f"{e['latency_reduction_percent']['w8a16']:.1f}% | {e['median_ms']['w8a16_full_prefix']:.3f} | "
    f"{e['median_ms']['cpu_fp16_decode']:.3f} |" for e in performance)
reductions={mode:[e['latency_reduction_percent'][mode] for e in performance] for mode in ['fp16','w8a16']}
report=f'''# NPU KV cache：原生多输出

2026-10-05，Lunar Lake 8086:643e、驱动 {summary['driver_version']}、编译器 8.3。
底层 session DLL 已有多输出能力；本轮扩展 PyTorch C++ 接入层，复用其原生 infer 接口。
注意力输出、K/V 直接绑定三个独立设备分配，删除旧版 Concat 打包和后续 DMA 拆分。
无需新增依赖，session DLL、SHAVE RMSNorm 内核、权重量化与基础 FP32 算术保留原版本。

## 接入与提交

`stack.prefill(prefix)`／`stack.decode(one_token,cache)` API 不变。输入为 FP16 隐藏状态，
模式 FP16／W8A16，B<=8、总长<=64、从零开始的 causal 前缀。K/V 按三个 GQA head
留在设备上，固定 64 槽位。旧快照不被覆盖，调用方不得编辑 state Tensor。
多输出要求完整、连续、独立的分配，不得和输入或其他输出共享存储。

C++ `run_many` 将多个 Tensor 绑定到一次原生 infer；旧 `run` 保留签名并委托该执行器。
同步 fence、异常及失败隔离策略保留。Python `_run_many` 检查输入、输出个数／shape／dtype、
存储和别名。本轮没有启用异步、原地缓存更新或任意子缓冲区绑定。

| 每层暖调用 | 旧打包版 | 多输出版 |
|---|---:|---:|
| B1 decode | 8 | 5 |
| B2 decode | 23 | 17 |
| B4 decode | 39 | 27 |
| B8 decode | 73 | 49 |

B1 为两次 RMSNorm、一次注意力／K/V、一次 FFN、一次 FP32 位置增量。
B>1 仍通过设备样本切分、B1 注意力图和设备拼接绕过编译器对批量动态权重矩乘的限制；
减少的是各样本的输出拆分。B8 还包含超过 65,536 计算分块的拼接。
暖 decode 区间没有显式主机复制／按位置新增编译，计数器排除驱动 DMA。
普通 forward 仍每层 4 次提交。

## 验证

10 个独立进程通过，累计 {summary['total_npu_calls']:,} 次 NPU 提交；
各进程 shutdown 后 live_buffers=0，context_failed=false。

| case | 检查记录 | 计时组 | NPU 提交 | 最终 live buffers |
|---|---:|---:|---:|---:|
{table}

{summary['numerical_records']} 条数值记录、{summary['fixed_input_cache_tensor_checks']} 条固定实际输入 K/V 检查，
覆盖起点 0／10／26、1／2／4 层、不同样本 B2／B8、63→64 边界与完整 64-token prefill。
单层／4 层 × 两种精度各追加 63 步，总计 {summary['continuous_decode_steps']} 步；
每步复用图且无显式主机复制，10 个位置抽查数值。旧快照和已有 K/V 逐位不变，
未用槽位为零，同输入／快照重复解码逐位一致。22 项模型缓存错误用法在执行前拒绝，
inference_mode 正向执行通过。

执行门槛未变：L2<0.3%，每项 `abs(error)/(abs(reference)+RMS)<1%`。
层输出按 token 的 576 项算 RMS，K/V 按 head 的 64 项算 RMS。
最大输出执行 L2 {summary['max_execution_relative_l2']*100:.4f}%，最大归一化误差
{summary['max_execution_scaled_error']*100:.4f}%；相对官方 CPU 浮点层最大 L2
{summary['max_quality_relative_l2_official']*100:.4f}%，通过已有 3% 质量门槛。
固定实际输入 K/V 最大 L2 {summary['max_local_cache_relative_l2']*100:.4f}%，
最大归一化误差 {summary['max_local_cache_scaled_error']*100:.4f}%。
整段 CPU 输入的累积舍入差异另记诊断，旧固定 atol 失败计数保留；比较口径见前轮报告。

通用绑定检查覆盖三个混合形状／FP16/FP32 输出；FP32 位模式包括 70000、1 ULP、
次正规数、负零和 infinity。此项为 DMA/IO 检查，不作为 FP32 矩乘能力证明。
50 次复用输出无新增分配／主机复制／编译，改变输入后观察到新值。13 项 Python／C++
错误绑定拒绝后可继续有效调用，旧单输出真正 FP32 add 通过。
设备 quick suite 覆盖 FP16/FP32 算术、转换、归约、矩阵、Linear、视图、标量缓存、
线程调用与 shutdown 后存储寿命；8 组普通 Smol forward 与改动前输出逐位一致。

8 组显式混合生成 token 全部与 CPU 参考一致：{summary['all_hybrid_tokens_equal']}。
前 2／4 层使用 NPU KV，其余层／embedding／final norm／logits 在 CPU，保留段外传输。
首步 logits 最大 L2 {summary['max_hybrid_logits_relative_l2']*100:.4f}%，最大 KL
{summary['max_hybrid_next_token_kl']:.6f}。这是短生成 smoke test，未验证全 30 层或广泛质量。

## 同进程性能

CPU 8 线程、21 轮交错计时、轮换顺序。旧打包适配器原样从 before/ 加载，仅重定位
内核路径；旧／新适配器使用同一个本轮 C++ 扩展，隔离输出图／拆分路径的改动。
编译、初始准备／预热、输入上传／输出下载和 CPU mask／RoPE 准备排除。
NPU 时间包含 Python、分配、提交、设备 DMA 和同步 fence，不是纯内核时间。
CPU FP16 为官方浮点参考，重复同一 decode 时 crop 成本包含，不代表 CPU 最优实现。

单位 ms，中位数；旧→新为缓存 decode 打包版→多输出版。

| B / 前缀→总长 | 层数 | FP16 旧→新 | W8A16 旧→新 | W8 延迟减少 | W8 重算前缀 | CPU FP16 decode |
|---|---:|---:|---:|---:|---:|---:|
{timings}

W8A16 六组延迟减少 {min(reductions['w8a16']):.1f}–{max(reductions['w8a16']):.1f}%，
FP16 为 {min(reductions['fp16']):.1f}–{max(reductions['fp16']):.1f}%。提交数减少不等于同等比例的延迟改善。
B1 W8 解码基本接近重算短前缀，尚不能认为建立了稳定 KV 加速优势；B4 仍慢于重算。
所有缓存路径仍慢于相应 CPU FP16 decode。桌面负载和频率未控制，小差异不宜过度解读。
全部原始 21 样本、各路径提交和零暖主机复制／新增编译记录均保存于 performance.json。

## 来源与后续

计时首次失败是 runner 把另一个模块定义的旧 SmolKVCache 当作层组；改为判断缓存是否
为 tuple 后通过。失败 JSON/log／runner 保存于 `failures/packed-cache-type/`，
不计入最终 10 进程统计。各 case 的生产哈希一致，runner 两版本均有快照和按 case 哈希。

K/V 本体 B1／4 层 192 KiB，B8／4 层 1.5 MiB；保留快照会额外占用内存。
显式 Tensor 峰值排除编译常量、CMX、驱动内部和 CPU 模型。Tensor 1,048,576 元素、
计算分块 65,536 和缓存 512 图的上限未放宽。未增加 W4A16、训练、padding、通用 generate、
任意位置／更长上下文或 TF32 路径。下一步可先消除独立位置更新小图，减少 batch
切分／拼接，再推进 W4A16；长上下文收益需另测。

本轮只重建 `_torch_intel_npu.pyd`；session DLL、RMSNorm ELF 和检查点哈希与前轮一致。
模型 revision：12fd25f77366fa6b3b4b768ec3050bf629380bac；safetensors SHA256：
{model_hash}。离线加载，不执行远程模型代码，不安装／导入 OpenVINO runtime。

```powershell
Set-Location E:\\Pytorch_xfire_NGPU
.\\build_torch_npu_device.ps1
.\\run_torch_npu.ps1 -Script .\\torch_npu_multi_output_demo.py
.\\run_torch_npu.ps1 -Script .\\torch_npu_smollm_kv_demo.py -ScriptArguments '--case','probe'
# 依次运行 numeric / boundary / guards / positions / hybrid / performance / regression。
.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\smollm_multi_device_probe.py
# NPU 测试必须串行；before/ 快照用于旧版性能和普通 forward 对照。
& .\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\summarize_smollm_multi.py
```

before/after、源码 diff、build.json、各 JSON/log/NPZ 与 summary.json 保存来源及原始数据。
前轮报告和记录保留于 `smollm-kv/`。
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8');(OUT/'after').mkdir(exist_ok=True)
snapshots={}
for name in ['torch_npu_smollm.py','torch_intel_npu.py','torch_npu_smollm_kv_demo.py','torch_npu_multi_output_demo.py',
             'TORCH_NPU_README.md','npunlock/docs/NPU4000.md','npunlock-probe/torch-device/backend.cpp',
             'npunlock-probe/torch-device/build.json','npunlock-probe/torch-device/_torch_intel_npu.pyd',
             'npunlock-probe/research/summarize_smollm_multi.py','npunlock-probe/research/smollm_multi_device_probe.py']:
    target=OUT/'after'/Path(name).name;shutil.copyfile(ROOT/name,target)
    snapshots[name]=dict(snapshot=str(target.relative_to(OUT)),sha256=digest(target))
summary['source_snapshots']=snapshots
for name in ['torch_npu_smollm.py','torch_intel_npu.py','torch_npu_smollm_kv_demo.py','npunlock-probe/torch-device/backend.cpp']:
    target=Path(name).name
    (OUT/f'{target}.diff').write_text(''.join(difflib.unified_diff((OUT/'before'/target).read_text().splitlines(True),
        (ROOT/name).read_text().splitlines(True),fromfile='before/'+target,tofile='after/'+target)))
summary['artifact_sha256']={str(path.relative_to(OUT)):digest(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name!='summary.json'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({key:summary[key] for key in ['status','total_npu_calls','numerical_records','fixed_input_cache_tensor_checks',
    'max_execution_relative_l2','all_hybrid_tokens_equal','performance']},indent=2))
