"""Export the completed benchmark's tables and a log-scale comparison figure."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'npunlock-probe/research/results/performance'
data = json.loads((OUT / 'benchmark.json').read_text())
assert data['status'] == 'passed', 'Wait for all benchmark cases to finish'
cases = {(c['workload'], c['precision'], c['mode']): c for c in data['cases']}
rows = []
for (workload, precision, mode), case in cases.items():
    if mode != 'npu':
        continue
    cpu1, cpu8, npu = case['results']
    gpu_case = cases.get((workload, precision, 'xpu'))
    gpu_cpu, gpu = gpu_case['results'] if gpu_case else (None, None)
    fixture = cpu8['fixture_sha256']
    assert all(r['fixture_sha256'] == fixture and r['input_sha256'] == cpu8['input_sha256'] and
               r['model_sha256'] == cpu8['model_sha256'] for r in case['results'])
    if gpu:
        assert gpu['fixture_sha256'] == fixture and gpu['input_sha256'] == cpu8['input_sha256'] and gpu['model_sha256'] == cpu8['model_sha256']
    rows.append({'workload': workload, 'precision': precision, 'cpu1_ms': cpu1['warm']['median_ms'],
                 'cpu8_ms': cpu8['warm']['median_ms'], 'gpu_cpu8_ms': gpu_cpu['warm']['median_ms'] if gpu_cpu else None,
                 'gpu_ms': gpu['warm']['median_ms'] if gpu else None, 'npu_ms': npu['warm']['median_ms'],
                 'npu_over_cpu8': npu['warm']['median_ms']/cpu8['warm']['median_ms'],
                 'npu_over_gpu': npu['warm']['median_ms']/gpu['warm']['median_ms'] if gpu else None,
                 'npu_first_call_ms': npu['first_call_ms'], 'npu_graphs': npu['cached_graphs'],
                 'npu_calls': npu['warm_npu_calls_per_operation'], 'npu_effective_gflops': npu.get('effective_gflops'),
                 'cpu_half_backward_fallback': cpu8['cpu_half_backward_fallback']})

with (OUT / 'benchmark.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

table = ['# Lunar Lake PyTorch 性能实测', '', f"测试开始：{data['started_at']}（Asia/Shanghai）。", '',
         'Core Ultra 7 258V（8 核），Arc 140V，8086:643e NPU，31.48 GiB RAM。接通 AC、Balanced 电源方案；未修改系统电源设置。', '',
         'NPU 与主 CPU 基线：PyTorch 2.14.1+cpu；GPU 与配套 CPU 基线：2.12.0+xpu。',
         '各设备加载同一份保存的输入与权重，核对 SHA256；版本仍有差异，CPU 2.12 的控制结果保留在 CSV。', '',
         '以下为 eager 驻留运算的预热中位耗时，单位 ms，越低越好。GPU 每次操作后同步，NPU 每张图自身同步；',
         '排除初始上传、结果校验下载和最终输出对象释放。前向采样 11 次、训练采样 7 次，另预热 3 次；JSON 包含全部样本与 IQR。',
         '每个案例使用独立进程，硬件测试串行运行。冷调用包含新增图编译和执行，不等同于纯编译时间；后端初始化另外记录。', '',
         '| 工作负载 | 精度 | CPU 1线程 | CPU 8线程 | GPU | NPU | NPU/CPU8 | NPU/GPU |',
         '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    table.append(f"| {r['workload']} | {r['precision']} | {r['cpu1_ms']:.4f} | {r['cpu8_ms']:.4f} | "
                 f"{r['gpu_ms']:.4f} | {r['npu_ms']:.3f} | {r['npu_over_cpu8']:.0f}× | {r['npu_over_gpu']:.0f}× |")
table += ['', 'elementwise：1,048,576 元素的 (x+0.25)*0.5；mm：方阵乘法；Linear 前向：128x256→256；',
          'Linear 训练：65537x2→2；Conv2d：1x4x176x176、4→4、3x3、padding=1；',
          'SmallResNet 训练：2x3x8x8、4 个残差块、BatchNorm、池化和 Linear，复用现有 make_small_resnet。', '',
          '训练包含 zero_grad、输入梯度清空、前向、FP32 MSE、反向、SGD（lr=.001、foreach/fused=False），',
          '以不同设备相同的步数推进权重。首次调用检查前向、输入/参数梯度和更新，沿用现有 FP16/FP32 容差；',
          '性能采样的后续步骤不逐步下载验证。所有 NPU 热采样零测得的显式主机 Tensor 复制/新增图编译。', '',
          'FP16 Conv CPU 反向关闭 oneDNN，因为本机 AVX2_VNNI_2 平台缺少该实现；FP32 保留 oneDNN。',
          'FP16 NPU 矩阵/卷积使用 FP32 计算。已知深层 FP16 精度限制不在本次重新认证，SmallResNet 只测 FP32。',
          '这些数字反映当前研究后端，不是硬件峰值、持续吞吐或功耗/能效测试。', '',
          '## NPU 冷调用与提交数', '', '| 工作负载 | 精度 | 首次调用 ms | 缓存图数 | 热调用图提交数 |', '|---|---|---:|---:|---:|']
for r in rows:
    table.append(f"| {r['workload']} | {r['precision']} | {r['npu_first_call_ms']:.1f} | {r['npu_graphs']} | {r['npu_calls']:.0f} |")

table += ['', '## 调用剖析', '', '额外的 cProfile 调用与正式采样分开。原生 run 时间包含驱动提交、DMA、SHAVE 计算和 fence 等待，不能当成纯硬件计算时间。', '']
for (workload, precision, mode), case in cases.items():
    if mode != 'npu':
        continue
    profile = case['results'][-1].get('profile')
    if not profile:
        continue
    native = next(e for e in profile['top_self'] if '_torch_intel_npu.run' in e['function'])
    top = profile['top_cumulative'][0]['cumulative_ms']
    table.append(f"- {workload} / {precision}：原生 run {native['self_ms']:.1f} ms，占所剖析调用约 {native['self_ms']/top:.1%}。")
    for e in profile.get('graph_kinds', [])[:3]:
        table.append(f"  {e['kind']}：{e['calls']} 次、{e['total_ms']:.1f} ms（含包装/分配/提交/等待）。")

table += ['', '## 显式输入搬运', '', '| 工作负载 | 精度 | 字节 | NPU 上传/下载 ms | GPU 上传/下载 ms |', '|---|---|---:|---:|---:|']
for r in rows:
    key = r['workload'], r['precision']
    nt = cases[(*key, 'npu')]['results'][-1]['input_transfer']
    gt = cases[(*key, 'xpu')]['results'][-1]['input_transfer']
    table.append(f"| {r['workload']} | {r['precision']} | {nt['bytes']} | {nt['upload']['median_ms']:.3f}/{nt['download']['median_ms']:.3f} | "
                 f"{gt['upload']['median_ms']:.3f}/{gt['download']['median_ms']:.3f} |")
table += ['', '这里只测输入大小的独立分配+复制，不是完整模型所有输入/权重传输或零拷贝。', '',
          '测试要求在首次 backward 前导入 torch_intel_npu，避免 Torch autograd 队列先按尚未注册的设备数量初始化。',
          '初期随机数据不一致与 autograd 初始化失败记录保留在 pilot；正式结果修正测试流程，没有改变 NPU 算法或放宽容差。', '',
          '复测：', '', '```powershell', r'.\run_torch_npu.ps1 -Benchmark -XpuPython C:\Users\LC\miniconda3\envs\blood\python.exe',
          '```', '', 'JSON：benchmark.json；CSV：benchmark.csv；单例原始 JSON/log 与 .pstats 位于同目录。']
(OUT / 'BENCHMARK.md').write_text('\n'.join(table)+'\n', encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fp32 = [r for r in rows if r['precision'] == 'float32']
fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
positions = np.arange(len(fp32))
for offset, field, label, color in ((-.25, 'cpu8_ms', 'CPU 8 threads / Torch 2.14', '#216e39'),
                                    (0, 'gpu_ms', 'Arc 140V / Torch 2.12 XPU', '#0969da'),
                                    (.25, 'npu_ms', 'NPU / Torch 2.14', '#c65d18')):
    ax.barh(positions+offset, [r[field] for r in fp32], height=.23, label=label, color=color)
ax.set_xscale('log')
ax.set_yticks(positions, [r['workload'] for r in fp32])
ax.invert_yaxis()
ax.set_xlabel('Warm operation / training-step median (ms, log scale; lower is faster)')
ax.set_title('Lunar Lake: FP32 PyTorch eager latency')
ax.grid(axis='x', which='both', alpha=.15)
ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, fontsize=8.5)
fig.savefig(OUT / 'benchmark-fp32.png', dpi=180)
fig.savefig(OUT / 'benchmark-fp32.svg')
plt.close(fig)
print('Exported BENCHMARK.md, benchmark.csv and FP32 PNG/SVG')
