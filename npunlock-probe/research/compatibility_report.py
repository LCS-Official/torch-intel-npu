"""Summarize completed bug regressions and the first reduction optimization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'npunlock-probe/research/results/compatibility'
PERF = OUT.parent / 'performance-reduce-optimized'
BEFORE = OUT.parent / 'performance'


def read(path):
    value = json.loads(path.read_text())
    assert value['status'] == 'passed', str(path)
    return value


def main():
    compatibility = read(OUT/'compatibility.json')
    device, training = (read(OUT/f'regression-{name}.json') for name in ('device', 'training'))
    reduction = read(OUT/'reduce-optimization.json')
    resnet_ab = read(OUT/'resnet-interleaved.json')
    stream = [read(OUT/f'regression-{name}.json') for name in ('k-f32', 'conv-dense')]
    after = [read(path) for path in sorted(PERF.glob('npu-*.json'))]
    assert len(after) == 6
    old = {(c['workload'], c['precision']): c for c in read(BEFORE/'benchmark.json')['cases'] if c['mode'] == 'npu'}
    rows = []
    for case in after:
        baseline = old[(case['workload'], case['precision'])]
        a, b = baseline['results'][-1], case['results'][-1]
        assert a['device'] == b['device'] == 'intel_npu'
        for field in ('fixture_sha256', 'input_sha256', 'model_sha256'):
            assert a[field] == b[field], field
        assert b['warm_host_payload_copies'] == b['warm_graph_compilations'] == 0
        stats = case['final_stats']
        assert stats['live_buffers'] == 0 and not stats['context_failed']
        rows.append({'workload': case['workload'], 'precision': case['precision'],
                     'old_ms': a['warm']['median_ms'], 'new_ms': b['warm']['median_ms'],
                     'speedup': a['warm']['median_ms']/b['warm']['median_ms'],
                     'calls_before': a['warm_npu_calls_per_operation'], 'calls_after': b['warm_npu_calls_per_operation']})
    # All optimized cases must represent the same backend and compiled artifacts.
    for case in after:
        for name in ('torch_intel_npu.py', 'torch_npu_autograd.py', 'torch_npu_geometry.py',
                     'npunlock-probe/torch-device/backend.cpp', 'npunlock-probe/torch-device/build.json',
                     'npunlock/tests/fixtures/npu4000/resident/manifest.json'):
            assert case['source_sha256'][name] == hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), name
    data = {'status': 'passed', 'desktop_load': 'User reported watching video during this round; desktop was not controlled idle',
            'compatibility': compatibility, 'device': device, 'training': training,
            'stream': stream, 'reduction': reduction, 'resnet_interleaved': resnet_ab, 'performance_comparison': rows,
            'optimized_performance_cases': after,
            'limitations': ['Late registration ordinary backward still requires early import or scoped single-thread scheduling',
                            'Deep pure FP16 training limits remain; no full Torch/operator/model parity',
                            'End-to-end before/after timings are separate runs; same-process kernel A/B is recorded separately']}
    (OUT/'summary.json').write_text(json.dumps(data, indent=2))
    lines = ['# NPU 兼容性修复与首轮性能优化（2026-10-04）', '',
             '已先修复可复现的正确性/布局问题，再优化归约索引。保持真正 FP32、补偿求和、既有精度容差与禁止 CPU 算术回退。', '',
             '| 问题 | 修复/边界 |', '|---|---|',
             '| 极大/极小 FP32 除法梯度错误 | 分母导数改用 -(g/y)*(x/y)，不计算 y*y；含 0/Inf/NaN |',
             '| mean(dtype=FP16) 的 FP32 输入梯度不符 | 先按输出 dtype 舍入 mean 的反向除法，再转回输入 dtype |',
             '| 完整转置叶 Tensor 不能训练 | 支持密集 empty_strided、算术/归约、clone/copy_/zero_、梯度布局/累加与 SGD；DMA 逆排列写回 |',
             '| neg/abs/sign/reciprocal 缺少反向 | 新增一阶梯度；reciprocal 保留 FP16 中间乘法的舍入/溢出 |',
             '| 性能 resume 可能复用旧内核结果 | 新增原生扩展/build/内核 manifest/低层编译源哈希检查，另存本轮结果 |', '',
             '五个独立兼容性进程通过；修改前失败复现位于 ../../compat-before/reproduced/。',
             f"原有设备回归 {len(device['checks'])} 组/{device['final_stats']['npu_calls']:,} 次提交、训练回归 {len(training['checks'])} 组/{training['final_stats']['npu_calls']:,} 次提交通过（Quick）。",
             '另回归 K 超长矩阵/Linear 和密集流式卷积；各进程上下文健康、shutdown 分配归零。', '',
             '## 晚导入 PyTorch 兼容办法', '',
             '先 CPU backward，再注册 PrivateUse1，普通 NPU backward 仍可能触发内部队列断言。',
             'PyTorch 只初始化一次设备队列：[引擎源码](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp)。',
             '优先在首次任何 backward 前导入后端；已验证如下局部办法，退出后恢复调度设置，数值算术仍在 NPU 执行。', '',
             '```python', 'with torch.autograd.set_multithreading_enabled(False):', '    loss.backward()', '```', '',
             '## 归约优化的独立 A/B', '',
             '内层循环从每项动态除法/取模换为单轴步进或多轴进位索引；遍历顺序、Neumaier 补偿和非规格化数处理不变。',
             f"{len(reduction['bitwise_checks'])} 个 shape/axes/kind 对照，每项随机有限值与 IEEE 特殊值两种模式，新旧结果逐位相同。",
             '同一进程驻留输入/输出，3 次预热、9 次新旧交错计时，包含同步执行，排除编译/上传/校验下载。', '',
             '| 形状 / 轴 / 类型 | 原内核 ms | 新内核 ms | 加速 |', '|---|---:|---:|---:|']
    for row in reduction['timings']:
        lines.append(f"| {row['shape']} / {row['axes']} / {row['kind']} | {row['old_ms']:.3f} | {row['new_ms']:.3f} | {row['speedup']:.2f}× |")
    lines += ['', '## 端到端代表工作负载', '',
              '使用上轮相同保存输入/权重并校验 SHA256；Torch 2.14.1+cpu。前向采样 11 次、训练采样 7 次。',
              '本轮 AC=1、电量 100%、Balanced；用户确认本轮在看视频，桌面未受控为空闲。其他负载/共享内存/功耗状态可能影响时间；',
              '跨轮结果用作趋势参考，严格量化应在桌面空闲时另做对照。同进程交错 A/B 可减小干扰，仍不能消除全部波动。',
              '训练包含 zero_grad、前向、FP32 MSE、反向、基本 SGD；首次调用检查前向/梯度/更新，热采样不逐步下载验证。',
              '以下是两轮独立运行的中位数，兼容性修复也已进入本轮；不能把整个差值解释为严格的同进程内核 A/B。', '',
              '| 工作负载 | 精度 | 上轮 ms | 本轮 ms | 加速 | 每次图提交（前→后） |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['workload']} | {row['precision']} | {row['old_ms']:.3f} | {row['new_ms']:.3f} | {row['speedup']:.2f}× | {row['calls_before']:.0f}→{row['calls_after']:.0f} |")
    lines += ['', '本轮所有热采样零测得的主机载荷复制/新增图编译；提交数未减少，改的是内核索引开销。',
              '这仍明显慢于此前 CPU/GPU 基线，尚未实现专用 GEMM、DPU 训练或异步提交。', '',
              '## 残差网络回退诊断', '',
              '第一次整轮测量 SmallResNet 中位数 868.637 ms、范围 721.985–1175.639 ms，明显慢于上轮 559.201 ms；',
              '原始结果保留在 ../performance-reduce-optimized/first-pass/，没有丢弃。剖析中归约略快，其他 DMA/原生调用及 Python 开销变大，',
              '这组跨轮结果不能证明内核退化。随后移除连续 Tensor 的重复验证/stride 排序，完整边界检查仍在；上表为最终代码重测。',
              '另用最终后端在同一进程交错运行两个相同网络，只切换新旧归约 ELF，保留两套图，不销毁/reimport。',
              f"3 对预热、9 对测量：旧 {resnet_ab['median_ms']['old']:.3f} ms、新 {resnet_ab['median_ms']['new']:.3f} ms，{resnet_ab['speedup']:.2f}×。",
              '所有 12 对步骤的输入梯度、参数梯度及 SGD 更新按原始 FP32 位模式逐位检查一致；热循环零显式复制/新增编译。',
              '这是同进程内核对照，不能量化其他阶段/系统波动的各自贡献。[原始 A/B 数据](resnet-interleaved.json)。', '',
              '## 仍保留的限制', '',
              '深层纯 FP16 训练的既有数值限制仍在；二阶导数、跨 CPU/NPU 可微传输、BF16/整数/RNG、Adam/动量 SGD、AMP、',
              '任意部分/重叠/偏移视图、GPU+NPU 混合 Tensor、完整 torchvision 模型尚不保证支持。',
              '没有声称已消除所有潜在 bug 或达到完整 PyTorch 后端兼容性。', '',
              '原始数据：[汇总 JSON](summary.json)、[兼容性](compatibility.json)、[内核逐位/A-B](reduce-optimization.json)，',
              '端到端原始样本、首次调用与 profiler 位于 ../performance-reduce-optimized/。', '',
              '复现：`run_torch_npu.ps1 -Script torch_npu_compat_demo.py`；`run_torch_npu.ps1 -Script torch_npu_reduce_check.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('passed:', len(compatibility['cases']), 'compatibility processes,', len(rows), 'performance cases')


if __name__ == '__main__':
    main()
