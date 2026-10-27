"""Audit migration results and retain a compact Markdown report before cleanup."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'npunlock-probe/research/results/torch-migration'


def main():
    reports = [json.loads((OUT/f'after-{case}.json').read_text(encoding='utf-8'))
               for case in ('models', 'sgd', 'updates', 'boundaries')]
    before = [json.loads(p.read_text(encoding='utf-8')) for case in ('models', 'sgd', 'updates', 'boundaries')
              if (p := OUT/f'before-{case}.json').is_file()]
    assert len(before) in (0, 4), 'Incomplete baseline; restore all four files or remove stale baseline files'
    if before:
        baseline_compute = [c for r in before[:3] for c in r['checks']]
        assert len(baseline_compute) == 22
        assert sum(c['status'] == 'passed' for c in baseline_compute) == 2
        assert sum(c['status'] == 'failed' for c in baseline_compute) == 20
    checks = [c for r in reports for c in r['checks']]
    assert len(checks) == 36 and all(r['status'] == 'passed' for r in reports)
    supported = sum(c['status'] == 'passed' for c in checks)
    blocked = sum(c['status'] == 'blocked' for c in checks)
    assert supported + blocked == len(checks)
    hashes = reports[0]['source_sha256']
    for report in reports:
        assert report['source_sha256'] == hashes
        for name, digest in hashes.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        final = report['final_stats']
        assert final['closed'] and not final['context_failed']
        assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_buffer_bytes',
                                          'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
        assert all(c['live_buffer_delta'] == 0 for c in report['checks'])
        assert sum(c['npu_calls'] for c in report['checks']) == report['stats_before_shutdown']['npu_calls']
    for report in before:
        assert not report['final_stats']['context_failed']
        for name in ('npunlock-probe/torch-device/backend.cpp', 'npunlock-probe/torch-device/_torch_intel_npu.pyd'):
            assert report['source_sha256'][name] == hashes[name]
    lines = [
        '# 普通 PyTorch 模型迁移兼容性验收', '',
        '日期：2026-10-16。本地开发版，尚未同步至 GitHub v0.1.0。', '',
        '本轮测试普通 ATen `intel_npu` 设备后端，不使用冻结 DPU/SmolLM 适配器；',
        '这是 CUDA 风格 PyTorch 模型流程的迁移检查，不是 CUDA 运行时或二进制兼容认证。', '',
        '## 环境与口径', '',
        f'- Windows x64 / Core Ultra 7 258V / NPU {reports[0]["final_stats"]["hardware_device"]}。',
        f'- NPU API driver_version={reports[0]["final_stats"]["driver_version"]}，Torch {reports[0]["final_stats"]["torch_version"]}，Python {sys.version.split()[0]}。',
        '- 使用现有 `.conda/npunlock`；CPU 参考 1 线程、关闭 oneDNN 以运行 FP16 backward。',
        '- CPU 初始化、固定随机种子 43、MSE；上传后独立训练；不重置 CPU 权重追随 NPU 轨迹。',
        '- CPU/NPU 输出、loss、输入/参数梯度、参数与动量逐项对照；FP16 rtol=0.005/atol=0.0008，FP32 rtol=0.0001/atol=0.000008。',
        '- 本轮未安装包、未下载模型、未重编译或修改 C++/SHAVE/驱动，不测吞吐/功耗。', '',
        '## 发现与修复', '',
        '1. 普通 SGD momentum 第二步执行 `buf.mul_`，原先被分解为 `mul.out` 并触发输出别名/布局拒绝。',
        '   新增 PrivateUse1 `mul_.Scalar/Tensor`：设备内读到独立结果，再写回原 Tensor；支持完整密集转置并保留 version。',
        '2. Nesterov 与 weight_decay 使用 `grad.add(..., alpha=...)`，原先只接受 alpha=1。',
        '   现在加减法支持非 bool 实数 alpha 与对应一阶梯度；有限 alpha 必须在输出 dtype 范围内。',
        '   非 1 alpha 使用 FP32 中间计算，alpha 按输出 dtype 舍入，最后返回原 dtype；alpha=1 保留旧路径。',
        '3. `mul_` 显式限定 no_grad、同 shape/dtype/device 的 Tensor 或 Python 实数；拒绝广播、部分存储、',
        '   不同 stride 的重叠别名与 bool。self-alias 安全写回，saved-tensor version guard 仍报错。', '',
        '没有通过修改 torch.cuda、替换模型 forward 或 CPU 算术回退来取得通过结果。', '',
        '## 普通模型训练与恢复', '',
        '| 模型 | 精度 | 四步 NPU loss（前三步训练、第四步从 checkpoint 恢复） | 各步显式 host copies | 其中标量缓存首次上传 |',
        '|---|---|---|---|---|',
    ]
    for row in reports[0]['checks']:
        assert row['steps'] == 4 and row['portable_checkpoint_resume'] and row['independent_cpu_trajectory']
        assert row['step_host_copies'] == row['scalar_cache_uploads']
        assert row['steady_host_payload_copies'] == [0, 0]
        lines.append(f'| {row["name"].split("-torch.")[0]} | {row["precision"]} | ' +
                     ' → '.join(f'{v:.8g}' for v in row['losses']) +
                     f' | {row["step_host_copies"]} | {row["scalar_cache_uploads"]} |')
    lines += ['',
        'MLP：5→4→2；卷积网络：Conv2d(1→2)、ReLU、全局池化、Linear；TinyResNet：三层卷积、单个残差分支、全局池化和 Linear。',
        '输入分别为 (2,5) 与 (2,1,4,4)；未加入 BatchNorm 或分类 loss。',
        '普通 `.to("intel_npu")`、forward、MSE.backward、SGD.step、zero_grad(set_to_none=False)、eval/inference_mode 通过。', '',
        '模型使用 lr=0.025、momentum=0.8、dampening=0.1、foreach=False、fused=False。',
        'checkpoint 显式将 detached 参数及 SGD 动量转为 CPU，用内存 BytesIO 保存，`torch.load(..., map_location="cpu", weights_only=True)` 后',
        '加载到新 NPU 模型/优化器并继续第四步；恢复后的输出、梯度、参数和动量仍匹配独立 CPU 轨迹。',
        '复制统计排除输入准备、数值检查下载及 checkpoint 搬运；第一/二步可首次上传 Python 标量缓存。',
        '第三步与恢复后的训练步显式 host payload copies 均为 0；不表示驱动内部没有 DMA。', '',
        '## SGD 与算子契约', '',
        '- basic、momentum、dampening、Nesterov、weight_decay、Nesterov+weight_decay、maximize：两种 dtype 共14组，每组4次更新。',
        '- SGD 使用非连续完整密集转置参数；每一步检查参数、布局与动量缓冲区。Python 数值超参数，显式 foreach=False/fused=False。',
        '- mul_：两种 dtype 的标量/Tensor、完整密集转置、self-alias、返回对象和版本计数。',
        '- add/sub：两种 dtype 的广播、一阶梯度、out、Python 标量与 alpha=0/0.8/-2/60000/Inf/NaN。',
        '- 12个非法更新/alpha/别名动作在没有新增 NPU 图提交的情况下拒绝；saved-tensor version 变更检查通过。', '',
        '## 仍阻止通用模型迁移的项目', '',
        '| 检查入口 | 本机实际首次失败位置 |', '|---|---|',
    ]
    for row in reports[3]['checks']:
        if row['status'] == 'blocked':
            lines.append(f'| {row["name"]} | `{row["error"].replace("|", "/")}` |')
    lines += ['',
        f'这些{blocked}项是已验证会明确拒绝的边界，不计为功能支持。',
        ('默认 BatchNorm 与整数 target 存储入口已通过；INT64与默认BatchNorm完整验证见 ../int64-batchnorm/REPORT.md。'
         if supported >= 27 else '默认 BatchNorm 的 num_batches_tracked 和分类 target 需要整数存储。'),
        ('有界CrossEntropyLoss及resident INT64 target已通过，完整范围见 ../classification/REPORT.md。'
         if supported >= 28 else 'CrossEntropyLoss 先遇到 log_softmax 缺口，修复该算子后还需解决 target 与 NLL 路径。'),
        'LayerNorm 当前由 PyTorch 分解到 native_batch_norm；SDPA 当前先遇到非二维矩乘限制，并未验证其他注意力组成。',
        ('AdamW普通单Tensor路径已通过，本地完整范围见 ../adamw/REPORT.md。' if supported >= 29 else
         'AdamW 在原地权重衰减后才遇到 lerp 缺口；未支持的组合可能已更新部分参数，捕获异常后应从原始 checkpoint 重建。'),
        '原生 torch.save(NPU Tensor) 走字节存储序列化，目前不支持 Uint8 设备存储；显式 CPU checkpoint 是本轮验证的入口。', '',
        '## 验收计数与既有回归', '',
        f'迁移最终验收：{supported}组计算/训练通过，{blocked}组明确未支持边界，共 {sum(r["stats_before_shutdown"]["npu_calls"] for r in reports):,} 次真实 NPU 调用。',
        '四个独立进程均无 live buffer 增量、无 context failure；shutdown 后 live/cache/scalar/quarantine/session 全部归零。',
        ('基线33组中只有基本SGD两种 dtype 通过；其余20个计算/训练配置因 mul_/alpha 等限制失败，另11项为此次记录的未支持项目。'
         if before else '本次重跑未提供修复前基线，以上仅审计当前实现；修复前对照见原始2026-10-16报告。'),
        '诊断开发期间修正过统计字段、标量缓存预期和首次未支持算子名称；这些不算作后端修复。', '',
    ]
    regressions = [
        ('设备 Quick', ROOT/'npunlock-probe/research/results/torch-npu-device.json'),
        ('训练 Quick', ROOT/'npunlock-probe/research/results/torch-npu-training.json'),
        ('原有兼容性', ROOT/'npunlock-probe/research/results/compatibility/compatibility.json'),
    ]
    total = sum(r['stats_before_shutdown']['npu_calls'] for r in reports)
    for name, path in regressions:
        r = json.loads(path.read_text(encoding='utf-8'))
        assert r['status'] == 'passed', (name, r['status'])
        finals = [c['final_stats'] for c in r['cases']] if 'cases' in r else [r['final_stats']]
        for final in finals:
            assert not final['context_failed'] and final['closed']
            assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_buffer_bytes',
                                              'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
        if 'cases' in r:
            calls = sum(c['stats_before_shutdown']['npu_calls'] for c in r['cases'])
            count = sum(len(c['checks']) for c in r['cases'])
        else:
            calls = r['final_stats']['npu_calls']
            count = len(r['checks'])
        total += calls
        lines.append(f'- {name}：{count}组 passed，{calls:,}次 NPU 调用。')
    lines += [f'- 最终迁移与既有回归合计：{total:,}次 NPU 调用（不含开发中失败诊断/重跑）。', '',
        '## 使用与复现', '', '```powershell',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_migration_demo.py',
        '.\\run_torch_npu.ps1 -Device -Quick',
        '.\\run_torch_npu.ps1 -Training -Quick',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_compat_demo.py',
        '.\\.conda\\npunlock\\python.exe .\\npunlock-probe\\research\\torch_migration_report.py', '```', '',
        '报告生成需当前 after 诊断及上述回归 JSON；四份 before 诊断可选，提供时额外审计修复前基线。',
        '按用户要求验收后删除 JSON/日志，保留本报告与源码；清理后重新执行上述命令可审计当前实现。',
        '重建修复前基线需恢复原始后端版本；不能直接从本报告恢复逐项原始数据。', '',
        '## 后续优先级与范围', '',
        ('有界分类loss已通过；后续优先扩展普通模型的下一处实际算子缺口。' if supported >= 28 else
         '优先完成分类训练所需的稳定 log_softmax 与 NLL 前后向；整数 target 存储本身已支持。'
         if supported == 27 else '优先补默认 BatchNorm 的整数计数/模型状态迁移，再做分类训练所需的整数 target、稳定 log_softmax 与 NLL 前后向。'),
        '随后扩展 LayerNorm/Embedding/批量矩乘和注意力'+('。' if supported >= 29 else '、AdamW。')+'每项均需独立数值与拒绝契约测试。',
        '本轮不认证任意 torchvision/HF 模型、CUDA扩展/运行时、torch.cuda API、多流、AMP、GPU+NPU协同、其他设备/驱动或长时间训练。', '',
        '## 最终代码指纹（SHA-256）', '', '```text',
    ]
    for name, digest in hashes.items():
        lines += [name, digest]
    lines += ['```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Audited {len(checks)} checks; final NPU calls including regression: {total:,}')


if __name__ == '__main__':
    main()
