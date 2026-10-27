"""Audit FP64 classification checks and strict-versus-learning training outcomes."""
import json
from math import log
from pathlib import Path

from classification_report import audit, digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/classification-precision'


def main():
    quality, stability = (json.loads((OUT/name).read_text(encoding='utf-8')) for name in ('quality.json', 'stability.json'))
    for result in (quality, stability): audit(result)
    assert quality['status'] == 'passed' and len(quality['checks']) == 10
    assert all(c['status'] == 'passed' for c in quality['checks'])
    assert stability['status'] == 'strict_failed' and len(stability['checks']) == 2
    for c in quality['checks'][:-1]:
        for path in ('intel_npu-half-loss', 'intel_npu-fp32-loss'):
            assert all(c['candidates'][path][k]['nonfinite'] == 0 for k in ('log_softmax', 'loss', 'input_gradient'))
    for c in quality['checks'][:2]:
        assert c['candidates']['intel_npu-fp32-loss']['input_gradient']['different_from_rounded_reference'] == 0
    wide = quality['checks'][-2]
    assert wide['candidates']['cpu-half-loss']['loss']['nonfinite'] == 1
    assert quality['checks'][-1]['cpu_half_nonfinite'] == 65536
    expected_digests = ('44663e71cfb8a16803eae45b8dd8ec2795fadfa5c50c66f391293764c85e9773',
                        'd4886fce5cb84bd1105abe3f3fbbca626a1afdfea81aa230c8f3c9a5e51e2770')
    for c, mode, first, expected_digest in zip(stability['checks'], ('none', 'tanh'), (5, 4), expected_digests):
        assert c['status'] == 'strict_failed' and not c['strict_independent_trajectory_passed']
        assert c['mode'] == mode and c['initial_input_and_parameters_sha256'] == expected_digest
        assert len(c['steps']) == 32 and c['training_accuracy'] == 1. and c['finite_learning_checks_passed']
        assert c['first_strict_failure']['step'] == first and c['first_strict_failure']['where'] == 'gradient:1.weight'
        assert all(not s['strict_failures'] for s in c['steps'][:4])
        assert c['final_npu_loss'] < c['steps'][0]['npu_loss']
    assert digest(ROOT/'torch_intel_npu.py') == '0523715f8c418584baf28e75f239b8c3575a814a4f28cd32b03ba2f29067f55f'
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = json.loads((fixture/'resident/manifest.json').read_text())
    for name, sha in manifest['source_sha256'].items(): assert digest(fixture/name) == sha
    for name, entry in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == entry['elf_sha256']
        assert digest(fixture/f'resident/{name}.s') == entry['assembly_sha256']
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    for name, key in (('backend.cpp', 'source_sha256'), ('_torch_intel_npu.pyd', 'extension_sha256')):
        assert digest(ROOT/'npunlock-probe/torch-device'/name) == build[key]
    assert digest(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll') == build['session_sha256']
    calls = quality['final_stats']['npu_calls']+stability['final_stats']['npu_calls']
    rows = ['# 分类损失FP64对照与32步混合训练边界', '',
        '日期：2026-10-18。本地开发版；本轮新增验证与报告，生产算子/内核保持上一轮版本，未推送GitHub。', '',
        '## 结论', '',
        '- 10组分类精度/容量检查通过，包括原none/tanh两组网络logits、随机形状、偏移/范围及宽类别边界。',
        '- 原两例显式FP32损失得到的FP16输入梯度，均与FP64参考最终舍入到FP16的结果相同。',
        '- 本机CPU原生FP16保存统计有额外舍入；NPU保持FP32中间统计。原生half两侧结果不同，不能据此直接认定NPU算错。',
        '- 32步none/tanh网络都保持有限数值并拟合固定四样本；严格独立CPU轨迹分别在第6/第5步失败。',
        '  **两组32步检查标记strict_failed，未计为严格通过。** 四步通过的上一轮结论不推广为长期逐项一致。', '',
        '## 对照口径', '',
        '硬件/软件沿用Windows x64、Core Ultra 7 258V、NPU 8086:643e、driver_version=1005540、Torch 2.14.1+cpu。',
        'CPU单线程、关闭oneDNN。所有路径使用完全相同的已量化FP16 logits，再分别运行CPU/NPU的原生half或显式FP32损失。',
        '独立参考是这些logits转成FP64后的稳定log_softmax/CrossEntropy及其导数，没有拿CPU half作数学真值。',
        'CPU FP64只用于离线核对，不进入NPU运行时或训练数据计算。',
        'FP32前向保留rtol=3e-5/atol=3e-6；FP16前向与FP16输入梯度沿用rtol=.003/atol=.0005。',
        '显式FP32损失依然向FP16 logits返回FP16梯度，因此对FP64真值有最终half舍入误差，不等于整条反向全用FP32。', '',
        '## 原两例的FP64误差', '',
        '| GELU来源 | 路径 | log_softmax最大绝对差 | 输入梯度RMS差 | 与FP64舍入后half梯度不同元素 |',
        '|---|---|---:|---:|---:|']
    for c in quality['checks'][:2]:
        for path, value in c['candidates'].items():
            rows.append(f"| {c['name'].removeprefix('original-logits-')} | {path} | {value['log_softmax']['max_abs']:.9g} | "
                        f"{value['input_gradient']['rms']:.9g} | {value['input_gradient']['different_from_rounded_reference']} / 12 |")
    rows += ['', '原生half梯度的CPU/NPU优劣随样例改变：none的NPU误差较小，tanh的CPU RMS误差略小，不能宣称NPU始终更准确。',
        '显式FP32损失在这两个样例中消除了除最终half舍入外的输入梯度差异。',
        '这支持保留现有FP32统计策略；本轮没有为模仿CPU half溢出而改写生产算子，也未将原FP16损失整网失败改记为通过。', '',
        '## 大类别边界', '',
        '| 场景 | CPU half | NPU half | NPU FP32 | FP64参考 |', '|---|---:|---:|---:|---:|',
        f"| 65,532类全零logits的CrossEntropy | +inf | {wide['candidates']['intel_npu-half-loss']['loss_value']:.10g} | "
        f"{wide['candidates']['intel_npu-fp32-loss']['loss_value']:.10g} | {log(wide['shape'][-1]):.10g} |",
        f"| 65,536类全零logits的log_softmax每项 | -inf | {quality['checks'][-1]['npu_half_value']:.10g} | "
        f"{quality['checks'][-1]['npu_fp32_value']:.10g} | {quality['checks'][-1]['fp64_value']:.10g} |", '',
        '65,532类CE使用现有NLL packet的额外4列，恰好达到65,536元素任务上限；65,536类只检查log_softmax。',
        '本机CPU last-dimension实现把sum(exp)和log(sum)保存为scalar_t，FP16可能在求和时溢出。',
        'NPU内部保留FP32统计，最终half值只是把有限结果舍入到FP16；此例不需要降低NPU精度来兼容CPU异常。', '',
        '## 32步训练', '',
        '网络仍为Linear(16,16)→混合LayerNorm(16)→GELU→Linear(16,3)，SGD lr=.01/momentum=.8；',
        'FP16 Linear/激活、FP32 LayerNorm参数、两侧都显式使用FP32 CrossEntropy。seed127与原先初始化/数据顺序相同，',
        'CPU/NPU各自连续更新，不重置CPU权重，不放宽原参数梯度/权重阈值。严格差异被记录后继续训练，用于观察有限性和学习行为。', '',
        '| GELU | 初始NPU loss | 32次更新后CPU / NPU loss | 固定训练集准确率 | 首次严格失败（1起算） |', '|---|---:|---:|---:|---|']
    for c in stability['checks']:
        first = c['first_strict_failure']
        rows.append(f"| {c['mode']} | {c['steps'][0]['npu_loss']:.9g} | {c['final_cpu_loss']:.9g} / {c['final_npu_loss']:.9g} | "
                    f"{c['training_accuracy']:.0%} | 第{first['step']+1}步，{first['where']}，{first['mismatched']}个元素，最大差{first['max_abs']:.12g} |")
    rows += ['', '32步说明该固定四样本例子能正常学习，不能证明泛化、真实数据集训练或任意长时间稳定性。',
        '后续的严格轨迹偏差尚未逐层定位；不能仅凭最终loss接近就宣称已修复，也不能把该偏差直接等同于发散。',
        '下一项有针对性的工作是重放第5/6步，在相同输入与上游梯度下核对局部导数及优化器状态，避免盲目改精度。', '',
        '## 验收和资源', '',
        f"最终保留验收共**{calls:,}次NPU调用**：精度探针{quality['final_stats']['npu_calls']}次，32步训练{stability['final_stats']['npu_calls']:,}次。",
        '10组精度检查通过；2组32步严格轨迹未通过，其有限学习检查通过但不替代严格验收。',
        '所有检查live buffer增量为零；两个进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。',
        '未改变torch_intel_npu.py、autograd、20份resident ELF/汇编、manifest和C++/DLL/PYD；无需安装依赖或重编内核。',
        '没有新增模型、图表或持久图缓存；未进行性能计时，也未重复与本轮无关的设备/卷积/低位回归。', '',
        '## 复现', '', '```powershell',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\classification_precision_probe.py',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\mixed_training_stability_probe.py',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/classification_precision_report.py', '```', '',
        '第二个命令按strict_failed返回非零，同时保留完整32步结果；不要忽略该返回码。',
        '报告脚本要求重跑后的两个原始JSON并核对源码/内核/资源指纹；验收后按空间约定删除JSON，历史报告保留。', '',
        '## 参考', '',
        '[PyTorch LogSoftmaxKernelImpl.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)',
        '用于核对CPU half保存中间统计的类型；本机实际安装头文件指纹列于下方。main链接只作导航，没有移植CPU SIMD代码。',
        '原统计/更新修复见[混合LayerNorm报告](../layernorm-mixed/REPORT.md)与[更新规则报告](../mixed-training-rounding/REPORT.md)。', '',
        '## SHA-256', '', '```text']
    files = set(name for result in (quality, stability) for name in result['source_sha256'])
    files.update(('npunlock-probe/research/classification_precision_report.py',
        'npunlock-probe/torch-device/backend.cpp', 'npunlock-probe/torch-device/_torch_intel_npu.pyd',
        'npunlock-probe/session-runtime/npunlock_session.dll',
        '.conda/npunlock/Lib/site-packages/torch/include/ATen/native/cpu/LogSoftmaxKernelImpl.h'))
    rows += [f'{name} {digest(ROOT/name)}' for name in sorted(files)]
    rows += ['```', '']
    (OUT/'REPORT.md').write_text('\n'.join(rows), encoding='utf-8')
    print(json.dumps({'quality_passed': 10, 'strict_32step_failed': 2, 'npu_calls': calls, 'resident_kernels_verified': len(manifest['kernels'])}, indent=2))


if __name__ == '__main__':
    main()
