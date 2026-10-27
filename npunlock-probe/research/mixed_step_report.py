"""Audit first-divergence attribution, the CPU rounding control and held-out learning."""
import json
from pathlib import Path

from classification_report import audit, digest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'npunlock-probe/research/results'
OUT = RESULTS/'mixed-step-diagnosis'


def main():
    traces = [json.loads((RESULTS/'mixed-training-rounding'/name).read_text(encoding='utf-8')) for name in
              ('trajectory-6step.json', 'trajectory-unfused-control.json', 'trajectory-after.json')]
    canonical, control, quick = traces
    validation = json.loads((OUT/'validation.json').read_text(encoding='utf-8'))
    for trace in traces: audit(dict(trace, checks=[]))
    audit(validation)
    assert all(t['source_sha256'] == canonical['source_sha256'] for t in traces)
    assert canonical['status'] == control['status'] == 'diagnosed' and quick['status'] == 'passed'
    assert not canonical['unfused_cpu_reference'] and control['unfused_cpu_reference']
    assert canonical['first_independent_failure'] == {'mode': 'none', 'step': 5, 'parameter': '1.weight'}
    assert control['first_independent_failure'] is None and quick['first_independent_failure'] is None
    local_layers, optimizer_replays, differing_updates = 0, 0, 0
    for trace in (canonical, control):
        assert len(trace['runs']) == 2
        for run in trace['runs']:
            assert len(run['steps']) == 6
            for step in run['steps']:
                assert len(step['layers']) == 4 and len(step['optimizer']) == 6
                for layer in step['layers']:
                    assert all(v['unequal'] == 0 for v in layer['identical_input_and_seed'].values())
                    local_layers += 1
                for update in step['optimizer']:
                    assert update['identical_state_momentum']['unequal'] == 0
                    assert update['npu_vs_separate_fp32']['unequal'] == 0
                    if 'cpu_vs_vector_and_scalar_tail' in update:
                        assert update['cpu_vs_vector_and_scalar_tail']['unequal'] == 0
                    else:
                        assert update['cpu_vs_fused_fp32']['unequal'] == 0
                    assert update['identical_state_weight']['outside_tolerance'] == 0
                    optimizer_replays += 1
                    if trace is canonical and update['identical_state_weight']['unequal']: differing_updates += 1
    assert differing_updates == 16
    for old, changed, first in zip(canonical['runs'], control['runs'], (5, 4)):
        assert old['mode'] == changed['mode']
        assert old['initial_input_and_parameters_sha256'] == changed['initial_input_and_parameters_sha256']
        failures = [s for s in old['steps'] if any(v['outside_tolerance'] for v in s['parameters'].values())]
        assert failures[0]['step'] == first
        for step in changed['steps']:
            assert all(v['unequal'] == 0 for v in (*step['parameters'].values(), *step['updated_weights'].values()))
            assert all(layer['forward']['unequal'] == layer['upstream_gradient']['unequal'] == 0 for layer in step['layers'])
    assert len(quick['runs']) == 2 and all(len(r['steps']) == 4 for r in quick['runs'])
    val = validation['checks'][0]
    assert val['status'] == 'strict_failed' and not val['strict_trajectory_passed']
    assert val['heldout_improvement_passed'] and len(val['steps']) == 32
    assert val['training_samples'] == val['heldout_samples'] == 64 and val['batch_size'] == 8
    assert val['data_sha256']['train'] != val['data_sha256']['heldout']
    assert val['final']['prediction_disagreements'] == 0
    assert val['final']['cpu_accuracy'] == val['final']['npu_accuracy'] == .75
    assert val['first_strict_failure']['step'] == 3
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
    calls = sum(t['final_stats']['npu_calls'] for t in traces)+validation['final_stats']['npu_calls']
    rows = ['# 第5/6步混合训练差异：SGD舍入归因与独立验证集', '',
        '日期：2026-10-18。本地开发版；本轮扩展研究/验收脚本并更新文档，生产后端与内核保持不变，未推送GitHub。', '',
        '## 结论', '',
        '- 原none第6步、tanh第5步的严格梯度差异已完成归因：SGD参数更新的融合/分步乘加及half尾部中间舍入足以解释这两个失败点。',
        f'- 两种CPU参照共{local_layers}次逐层重放，相同输入/权重/上游梯度下前向与局部导数均逐位相同。',
        f'- {optimizer_replays}次相同状态SGD重放中，动量逐位一致；NPU参数更新均与分步FP32乘加一致，CPU均与本机向量/标量尾部规则一致。',
        '- 只将CPU参照改为NPU的分步更新顺序，两个网络各六步的前向、梯度、参数全部相同；NPU仍运行原torch.optim.SGD。',
        '- 标准CPU参照的严格失败保留，未放宽阈值；对照实验不能当作标准CPU轨迹通过。',
        '- 新增64训练/64独立验证样本的32步合成分类实验：两侧验证准确率均为75%，预测完全相同；该实验也有严格轨迹失败，不能替代严格验收。', '',
        '## 原失败重放', '',
        '沿用seed127、batch4、Linear(16,16)→LayerNorm(16)→GELU→Linear(16,3)，FP16 Linear/激活、FP32 LayerNorm参数，',
        '两侧显式FP32 CrossEntropy；SGD lr=.01/momentum=.8/foreach=False/fused=False，CPU单线程并关闭oneDNN。',
        '分别连续运行六步；CPU/NPU从相同初始化独立更新，不将CPU参数重置为NPU状态。',
        '同输入局部重放只用于诊断：把NPU层输入、参数、上游梯度复制给一个CPU shadow，核对该层实际输出与导数。', '',
        '| GELU | 首次严格失败（1起算） | LayerNorm weight梯度最大差 | 样本CPU / NPU |', '|---|---|---:|---|']
    for run, first in zip(canonical['runs'], (5, 4)):
        difference = run['steps'][first]['parameters']['1.weight']
        example = difference['examples'][0]
        rows.append(f"| {run['mode']} | 第{first+1}步 | {difference['max_abs']:.12g} | index {example['index']}：{example['cpu']:.15g} / {example['npu']:.15g} |")
    rows += ['', '原FP32参数梯度rtol=3e-5/atol=3e-6，FP16 Tensor rtol=.003/atol=.0005不变。',
        '同输入/上游梯度下LayerNorm/GELU/Linear的局部前后向均相同；该样例不支持把失败归因于LayerNorm局部导数或GELU实现。', '',
        '## 最早分叉与算术顺序', '',
        '两种GELU都从第2步的FP32 LayerNorm bias参数更新出现微小差异；相同状态的动量仍逐位相同。',
        '标准CPU向量add_使用FMA，对alpha*momentum+parameter只进行一次最终FP32舍入；当前NPU先执行FP32 mul，再执行FP32 add。',
        '这是FP32运算顺序不同，没有将FP32参数降为FP16。后续FP16参数的小张量/标量尾部还会出现CPU额外的half乘积舍入。',
        'CPU half向量主体先转FP32再FMA；本机AVX2每次处理32个half，尾部走标量规则。',
        '诊断用UCRT fmaf核对融合结果，用独立的CPU乘法/加法核对分步结果；没有把普通乘加冒充FMA。', '',
        '第2步的同状态bias样本：', '', '| GELU | 参数/index | CPU融合结果 | NPU分步结果 | 绝对差 |', '|---|---|---:|---:|---:|']
    for run in canonical['runs']:
        update = next(u for u in run['steps'][1]['optimizer'] if u['parameter'] == '1.bias')
        e = update['identical_state_weight']['examples'][0]
        rows.append(f"| {run['mode']} | 1.bias/{e['index']} | {e['cpu']:.16g} | {e['npu']:.16g} | {update['identical_state_weight']['max_abs']:.12g} |")
    rows += ['', f'原标准参照共72次同状态参数更新，{differing_updates}次有数值差异，但每次局部更新均在既有单算子容差内；',
        '对应24次FP32更新全部匹配CPU FMA，48次FP16更新全部匹配向量FMA主体加half标量尾部；NPU全部匹配分步FP32更新。',
        '小差异经过后续half舍入和反向传播，可以越过原FP32参数梯度阈值；不能从单步参数接近推断任意长度轨迹相同。', '',
        '## 改变CPU更新规则的对照', '',
        '保留相同网络、数据、NPU实现及NPU标准SGD，只将独立CPU参照的参数更新改为：', '',
        '```text', 'alpha = round_to_parameter_dtype(-lr)',
        'parameter = round_to_parameter_dtype(FP32(parameter) + FP32(momentum) * FP32(alpha))', '```', '',
        'CPU动量仍按既有乘法与加法更新，未读取NPU值来覆盖CPU轨迹；同状态shadow副本只参与核对。',
        '在两个网络共12步中，CPU/NPU四层前向、上游梯度、全部参数梯度及更新后参数均逐位相同。',
        '这项受控变化足以消除原第5/6步分叉，提供更新舍入是其来源的证据；它是诊断对照，不是新NPU模式或标准CPU兼容性修复。',
        '该结论限定于这两个六步重放，不把全部模型/其他输入的差异都归为舍入。', '',
        '## 独立验证集', '',
        '另用seed131的默认GELU混合网络，固定seed601生成一个线性teacher、64训练样本及64未参与训练的样本。',
        'FP16输入16维/3分类，batch8，四个epoch共32次更新，SGD lr=.03/momentum=.8；CPU/NPU独立训练，使用原标准优化器。',
        'teacher与数据在CPU准备，属于实验输入生成；模型前后向与更新在NPU执行，CPU参数更新对照未用于这个实验。', '',
        '| 指标 | CPU | NPU |', '|---|---:|---:|',
        f"| 初始验证loss | {val['initial']['cpu_loss']:.10g} | {val['initial']['npu_loss']:.10g} |",
        f"| 最终验证loss | {val['final']['cpu_loss']:.10g} | {val['final']['npu_loss']:.10g} |",
        f"| 初始验证准确率 | {val['initial']['cpu_accuracy']:.4%} | {val['initial']['npu_accuracy']:.4%} |",
        f"| 最终验证准确率 | {val['final']['cpu_accuracy']:.2%} | {val['final']['npu_accuracy']:.2%} |", '',
        f"最终64个预测全部相同；验证loss绝对差{abs(val['final']['cpu_loss']-val['final']['npu_loss']):.12g}，logits最大差{val['final']['logits_max_abs']:.12g}。",
        f"该实验原严格轨迹首次在第{val['first_strict_failure']['step']+1}步{val['first_strict_failure']['where']}超差，状态仍为strict_failed。",
        '没有用新的宽阈值把它改记为严格通过；该新输入上的逐项偏差未做完整局部归因。',
        '验证集提升说明这个合成任务未观察到预测质量下降，不认证真实数据集、任意模型、长期训练或AMP。', '',
        '输入/teacher及初始化指纹：', '', '```text']
    rows += [f'{name} {sha}' for name, sha in val['data_sha256'].items()]
    rows += [f"initial_parameters {val['initial_parameters_sha256']}", '```', '',
        '## 验收与资源', '',
        '| 入口 | 结果 | NPU调用 |', '|---|---|---:|',
        f"| 标准CPU六步归因 | 重现原两组严格失败；48层局部/72状态重放完成 | {canonical['final_stats']['npu_calls']} |",
        f"| 分步CPU对照六步 | 两个网络参数/梯度相同；不计作标准CPU通过 | {control['final_stats']['npu_calls']} |",
        f"| 原默认四步入口回归 | 原两组通过 | {quick['final_stats']['npu_calls']} |",
        f"| 64/64独立验证集 | 学习/验证提升；严格轨迹失败保留 | {validation['final_stats']['npu_calls']} |", '',
        f'最终验收共**{calls:,}次NPU调用**。所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。',
        '生产Python后端/autograd、20份resident ELF及汇编/manifest、C++/DLL/PYD均未改；未新增依赖、模型、图表或持久图缓存。',
        '本轮未重复设备/卷积/低位模型全回归，也未进行性能计时；没有为匹配CPU逐位结果而引入额外软件FMA生产路径。', '',
        '## 复现', '', '```powershell',
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\mixed_training_rounding_probe.py -ScriptArguments @('--steps','6')",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\mixed_training_rounding_probe.py -ScriptArguments @('--steps','6','--unfused-reference')",
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\mixed_training_rounding_probe.py',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\mixed_validation_probe.py',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/mixed_step_report.py', '```', '',
        '六步诊断返回零表示重现了预期差异并完成核对，原标准轨迹仍有严格失败；独立验证集命令按strict_failed返回非零。',
        '审计需四份新生成JSON；验收后按空间约定清理，保留本报告与脚本，旧报告作为历史记录。', '',
        '## 参考', '',
        '[PyTorch add ufunc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ufunc/add.h)区分标量表达式与向量fmadd；',
        '本机vec256_float.h/vec256_half.h、Loops.h、Half.h及sgd.py用于核对实际CPU控制流，指纹列于下方。',
        '[PyTorch数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)指出求值顺序和平台变化可使数学等价计算产生不同浮点结果。',
        '本轮不是仅凭该说明排除问题，而是以上述相同状态重放与改变CPU舍入顺序的实验完成归因；未移植CPU SIMD源码。', '',
        '## SHA-256', '', '```text']
    files = set(name for result in (*traces, validation) for name in result['source_sha256'])
    files.update(('npunlock-probe/research/mixed_step_report.py', 'npunlock-probe/torch-device/backend.cpp',
        'npunlock-probe/torch-device/_torch_intel_npu.pyd', 'npunlock-probe/session-runtime/npunlock_session.dll',
        '.conda/npunlock/Lib/site-packages/torch/optim/sgd.py',
        '.conda/npunlock/Lib/site-packages/torch/include/ATen/cpu/vec/vec256/vec256_float.h',
        '.conda/npunlock/Lib/site-packages/torch/include/ATen/cpu/vec/vec256/vec256_half.h',
        '.conda/npunlock/Lib/site-packages/torch/include/ATen/native/cpu/Loops.h',
        '.conda/npunlock/Lib/site-packages/torch/include/torch/headeronly/util/Half.h'))
    rows += [f'{name} {digest(ROOT/name)}' for name in sorted(files)]
    rows += ['```', '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'REPORT.md').write_text('\n'.join(rows), encoding='utf-8')
    print(json.dumps({'layer_replays': local_layers, 'optimizer_replays': optimizer_replays,
        'canonical_different_updates': differing_updates, 'npu_calls': calls, 'validation_status': val['status']}, indent=2))


if __name__ == '__main__':
    main()
