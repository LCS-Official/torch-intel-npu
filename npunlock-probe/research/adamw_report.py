"""Audit AdamW acceptance and retain failed model trajectories explicitly."""
import json
from pathlib import Path
import re
import sys
from classification_report import ROOT, RESULTS, audit, digest, read

OUT = RESULTS/'adamw'


def main():
    import torch
    new = [read(OUT/f'{case}.json') for case in ('primitives','optimizer','contracts')]
    assert [len(r['checks']) for r in new] == [2,13,1]
    assert all(c['status'] == 'passed' for r in new for c in r['checks'])
    assert all(r['source_sha256'] == new[0]['source_sha256'] for r in new)
    for c in new[1]['checks'][:12]:
        assert c['checkpoint_resume'] and c['zero_and_none_gradients'] and c['state_and_dense_transpose']
        assert c['step_host_copies'] == c['scalar_uploads'] and len(c['step_host_copies']) == 4
    assert new[1]['checks'][-1]['cpu_and_npu_zero_gradient_nonfinite']
    assert len(new[2]['checks'][0]['rejected_without_graph_submit']) == 19
    reference_sha = digest(Path(sys.modules[torch.optim.Adam.__module__].__file__))
    assert all(r['reference_adam_sha256'] == reference_sha for r in new)
    models = [json.loads((OUT/f'{name}.json').read_text()) for name in ('training','training-no-bn-bias')]
    assert [r['bn_bias'] for r in models] == [True,False]
    assert models[0]['source_sha256'] == models[1]['source_sha256']
    for result in models:
        assert len(result['checks']) == 6
        for c in result['checks']:
            if c['status'] == 'failed':
                assert 'AssertionError' in c['traceback'] and 'identical-gradient AdamW update checked=True' in c['traceback']
            else:
                assert c['status'] == 'passed' and len(c['losses']) == 4 and c['checkpoint_resume']
                assert c['identical_gradient_optimizer_check'] and c['independent_cpu_trajectory']
                assert c['step_host_copies'] == [n+1 for n in c['scalar_uploads']]
    assert all(c['status'] == 'passed' for c in models[1]['checks'] if c['name'].endswith('float32'))
    classification = [read(RESULTS/f'classification/{case}.json') for case in ('logsoftmax','nll','contracts','training')]
    assert all(r['source_sha256'] == classification[0]['source_sha256'] for r in classification)
    assert classification[0]['source_sha256'] == models[0]['source_sha256']
    migration = [read(RESULTS/f'torch-migration/after-{case}.json') for case in ('models','sgd','updates','boundaries')]
    migration_checks = [c for r in migration for c in r['checks']]
    supported = sum(c['status'] == 'passed' for c in migration_checks)
    blocked = sum(c['status'] == 'blocked' for c in migration_checks)
    assert supported >= 29 and supported+blocked == 36
    baseline_path = RESULTS/'torch-migration/adamw-before-boundaries.json'
    baseline = read(baseline_path) if baseline_path.is_file() else None
    if baseline:
        before_adamw = next(c for c in baseline['checks'] if c['name'] == 'adamw')
        assert before_adamw['status'] == 'blocked' and 'aten.lerp.Scalar_out' in before_adamw['error']
    old = [read(RESULTS/f'torch-npu-{case}.json') for case in ('device','training')]
    old += read(RESULTS/'compatibility/compatibility.json')['cases']
    all_results = new+models+classification+migration+old
    for r in all_results:
        audit(r)
    if baseline:
        final_before = baseline['final_stats']
        assert final_before['closed'] and not final_before['context_failed']
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    assert digest(ROOT/'npunlock-probe/torch-device/backend.cpp') == build['source_sha256']
    assert digest(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd') == build['extension_sha256']
    assert digest(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll') == build['session_sha256']
    if baseline:
        assert baseline['source_sha256']['npunlock-probe/torch-device/backend.cpp'] == build['source_sha256']
        assert baseline['source_sha256']['npunlock-probe/torch-device/_torch_intel_npu.pyd'] == build['extension_sha256']
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = json.loads((fixture/'resident/manifest.json').read_text())
    for name,sha in manifest['source_sha256'].items():
        assert digest(fixture/name) == sha
    for name,record in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == record['elf_sha256']
        assert digest(fixture/f'resident/{name}.s') == record['assembly_sha256']
    total_calls = sum(r['final_stats']['npu_calls'] for r in all_results)
    failed = [c for r in models for c in r['checks'] if c['status'] == 'failed']
    passed = [c for r in all_results for c in r['checks'] if c['status'] == 'passed']
    rows = ['# AdamW设备更新与模型训练验收', '',
        '日期：2026-10-18。本地开发版，未推送GitHub；此前报告保持为历史记录。', '',
        '## 结论与明确限制', '',
        '标准torch.optim.AdamW普通单Tensor路径已跑通；FP16/FP32独立更新、一二阶矩、AMSGrad、maximize、zero/None梯度与CPU checkpoint恢复通过。',
        '三个FP32分类模型在关闭紧邻BatchNorm的冗余卷积bias后，完成四步独立CPU轨迹、显式CPU checkpoint恢复和eval检查。',
        '纯FP16整网仍有严格数值检查失败，不能把独立更新通过推广到完整FP16模型训练兼容。',
        '原含冗余bias的FP32卷积/残差网络也有参数轨迹失败；原测试及失败位置保留在下表，没有放宽容差或计作通过。', '',
        '## 本轮实现与执行位置', '',
        '- 新增lerp_.Scalar、addcmul_、addcdiv_、浮点标量add_、sqrt/maximum前向与受限out。',
        '- 三个更新算子复用既有SHAVE算术，以FP32 opmath计算并在写回时转为原dtype；lerp采用稳定分支，',
        '  系数先按FP32舍入再选择分支，覆盖0.5附近与±最大有限值的极端输入；',
        '  addcmul/addcdiv保持value*a*b及value*a/b的顺序，没有替换为低精度乘法。',
        '- sqrt新增SHAVE入口，直接复用resident_activation.c已有的整数开方；maximum新增独立IEEE位排序，保留subnormal比较与NaN传播。',
        '- 参数、梯度消费和moments更新在NPU；标准Torch的step在CPU，偏差修正用Python数值，再作为标量常量参与NPU计算。',
        '  这些是原优化器控制流，不是CPU参数/动量算术回退；不能宣传为全部控制逻辑在NPU或完全零拷贝。',
        '- PyTorch把标量add_分派为Tensor重载时，允许无梯度CPU零维实数常量；没有引入CPU大Tensor混合算术。',
        '- 没有改写torch.optim.AdamW、torch.cuda、用户模型；C++扩展、session DLL及既有ELF不变。',
        '- 新增两个源码/ELF/汇编/manifest伙伴及构建入口；无新增依赖、模型下载、图片或持久图缓存。', '',
        '支持边界：no_grad更新、FP16/FP32、完整密集排列、匹配shape/dtype/device，Python数值超参数；',
        'foreach=False、fused=False、capturable=False、differentiable=False。不认证Adam、稀疏梯度、Tensor超参数、AMP或高阶autograd。',
        '拒绝广播、部分存储、不同stride的重叠别名；精确self-alias先读到独立结果再写回。sqrt/maximum未实现autograd，NaN payload和signed-zero ties不作CPU逐位认证。', '',
        '## CPU参照与测试口径', '',
        f'Windows x64 / Core Ultra 7 258V / NPU {new[0]["final_stats"]["hardware_device"]} / API driver_version={new[0]["final_stats"]["driver_version"]} / 编译器8.3。',
        f'Python {sys.version.split()[0]} / Torch {new[0]["final_stats"]["torch_version"]}；现有.conda/npunlock，CPU1线程。更新测试种子61；整网种子53、关闭oneDNN。',
        '普通assert_close仍为FP16 rtol=0.003/atol=0.0005，FP32 rtol=0.00003/atol=0.000003；没有为AdamW放宽。',
        f'实际使用的本地torch/optim/adam.py SHA-256：`{reference_sha}`。', '',
        ('修复前边界重放：AdamW在参数weight decay后遇到aten.lerp.Scalar_out缺口；' if baseline else
         '当前重跑未提供修复前基线，只审计现有实现；修复前缺口见首轮报告。'),
        (f'修复前torch_intel_npu.py SHA-256：`{baseline["source_sha256"]["torch_intel_npu.py"]}`。' if baseline else ''),
        '不支持的优化器组合可能已部分更新参数/状态，异常后应从原始checkpoint重建。', '',
        '## 更新算子与优化器状态', '',
        '| 检查 | 数值比较/位模式 | NPU调用 |', '|---|---|---:|']
    for c in new[0]['checks']:
        rows.append(f'| {c["name"]} | {c["pointwise_comparisons"]}项pointwise；{c["sqrt_bit_patterns"]:,}种sqrt输入 | {c["npu_calls"]} |')
    rows += ['', 'sqrt有限输出与FP64独立参考逐位一致，含全部65,536个half编码、8,192个FP32位模式及特殊值；另检查65,537元素分块。',
        '本机Torch FP32 sqrt在三个样本上与独立FP64/math.sqrt参照相差1 ULP；NPU与独立参考一致，未为迎合CPU近似结果修改开方。',
        '12组配置为两种dtype × 默认betas/no-decay/自定义betas/maximize/AMSGrad/组合；使用完整密集转置参数。',
        '单参数配置lr=0.002；默认betas=(0.9,0.999)，自定义=(0.1,0.95)，weight_decay为默认0.01、0或组合0.1。',
        '每组4次实际更新、中间None梯度不提交图且step不递增、零梯度正常递增，最后一次从CPU checkpoint恢复。',
        '逐项对照参数/exp_avg/exp_avg_sq/可选max_exp_avg_sq/CPU step，检查布局及从未获得梯度的参数不分配state。',
        '更新区间的显式host copies等于首次上传的标量常量数；逐步变化的偏差修正常量可继续上传，不能声称所有热步没有host copies。',
        '19种非法输入/别名/grad模式调用在无新增图提交时拒绝；saved-tensor version guard通过。', '',
        'FP16 epsilon边界单列：默认eps=1e-8在half中成为零，零梯度时CPU/NPU均产生NaN。该项只证明行为一致，不算健康训练通过。',
        '普通FP16检查显式使用eps=1e-4，仍保持half参数/half moments，没有自动提高epsilon或使用FP32 master state。', '',
        '## 整网训练与失败诊断', '',
        '沿用MLP(5→8→3)、Conv+默认BN和三层卷积残差网络，batch4、CrossEntropyLoss。AdamW lr=0.001；',
        'FP32 eps=1e-8，FP16 eps=1e-4；其余默认betas/weight_decay，foreach/fused关闭。三步训练后CPU checkpoint恢复，继续第四步。',
        'CPU模型保持独立轨迹；没有重设CPU参数追随NPU。另设独立诊断shadow：把NPU更新前参数/moments复制到CPU，',
        '以同一份NPU梯度调用标准CPU AdamW，严格检查本次参数与moments更新；它不替代独立模型轨迹。验证复制均在hot-copy统计区间之外。',
        '关闭BN前bias的对照保持全部其他初始权重和RNG消耗不变；只在测试模型中显式去掉紧邻BN的bias，后端不自动改模型。', '',
        '| BN前卷积bias | 模型/精度 | 独立轨迹状态 | 四步loss或首个失败位置 |', '|---|---|---|---|']
    for r in models:
        for c in r['checks']:
            detail = ' → '.join(f'{v:.8g}' for v in c['losses']) if c['status'] == 'passed' else next(
                line for line in c['traceback'].splitlines() if 'identical-gradient AdamW update checked=True' in line)
            if c['status'] == 'failed':
                magnitude = re.search(r'Greatest absolute difference: ([^ ]+)',c['error']).group(1)
                compared = next((line.strip() for line in c['traceback'].splitlines()
                                 if line.strip().startswith(('close(p,','close(p.grad,','close(value,'))), 'CPU comparison')
                detail += f'; {compared}; max_abs={magnitude}'
            rows.append(f'| {r["bn_bias"]} | {c["name"]} | {c["status"]} | {detail} |')
    rows += ['', '含bias的FP32 Conv与残差网络在第1步先偏离BN前卷积bias；相同梯度AdamW更新检查通过。',
        '理论上此bias被BN抵消；微小反向归约残差经过小epsilon的AdamW归一化会被放大，这是与SGD不同的敏感处。',
        '关闭冗余bias后三个FP32网络通过，支持这一判断；不能据此声称浮点反向逐位等同CPU或忽略任意微小梯度。',
        '纯FP16参数/梯度量化与二阶矩下溢仍会造成明显轨迹偏差；本轮不保证纯FP16整网质量或长期收敛，优先使用FP32训练。', '',
        '## 最终计数与资源', '',
        f'- 更新算子/状态/epsilon边界/拒绝契约16组通过；SGD分类11组通过；迁移{supported}组功能通过、{blocked}组明确blocked；既有设备33/训练11/兼容性5组通过。',
        f'- AdamW两种模型配置共12组：{12-len(failed)}组严格通过，{len(failed)}组独立轨迹失败，失败组不计作通过。',
        f'- 上述最终验收与显式失败诊断共{total_calls:,}次真实NPU调用，{len(passed)}组检查passed，另{blocked}组blocked、{len(failed)}组failed。',
        '  不含开发期重复实验或修复前边界重放；passed中包含非有限epsilon行为一致性检查，不能全部解释为模型训练通过。',
        '- 4项classification载体、3项INT64载体、5项源码导出离线回归，共12项通过。',
        '- 49组既有回归未调用lerp；最后系数边界修复后重跑所有AdamW路径及分类/迁移套件，不重复无关旧算子回归。',
        '- 全部进程无context failure，各检查live buffer无增量，shutdown后live/cache/scalar/quarantine/session归零。', '',
        '## 使用、复现与后续', '', '```python',
        'import torch', 'import torch_intel_npu',
        'model = model.float().to("intel_npu")',
        'optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,',
        '    foreach=False, fused=False, capturable=False, differentiable=False)', '```', '',
        '卷积后紧接BatchNorm时，在模型定义中显式使用Conv2d(...,bias=False)。checkpoint仍先把参数/状态转到CPU，',
        '不使用原始NPU Tensor序列化/deepcopy。Adam本体、LayerNorm/GELU/softmax/Embedding、bmm/SDPA、AMP和GPU+NPU协同未在本轮补齐。', '',
        '```powershell', "foreach ($case in @('primitives','optimizer','contracts')) {",
        "  .\\run_torch_npu.ps1 -Script .\\torch_npu_adamw_demo.py -ScriptArguments @('--case',$case)", '}',
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_classification_demo.py -ScriptArguments @('--case','training','--optimizer','adamw')",
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_classification_demo.py -ScriptArguments @('--case','training','--optimizer','adamw','--no-bn-bias')", '```', '',
        '两条AdamW整网命令保留严格失败，当前会返回非零；分别保存training.json与training-no-bn-bias.json。',
        '完整审计还需重跑classification四个SGD用例、migration四个进程、Device/Training Quick及compatibility；',
        '生成上述JSON后执行本脚本；修复前边界基线可选，提供时额外审计。旧基线需原始后端版本，不能用新版本生成blocked基线。',
        '按用户约定审计后清理原始JSON/日志/编译响应，保留源码、内核与Markdown；清理后需重跑生成审计输入。', '',
        '实现语义参考[PyTorch Lerp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Lerp.h)、',
        '[CPU PointwiseOps](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp)。',
        '[PyTorch性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide#disable-bias-for-convolutions-directly-followed-by-a-batch-norm)',
        '也建议关闭紧邻BatchNorm的卷积bias；这里没有声称把这种使用建议自动写入后端。来源关系详见REFERENCES.md。', '',
        '## 当前SHA-256', '', '```text']
    fingerprints = {**new[0]['source_sha256'], **models[0]['source_sha256'], **migration[0]['source_sha256']}
    for name in ('npunlock-probe/research/build_optimizer_kernels.py','npunlock-probe/research/build_resident_kernels.py',
                 'npunlock-probe/session-runtime/npunlock_session.dll',
                 'npunlock/tests/fixtures/npu4000/resident/sqrt-f32.elf','npunlock/tests/fixtures/npu4000/resident/maximum-f32.elf'):
        fingerprints[name] = digest(ROOT/name)
    for name,sha in fingerprints.items():
        rows += [name,sha]
    rows += ['```','']
    (OUT/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    print(f'Audited {len(passed)} passed, {blocked} blocked and {len(failed)} failed trajectory checks; {total_calls:,} NPU calls')


if __name__ == '__main__':
    main()
