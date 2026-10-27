"""Audit the alpha fix and retain numerical findings as a Markdown report."""
import hashlib
import json
from pathlib import Path

from classification_report import audit, digest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'npunlock-probe/research/results'
OUT = RESULTS/'mixed-training-rounding'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def legacy_core_sha():
    source = (ROOT/'torch_intel_npu.py').read_text(encoding='utf-8')
    new_guard = '''        _update_coefficient(alpha)
        if x.dtype == torch.float16 and isfinite(alpha) and abs(alpha) > 65504:
            raise ValueError("NPU in-place alpha must be representable in FP16")
'''
    old_guard = '''        if type(alpha) not in (int, float):
            raise ValueError("NPU in-place alpha must be real")
'''
    new_scale = '''            # Tensor add_ converts alpha to the input dtype before FP32 opmath.
            scale = _cast(_scalar(alpha, x.dtype), torch.float32)
            right = _binary_op(right, scale, kind="mul")
'''
    old_scale = '''            right = _binary_op(right, alpha, kind="mul")
'''
    assert source.count(new_guard) == source.count(new_scale) == 1
    return hashlib.sha256(source.replace(new_guard, old_guard).replace(new_scale, old_scale).encode()).hexdigest()


def resources(result):
    final = result['final_stats']
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_buffer_bytes',
        'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    assert all(c.get('live_buffer_delta', 0) == 0 for c in result.get('checks', []))


def main():
    before, after = (read(OUT/f'trajectory-{stage}.json') for stage in ('before', 'after'))
    baseline = read(OUT/'updates-before.json')
    updates = read(OUT/'updates-after.json')
    mixed = [read(RESULTS/'layernorm-mixed'/name) for name in ('training-fp32-loss.json', 'training.json')]
    migration = [read(RESULTS/'torch-migration'/f'alpha-fix-{case}.json') for case in ('updates', 'sgd')]
    norm = read(RESULTS/'norm-activation/training.json')
    adam = read(RESULTS/'adamw/optimizer.json')
    current = [updates, *mixed, *migration, norm, adam]
    for result in current: audit(result)
    audit(dict(after, checks=[]))
    for result in (before, baseline): resources(result)
    old_sha = legacy_core_sha()
    assert old_sha == '29d81705ddc2d020008088133d6a333d7a52688ff9be0ee8b554fe6a3a8a3922'
    for result in (before, baseline):
        assert result['source_sha256']['torch_intel_npu.py'] == old_sha
        # Historical diagnostic scripts evolved; their old hashes remain evidence, not current-file assertions.
        for name, sha in result['source_sha256'].items():
            if name not in ('torch_intel_npu.py', 'npunlock-probe/research/mixed_training_rounding_probe.py',
                            'npunlock-probe/research/add_alpha_regression.py'):
                assert digest(ROOT/name) == sha
    assert before['first_independent_failure'] == {'mode': 'tanh', 'step': 2, 'parameter': '1.weight'}
    assert after['first_independent_failure'] is None and after['status'] == 'passed'
    assert len(before['runs']) == len(after['runs']) == 2
    for previous, fixed in zip(before['runs'], after['runs']):
        assert previous['mode'] == fixed['mode']
        assert previous['initial_input_and_parameters_sha256'] == fixed['initial_input_and_parameters_sha256']
        assert len(previous['steps']) == len(fixed['steps']) == 4
        for run in (previous, fixed):
            for step in run['steps']:
                assert len(step['layers']) == 4
                assert all(v['unequal'] == 0 for layer in step['layers'] for v in layer['identical_input_and_seed'].values())
        for step in fixed['steps']:
            assert step['loss']['outside_tolerance'] == 0
            assert all(v['outside_tolerance'] == 0 for v in (*step['parameters'].values(), *step['updated_weights'].values()))
    assert all(v['unequal'] == 0 for step in after['runs'][1]['steps'] for v in step['parameters'].values())
    assert all(layer['forward']['unequal'] == 0 and layer['upstream_gradient']['unequal'] == 0
               for step in after['runs'][1]['steps'] for layer in step['layers'])
    assert [len(r['checks']) for r in current] == [6, 2, 2, 5, 14, 6, 13]
    assert all(c['status'] == 'passed' for r in (updates, mixed[0], *migration, norm, adam) for c in r['checks'])
    assert all(c['status'] == 'failed' and 'step=0, parameter=1.weight' in c['error'] for c in mixed[1]['checks'])
    assert all(c['steps'] == 4 and c['fp32_loss'] for c in mixed[0]['checks'])
    assert updates['checks'][1]['unequal_finite_or_inf'] == 2
    assert updates['checks'][2]['half_patterns'] == 65536 and updates['checks'][2]['alphas'] == 10
    assert baseline['checks'][1]['status'] == 'failed' and '10070 / 65536' in baseline['checks'][1]['error']
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = read(fixture/'resident/manifest.json')
    assert len(manifest['kernels']) == 20
    for name, sha in manifest['source_sha256'].items(): assert digest(fixture/name) == sha
    for name, entry in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == entry['elf_sha256']
        assert digest(fixture/f'resident/{name}.s') == entry['assembly_sha256']
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    for name, key in (('backend.cpp', 'source_sha256'), ('_torch_intel_npu.pyd', 'extension_sha256')):
        assert digest(ROOT/'npunlock-probe/torch-device'/name) == build[key]
    assert digest(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll') == build['session_sha256']
    passed = sum(c['status'] == 'passed' for r in current for c in r['checks'])
    failed = sum(c['status'] == 'failed' for r in current for c in r['checks'])
    calls = sum(r['final_stats']['npu_calls'] for r in current)+after['final_stats']['npu_calls']
    rows = ['# 混合训练第三步偏差：Tensor add_ alpha修复', '',
        '日期：2026-10-18。本地开发版；此前报告作为历史记录保留，公开source-release及GitHub未改动。', '',
        '## 结果', '',
        '- 原GELU(tanh)+显式FP32损失的第三步FP32 LayerNorm参数梯度失败已修复。',
        '- none/tanh两组混合网络均通过四步独立CPU对照；原FP32梯度阈值rtol=3e-5/atol=3e-6不变。',
        '- tanh样例四步的四层前向、上游梯度及所有参数梯度均逐位相同；更新权重仍有少量可接受的FP32舍入差异。',
        '- FP16分类损失的两组严格失败仍然存在，不把它们计作支持；未认证通用AMP、长期收敛或任意Transformer。', '',
        '## 原因与修改', '',
        '网络为Linear(16,16)→LayerNorm(16)→GELU→Linear(16,3)，Linear/激活FP16，LayerNorm参数FP32。',
        '固定seed127、batch4、CPU单线程且关闭oneDNN，SGD lr=.01/momentum=.8/foreach=False/fused=False。',
        '保留none后tanh的原随机初始化/输入顺序，CPU/NPU各自连续更新，不将CPU轨迹重置为NPU权重。',
        '修复前首次严格失败为step=2（第3步）的1.weight；逐层重放发现所有8步、4层在相同输入/参数/上游梯度下局部结果均相同。',
        f"该步FP32 LayerNorm weight梯度最大差由{before['runs'][1]['steps'][2]['parameters']['1.weight']['max_abs']:.12g}降为{after['runs'][1]['steps'][2]['parameters']['1.weight']['max_abs']:.12g}。",
        '差异从首步SGD权重更新开始，第二步放大后改变第三步前向，最终跨过LayerNorm half累加边界。', '',
        '原Tensor add_把alpha直接当FP32标量乘梯度；本机CPU先将alpha转换为Tensor dtype。',
        '`-.01`转为FP16是`-0.01000213623046875`，不能用FP32的`-0.009999999776482582`替代。',
        '修复在torch_intel_npu.py的_add_inplace：按输入dtype生成标量，再用设备转换/FP32乘加及最终dtype转换完成更新。',
        '同时在提交前拒绝bool、超FP32范围和有限FP16 alpha超65504；仍要求no_grad、相同shape/dtype/device。',
        '没有更改GELU/LayerNorm C源码、20份resident ELF/汇编、manifest、autograd、C++/DLL/PYD或torch.optim.SGD。',
        '这项修改直接影响标准优化器所调用的设备更新算子，用户不需要改模型结构或手工重写更新公式。', '',
        '## 可核对样本', '',
        '| 场景 | CPU更新 | 原NPU更新 | 修复后NPU |', '|---|---:|---:|---:|',
        '| none首步0.weight[61]，旧值0.00015676021575927734，梯度0.1925048828125 | -0.0017690658569335938 | -0.0017681121826171875 | -0.0017690658569335938 |',
        '| tanh首步0.weight[51]，旧值-0.01300811767578125，梯度0.09649658203125 | -0.01397705078125 | -0.01396942138671875 | -0.01397705078125 |', '',
        '两条轨迹的初始输入+参数SHA-256：', '', '```text']
    rows += [f"{r['mode']} {r['initial_input_and_parameters_sha256']}" for r in after['runs']]
    rows += ['```', '', '## 训练结果与保留边界', '',
        '| 损失 | GELU | 修复后结果 |', '|---|---|---|']
    rows += [f"| FP32 | {c['name'].removeprefix('mixed-training-')} | passed；loss："+
             ', '.join(f'{v:.9g}' for v in c['losses'])+' |' for c in mixed[0]['checks']]
    rows += [f"| FP16 | {c['name'].removeprefix('mixed-training-')} | failed；{c['error'].splitlines()[0]} |" for c in mixed[1]['checks']]
    rows += ['', 'FP16损失仍在第一步进入更新前失败，沿用[上一轮的log_softmax保存值舍入诊断](../layernorm-mixed/REPORT.md)。',
        '显式FP32损失在CPU/NPU两侧同时使用，不是后端偷偷切换精度，也不是FP16损失已修复。',
        '默认none配置第四步仍有少量half梯度差异，在原容差内；不能将本轮结果泛化为所有Tensor逐位一致。', '',
        '## add_数值边界', '',
        '- 原256元素参数向量的四个样本放在64元素向量中严格重放，通过；保留向量上下文，未拿CPU标量尾部作同一运算顺序。',
        '- 65,536种half编码、与其滚动排列形成的另一操作数，10种alpha（含极小、极大、0/1、Inf/NaN），655,360个结果与CPU数值相等。',
        '  NaN位置一致；这不是所有输入对的穷举，也不认证signed-zero或NaN payload逐位相同。',
        '- 原版在-.01首项就有10,070/65,536数值不等；修复后10项均通过，未放宽这项相等判定。',
        '- 密集转置、精确self-alias、version递增和保存Tensor版本保护通过。',
        '- CPU小张量/标量尾部会额外把alpha*y中间量转回half；本NPU仍保留FP32中间量。',
        f"  明确的4元素边界有{updates['checks'][1]['unequal_finite_or_inf']}个数值不等，最大差{updates['checks'][1]['max_abs']:.12g}，仅按原half容差通过。",
        '  不为迎合CPU向量宽度而在NPU按32元素拆分算法；尚不承诺跨CPU线程/架构/形状的逐位轨迹相同。', '',
        '## 回归与资源', '',
        '| 检查 | 通过/失败 | NPU调用 |', '|---|---:|---:|']
    labels = ['add_精度/契约', '混合网络FP32损失', '混合网络FP16损失边界', '既有更新接口', 'SGD七种配置×两种dtype', '残差前馈训练', 'AdamW状态/恢复/epsilon边界']
    for label, result in zip(labels, current):
        good = sum(c['status']=='passed' for c in result['checks'])
        rows.append(f"| {label} | {good}/{len(result['checks'])-good} | {result['final_stats']['npu_calls']} |")
    rows += [f"| 修复后逐层辅助诊断：两组×四步×四层 | 全部局部结果相同 | {after['final_stats']['npu_calls']} |", '',
        f'保留的最终修复后验收及辅助诊断共**{calls:,}次NPU调用**，主检查**{passed}组通过、{failed}组已知严格失败**。',
        f"修复前辅助诊断{before['final_stats']['npu_calls']+baseline['final_stats']['npu_calls']:,}次另计；探索中的中间复测不并入最终验收统计。",
        '逐组live buffer增量为零，所有进程shutdown后的live/cache/scalar/quarantine/session均归零，无context failure。',
        '12项INT64/分类载体/源码导出离线检查通过。未重复完整设备/卷积/低位模型回归；本轮没有性能计时或长期训练。',
        '修复会增加FP16标量到FP32的设备转换，不能据此宣称速度提升；没有启用持久图缓存、下载模型或安装依赖。', '',
        '## 复现', '', '```powershell',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\add_alpha_regression.py',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\mixed_training_rounding_probe.py',
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\layernorm_mixed_acceptance.py -ScriptArguments @('--case','training','--fp32-loss')",
        ".\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\layernorm_mixed_acceptance.py -ScriptArguments @('--case','training')",
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_migration_demo.py -ScriptArguments @('--child','updates','--label','alpha-fix')",
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_migration_demo.py -ScriptArguments @('--child','sgd','--label','alpha-fix')",
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_norm_activation_demo.py -ScriptArguments @('--case','training')",
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_adamw_demo.py -ScriptArguments @('--case','optimizer')", '```', '',
        'FP16损失命令按既有严格失败返回非零。报告审计需要上述原始结果及修复前诊断；后者只能在历史后端上生成。',
        '审计脚本通过精确撤销本次两个源码片段，在内存中核对旧后端指纹；不会写回旧代码或用当前结果伪造基线。',
        '修复前诊断脚本有后续命令行/标量尾部检查调整，旧脚本指纹作为历史记录，不宣称其与当前文件相同。',
        '按约定验收后清理原始JSON，保留代码、报告和运行所需内核；清理后不能直接重跑旧结果审计。', '',
        '## 参考', '',
        '[PyTorch ufunc/add.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ufunc/add.h)的标量T与向量接口用于核对alpha dtype和乘加顺序。',
        '本机安装的Loops.h、Half.h、vec256_half.h用于核对CPU尾部与向量处理；本轮没有复制CPU SIMD实现。',
        '链接main仅作导航，本机参考以Torch 2.14.1+cpu与下列安装文件指纹为准。', '',
        '## SHA-256', '', '```text', f'torch_intel_npu.py.before {old_sha}']
    files = sorted(set(name for r in (*current, after) for name in r['source_sha256']))
    files += ['npunlock-probe/research/mixed_training_rounding_report.py',
              'npunlock-probe/session-runtime/npunlock_session.dll',
              '.conda/npunlock/Lib/site-packages/torch/optim/sgd.py',
              '.conda/npunlock/Lib/site-packages/torch/include/ATen/native/cpu/Loops.h',
              '.conda/npunlock/Lib/site-packages/torch/include/torch/headeronly/util/Half.h',
              '.conda/npunlock/Lib/site-packages/torch/include/ATen/cpu/vec/vec256/vec256_half.h']
    rows += [f'{name} {digest(ROOT/name)}' for name in dict.fromkeys(files)]
    rows += ['```', '']
    (OUT/'REPORT.md').write_text('\n'.join(rows), encoding='utf-8')
    print(json.dumps({'passed':passed, 'known_failed':failed, 'npu_calls':calls, 'resident_kernels':20}, indent=2))


if __name__ == '__main__':
    main()
