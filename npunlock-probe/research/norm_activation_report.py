"""Audit LayerNorm/GELU runs, retaining historical failures when present."""
import json
from classification_report import ROOT, RESULTS, audit, digest, read

OUT = RESULTS/'norm-activation'


def main():
    new = [json.loads((OUT/f'{case}.json').read_text()) for case in ('layernorm','gelu','contracts','training')]
    assert [len(r['checks']) for r in new] == [3,4,1,6]
    assert all(r['source_sha256'] == new[0]['source_sha256'] for r in new)
    failures = [c for r in new for c in r['checks'] if c['status'] == 'failed']
    assert len(failures) <= 1
    if failures:
        assert failures[0]['name'] == 'layernorm-torch.float16-mixed-True'
        assert 'AssertionError' in failures[0]['traceback'] and 'gradient' in failures[0]['error']
    mixed_fixed = not failures
    assert all(c['comparisons'] == 24 for c in new[1]['checks'])
    assert all(c['status'] == 'passed' and c['steps'] == 4 and c['independent_cpu_trajectory']
               and c['losses'][-1] < c['losses'][0] for c in new[3]['checks'])
    assert new[2]['checks'][0]['rejected_without_submit'] == 13
    migration = [read(RESULTS/f'torch-migration/after-{case}.json') for case in ('models','sgd','updates','boundaries')]
    migrated = [c for r in migration for c in r['checks']]
    assert sum(c['status'] == 'passed' for c in migrated) == 31
    assert sum(c['status'] == 'blocked' for c in migrated) == 5
    for name in ('layernorm','gelu'):
        row = next(c for c in migrated if c['name'] == name)
        assert row['forward_backward_cpu_reference']
    classification = [read(RESULTS/f'classification/{case}.json') for case in ('logsoftmax','nll','contracts','training')]
    adamw = [read(RESULTS/f'adamw/{case}.json') for case in ('primitives','optimizer','contracts')]
    old = [read(RESULTS/f'torch-npu-{case}.json') for case in ('device','training')]
    old += read(RESULTS/'compatibility/compatibility.json')['cases']
    assert sum(len(r['checks']) for r in old) == 49
    assert sum(len(r['checks']) for r in classification) == 11
    assert sum(len(r['checks']) for r in adamw) == 16
    all_results = new+migration+classification+adamw+old
    for result in all_results:
        audit(result)
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = json.loads((fixture/'resident/manifest.json').read_text())
    for name, sha in manifest['source_sha256'].items():
        assert digest(fixture/name) == sha
    for name, entry in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == entry['elf_sha256']
        assert digest(fixture/f'resident/{name}.s') == entry['assembly_sha256']
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    assert digest(ROOT/'npunlock-probe/torch-device/backend.cpp') == build['source_sha256']
    assert digest(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd') == build['extension_sha256']
    assert digest(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll') == build['session_sha256']
    runs = json.loads((OUT/'regression.json').read_text())
    assert len(runs) == 15
    assert [(r['name'],r['returncode']) for r in runs if r['returncode']] == ([] if mixed_fixed else [('new-layernorm',1)])
    checks = [c for r in all_results for c in r['checks']]
    counts = {status: sum(c['status'] == status for c in checks) for status in ('passed','blocked','failed')}
    calls = sum(r['final_stats']['npu_calls'] for r in all_results)
    rows = ['# LayerNorm、GELU与残差前馈训练验收', '',
        '日期：2026-10-18。本地开发版；公开source-release与GitHub未改动。历史报告保留原结果。', '',
        f'Windows x64 / Core Ultra 7 258V / NPU {new[0]["final_stats"]["hardware_device"]} / API driver_version={new[0]["final_stats"]["driver_version"]} / 编译器8.3 / Torch {build["torch_version"]}。',
        '新增检查固定CPU单线程、关闭MKLDNN、随机种子73；独立CPU模型与NPU模型共享初始参数和输入。', '',
        '## 结论与限制', '',
        '- 同dtype FP16/FP32 LayerNorm前向、保存统计与一阶梯度通过有限CPU对照。',
        '- GELU默认erf公式与tanh公式的FP16/FP32前后向及out通过；不保证逐位同CPU。',
        '- 含LayerNorm、Linear、GELU、残差与CrossEntropy的前馈网络：FP16/FP32 SGD及FP32 AdamW，',
        '  两种GELU各四步，损失、全部参数梯度与更新按独立CPU训练轨迹比较，六种组合通过。',
        ('- FP16输入配FP32 affine的独立LayerNorm检查已通过有序统计修复；更广整网边界见[后续报告](../layernorm-mixed/REPORT.md)。' if mixed_fixed else
         '- FP16输入配FP32 affine的宽LayerNorm仍有严格参数梯度失败，**暂不认证此混合组合**。'),
        ('  使用有源代码的SHAVE Welford统计，混合normalized width<=65,536；其他参数沿用原契约。' if mixed_fixed else
         '  没有放宽容差、把失败改成通过或重置CPU轨迹。当前完整验收命令因此返回非零。'),
        '- 本轮没有认证完整Transformer、通用bmm/SDPA/softmax/Embedding、AMP或二阶梯度。', '',
        '## 执行路径与契约', '',
        '设备仍为intel_npu PrivateUse1；标准nn/F入口直接调用本后端，无CPU张量算术回退。',
        'LayerNorm复用FP32归约、平方、rsqrt与广播，前向使用完整统计，保存统计按输入/affine dtype舍入；',
        'FP16参数梯度逐行经过FP16累加，匹配单线程CPU参考的缓冲语义。此路径提交次数较多，尚未优化性能。',
        'rank 1–4、非空normalized_shape匹配尾部、eps为正有限FP32、可选weight/bias，',
        '输入/affine须为完整密集存储，遵守全局1,048,576元素与4MiB allocation界限；不提供native out重载。',
        '混合FP32 affine必须有FP32 weight；只有FP32 bias的半精度组合，CPU本身的反向也会类型报错，本后端明确拒绝。',
        '高动态范围近常量行、不同CPU线程归约顺序、全部FP32编码和长时间训练未全面认证。', '',
        'GELU使用四个新SHAVE FP32内核；FP16输入设备内转FP32，结果/梯度转回FP16。',
        'erf采用A&S 7.1.26公开数学系数，自行实现求值；tanh采用exp及小值奇多项式；',
        '复用本项目exp_negative与IEEE subnormal辅助函数，解析一阶导数按各自公式计算。',
        '使用已有Power载体（前向第二只读绑定复用输入），不改代码窗口大小、MemRef布局、任务调度或驱动指纹校验。',
        '仅GELU构建单元对IEEE辅助函数禁止展开并使用-Os，原辅助源码与已有内核不改；不引入libm/OpenVINO runtime。',
        'GELU rank 0–4、最多1,048,576元素，内部65,536元素分块；out只用于无梯度、精确shape/dtype且不别名。', '',
        '## 测试与严格失败', '',
        '容差沿用FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005，NaN位置须一致；',
        '依据被比较Tensor的dtype选择阈值，FP32 affine梯度仍使用严格FP32门槛。',
        'LayerNorm包含rank 1–4、多尾轴/全轴、affine四种选项、完整密集转置、单元素、4096宽层、',
        ('17×4096跨块、常量/近常量行及NaN/Inf；本次独立混合组完整通过，不据此认证任意混合整网。' if mixed_fixed else
         '17×4096跨块、常量/近常量行及NaN/Inf；混合组在首个严格失败处停止，不能视为全部混合用例通过。'),
        'GELU每个dtype/公式24项比较：4097点网格、特殊/极大值、密集排列、65,537元素跨块、',
        '全部65,536个FP16位模式或8192个随机FP32位模式，以及标量；包括直接out前向和反向。',
        '另有13项无提交拒绝检查和两项保存Tensor版本保护。FP16全部编码覆盖不等于正确舍入认证。', '',
        '| 检查组 | 状态 | 数量/说明 |', '|---|---|---|']
    for result in new:
        for c in result['checks']:
            detail = (str(c.get('comparisons',''))+'项比较') if 'comparisons' in c else ('4步独立训练' if 'losses' in c else '13项拒绝/2项版本保护')
            if c['status'] == 'failed':
                detail = c['error'].replace('\n',' ').replace('|','/')
            rows.append(f'| {c["name"]} | {c["status"]} | {detail} |')
    rows += ['', '半精度CPU LayerNorm即使使用FP32 affine也会将参数梯度累加缓冲存成FP16；',
        'FP32均值/逆标准差及乘加顺序的微小差异可跨过该FP16舍入边界，再体现为FP32梯度的一次half ULP跳变。',
        ('后续改编CPU Welford分组/级联顺序并用SHAVE整数融合运算复现，原失败已通过；没有保证所有CPU架构逐位一致。' if mixed_fixed else
         '这是失败机制解释，不能据此认定本实现与CPU逐位一致；严格失败保留供后续修复。'), '',
        '### 六组训练损失', '', '| 配置 | 四步loss |', '|---|---|']
    for c in new[3]['checks']:
        rows.append(f'| {c["name"]} | '+', '.join(f'{v:.8g}' for v in c['losses'])+' |')
    rows += ['', '## 回归与资源', '',
        f'- 最终同版验收合计**{calls:,}次真实NPU调用**，{counts["passed"]}组passed、{counts["blocked"]}组明确blocked、{counts["failed"]}组严格failed。',
        '- 既有device Quick 33、training Quick 11、兼容性5、SGD分类11及AdamW更新/状态/边界16组通过。',
        '- 迁移检查31组功能通过/5组明确未支持；新增LayerNorm/GELU迁移探针包含非平凡输入、CPU前向与反向对照。',
        '- 剩余blocked：'+', '.join(c['name'] for c in migrated if c['status']=='blocked')+'。',
        '- 新增及迁移/分类/AdamW的逐组live buffer增量为零；所有进程shutdown后live/cached/scalar/quarantine/session归零，context无失败。',
        '- 12项离线测试通过（INT64载体3、分类载体4、源码导出5）；8份相关Python源码通过语法解析。',
        '- AdamW的FP16默认epsilon零梯度NaN一致性仅为既有边界行为检查，不计作健康训练认证；旧整网FP16限制仍在。',
        '- 本轮不复测历史AdamW卷积BN冗余bias诊断、低位LLM质量或DPU适配器性能；不声称性能提升。', '',
        '## 复现', '', '```powershell',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_norm_activation_demo.py',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\norm_activation_regression.py',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/norm_activation_report.py', '```', '',
        ('当前独立LayerNorm/GELU完整入口已通过；更广的混合整网精度检查与已知失败见后续报告。' if mixed_fixed else
         '前两个完整验收入口保留严格混合失败而返回1；第三步仅在除该已记录失败外的全部检查及指纹均符合时写报告。'),
        '最终验收JSON/日志和编译响应在报告审计后清理；复现需先重跑上述测试。', '',
        '## 数学与语义参考', '',
        '- [A&S p.299，7.1.26](https://personal.math.ubc.ca/~cbm/aands/page_299.htm)：公开数学公式和系数，非第三方源码移植。',
        '- [PyTorch Activation.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/Activation.cpp)：两种公式与一阶导数。',
        '- [PyTorch layer_norm_kernel.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)：保存统计及参数梯度缓冲的dtype。', '',
        '## 源码及内核指纹（SHA-256）', '', '```text']
    files = list(new[0]['source_sha256'])+['torch_npu_migration_demo.py',
        'npunlock-probe/research/build_gelu_kernels.py','npunlock-probe/research/build_resident_kernels.py',
        'npunlock-probe/research/norm_activation_regression.py','npunlock-probe/research/norm_activation_report.py',
        'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd',
        'npunlock-probe/session-runtime/npunlock_session.dll','npunlock/tests/fixtures/npu4000/resident/manifest.json']
    files += [f'npunlock/tests/fixtures/npu4000/resident/{name}.elf' for name in manifest['kernels'] if name.startswith('gelu-')]
    rows += [name+' '+digest(ROOT/name) for name in files]
    rows += ['```','']
    (OUT/'REPORT.md').write_text('\n'.join(rows), encoding='utf-8')
    print(f'Audited {counts}; {calls:,} NPU calls')


if __name__ == '__main__':
    main()
