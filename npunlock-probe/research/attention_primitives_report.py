"""Audit softmax/bmm hardware checks and write a Markdown-only acceptance report."""
import json
from classification_report import ROOT, RESULTS, audit, digest, read

OUT = RESULTS/'attention-primitives'
CASES = ('softmax', 'bmm', 'bmm-large', 'contracts', 'attention', 'training')


def main():
    new = [read(OUT/f'{case}.json') for case in CASES]
    assert [len(r['checks']) for r in new] == [2, 2, 2, 1, 2, 2]
    assert all(r['source_sha256'] == new[0]['source_sha256'] for r in new)
    assert all(c['status'] == 'passed' for r in new for c in r['checks'])
    assert len(new[3]['checks'][0]['rejected_without_submit']) == 12
    assert new[3]['checks'][0]['saved_version_guards']
    for result in new:
        for c in result['checks']:
            copies = c.get('compute_copies', c.get('step_copies', []))
            copies = [copies] if isinstance(copies, dict) else copies
            assert all(v['copies'] == v['scalar_uploads']+v['error_flag_reads'] for v in copies)
    for c in new[-1]['checks']:
        assert c['steps'] == len(c['losses']) == 4 and c['all_parameters_and_gradients']
        assert c['losses'][-1] < c['losses'][0] and c['explicit_fp32_classification_loss']
    old = [read(RESULTS/f'classification/{case}.json') for case in ('logsoftmax', 'nll', 'contracts')]
    boundaries = read(RESULTS/'torch-migration/attention-boundaries.json')
    assert [len(r['checks']) for r in old] == [2, 2, 1]
    assert all(c['status'] == 'passed' for r in old for c in r['checks'])
    assert len(boundaries['checks']) == 11
    assert sum(c['status'] == 'passed' for c in boundaries['checks']) == 7
    assert {c['name'] for c in boundaries['checks'] if c['status'] == 'blocked'} == {
        'embedding', 'batched-matmul', 'sdpa', 'native-checkpoint'}
    all_results = new+old+[boundaries]
    for result in all_results:
        audit(result)
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = json.loads((fixture/'resident/manifest.json').read_text())
    assert len(manifest['kernels']) == 22
    for name, sha in manifest['source_sha256'].items():
        assert digest(fixture/name) == sha
    for name, row in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == row['elf_sha256']
        assert digest(fixture/f'resident/{name}.s') == row['assembly_sha256']
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    for path, key in (('npunlock-probe/torch-device/backend.cpp', 'source_sha256'),
                      ('npunlock-probe/torch-device/_torch_intel_npu.pyd', 'extension_sha256'),
                      ('npunlock-probe/session-runtime/npunlock_session.dll', 'session_sha256')):
        assert digest(ROOT/path) == build[key]
    for name in ('softmax', 'softmax-backward'):
        for stage in ('compile', 'assemble', 'link'):
            row = json.loads((RESULTS/f'{name}-{stage}.json').read_text())
            assert row['exit_code'] == row['worker_status'] == row['tool_return'] == 0
    checks = [c for r in all_results for c in r['checks']]
    assert len(checks) == 27 and sum(c['status'] == 'passed' for c in checks) == 23
    calls = sum(r['final_stats']['npu_calls'] for r in all_results)
    lines = ['# softmax、bmm与小型attention训练验收', '',
        '日期：2026-10-18。本地开发版；公开source-release与GitHub未改。', '',
        'Windows x64 / Core Ultra 7 258V / NPU 8086:643e / API driver_version=1005540 / 编译器8.3 / Torch 2.14.1+cpu / Python 3.12.14。',
        'CPU固定单线程、关闭MKLDNN、随机种子83；CPU/NPU使用相同初始输入和参数，独立执行训练轨迹。', '',
        '## 结果', '',
        '- 新增11组功能检查全部通过：softmax、bmm、小型attention的FP16/FP32前后向及四步SGD训练。',
        '- 原分类logsoftmax/NLL/拒绝契约5组回归通过；迁移边界7通过/4明确未支持。',
        f'- 最终有效验收共23组通过、4组明确未支持，**{calls:,}次NPU调用**；另12项离线回归通过。',
        '- 容差未放宽：FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005；NaN位置须一致。',
        '- 所有检查live buffer无泄漏；所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。', '',
        '## 实现与使用边界', '',
        '标准PrivateUse1与AutogradPrivateUse1注册，直接使用torch.softmax/F.softmax/nn.Softmax和torch.bmm。',
        'softmax仅支持rank 1–4、最后一轴、非空且总元素<=65,536；显式dtype=FP32通过。',
        '两个新SHAVE C内核分别计算稳定softmax与解析梯度y*(g-sum(g*y))，复用自编exp_negative/IEEE辅助函数。',
        'FP16在设备内转换成FP32计算后转回；保留subnormal尾部。NaN、含+Inf或全-Inf行与CPU一致传播NaN。',
        '前后向out均需无梯度、精确shape/dtype且不别名；内部half_to_float=True拒绝。',
        '初轮out检查发现新softmax反向out的schema关键字为grad_input，现已修复注册并通过直接调用检查；log_softmax的out关键字仍为out。', '',
        'bmm仅支持(B,M,K)×(B,K,N)、相同dtype与批次，不广播；FP16/FP32，输入与输出遵守全局1,048,576元素/4MiB存储限制。',
        '输入可为覆盖完整存储的密集转置；不接受部分/重叠/zero-stride视图。',
        '小形状按批次合并展开乘积，再进行FP32补偿归约；工作区大于65,536时拆批次并复用已有二维矩乘的M/N/K分块。',
        '尚未接入DPU batched GEMM；本轮扩展兼容性，不提供吞吐或加速结论。', '',
        '普通三维@/matmul仍拒绝；没有接入MultiheadAttention、fused SDPA、Embedding、mask/dropout专用接口、通用Transformer、AMP或二阶梯度。',
        '小型四步训练不代表长期收敛；此前混合训练严格轨迹失败保持为独立已知边界。', '',
        '```python', 'import torch', 'import torch_intel_npu', '',
        '# q/k/v为同dtype、同批次的三维intel_npu Tensor。',
        'scores = torch.bmm(q, k.transpose(1, 2)) * (q.shape[-1] ** -0.5)',
        'output = torch.bmm(torch.softmax(scores, dim=-1), v)', '```', '',
        '## 覆盖范围', '',
        'softmax每种dtype共52项数值比较：rank 1–4、宽度1/3/17/257/4096/65536、显式FP32输出、',
        '完整密集排列、前后向out、NaN/Inf、极大logits、FP32 subnormal与非有限上游梯度。',
        'bmm小形状每种dtype覆盖3种形状及双侧/左侧/右侧梯度；另检查5种分块形状(B,M,K,N)：',
        '(3,16,64,32)、(2,16,257,17)、(5,128,1,128)、(65537,1,1,1)、(1,1,65537,1)。',
        '包括超过单块的展开工作区、81,920元素输出、65,537批次和K维串流；每组均核对forward/out/双方梯度。',
        '契约包含12项不新增图提交的拒绝检查，以及softmax输出与bmm输入被原地修改后的保存版本保护。',
        '组合attention核对Q/K/V三路梯度；小网络含LayerNorm、Q/K/V Linear、attention、投影、残差、mean pooling及三分类head。',
        '两种dtype各四步标准SGD，foreach=False/fused=False、lr=.025，CrossEntropy显式使用FP32 logits。', '',
        '| 检查 | 状态 | NPU调用 |', '|---|---|---:|']
    lines += [f'| {c["name"]} | {c["status"]} | {c["npu_calls"]} |' for c in checks]
    lines += ['', '### 四步训练损失', '', '| dtype | loss |', '|---|---|']
    lines += [f'| {c["name"]} | '+', '.join(f'{v:.9g}' for v in c['losses'])+' |' for c in new[-1]['checks']]
    lines += ['', '训练计数区间包住forward/backward/step，不包含显式CPU参考及下载比较。',
        '首步只上传新增标量常量并读取一次分类错误标志；随后每步只有一次4字节错误标志读取。',
        '独立softmax/bmm/attention计算区间仅允许新增标量常量上传；不使用CPU Tensor算术回退。',
        '计数覆盖显式Tensor复制，不覆盖驱动图内DMA、布局搬运或编译器；共享内存不表示无数据移动。', '',
        '## 构建、回归与复现', '',
        '新增resident_softmax.c、两份ELF/汇编及构建脚本；原20份resident ELF/汇编未重编，22份与manifest逐一核对。',
        'C++设备源码、PYD与session DLL保持原构建指纹，无新增依赖、模型、图表或持久图缓存。',
        '未重复历史全套36项迁移、卷积大形状、低位LLM或性能测试；本轮采用共享分类与迁移边界回归。', '',
        '```powershell', './.conda/npunlock/python.exe -B npunlock-probe/research/build_softmax_kernels.py',
        "foreach ($case in @('softmax','bmm','bmm-large','contracts','attention','training')) {",
        "    ./run_torch_npu.ps1 -Script ./torch_npu_attention_demo.py -ScriptArguments @('--case',$case)", '}',
        "foreach ($case in @('logsoftmax','nll','contracts')) {",
        "    ./run_torch_npu.ps1 -Script ./torch_npu_classification_demo.py -ScriptArguments @('--case',$case)", '}',
        "./run_torch_npu.ps1 -Script ./torch_npu_migration_demo.py -ScriptArguments @('--child','boundaries','--label','attention')",
        './.conda/npunlock/python.exe -B npunlock-probe/research/attention_primitives_report.py', '```', '',
        '构建需既有外部MoviTools。审计需要上述原始JSON/编译状态，验收后按约定清理，仅保留源码、内核和本报告。', '',
        '接口参考：[PyTorch softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)、',
        '[PyTorch bmm](https://docs.pytorch.org/docs/2.14/generated/torch.bmm.html)；实际out关键字按本机安装schema核对。', '',
        '## SHA-256', '', '```text']
    names = set(new[0]['source_sha256']) | {
        'npunlock-probe/research/attention_primitives_report.py', 'npunlock-probe/research/build_softmax_kernels.py',
        'npunlock-probe/research/build_resident_kernels.py', 'npunlock-probe/torch-device/backend.cpp',
        'npunlock-probe/torch-device/_torch_intel_npu.pyd', 'npunlock-probe/session-runtime/npunlock_session.dll'}
    names.update(f'npunlock/tests/fixtures/npu4000/resident/{name}.{ext}'
                 for name in ('softmax', 'softmax-backward') for ext in ('elf', 's'))
    lines += [f'{name} {digest(ROOT/name)}' for name in sorted(names)]
    lines += ['```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'passed': 23, 'blocked': 4, 'npu_calls': calls, 'resident_kernels': 22}))


if __name__ == '__main__':
    main()
