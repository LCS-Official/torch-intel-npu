"""Audit bounded matmul/SDPA, differentiable layouts and real-device regression."""
import json
from classification_report import ROOT, RESULTS, audit, digest, read

OUT = RESULTS/'sdpa'
CASES = ('matmul', 'matmul-large', 'sdpa', 'sdpa-masks', 'layouts', 'contracts', 'training')


def main():
    new = [read(OUT/f'{case}.json') for case in CASES]
    assert [len(r['checks']) for r in new] == [2, 2, 2, 2, 2, 1, 2]
    assert all(r['source_sha256'] == new[0]['source_sha256'] for r in new)
    assert all(c['status'] == 'passed' for r in new for c in r['checks'])
    assert len(new[5]['checks'][0]['rejected_without_submit']) == 24
    assert new[5]['checks'][0]['saved_version_guards']
    assert [len(c['cases']) for c in new[2]['checks']] == [7, 8]
    assert all('score-limit' in c['cases'] and c['one_sided_gradients'] for c in new[2]['checks'])
    assert 'prescale-overflow' in new[2]['checks'][1]['cases']
    assert all(c['mask_gradients'] and 'fully-masked' in c['cases'] for c in new[3]['checks'])
    assert all(c['concurrent_backward_threads'] == 2 and c['shared_permutation_session_creations'] == 1 for c in new[4]['checks'])
    for result in new:
        for c in result['checks']:
            assert all(v['copies'] == v['scalar_uploads']+v['error_flag_reads']
                       for v in c.get('compute_copies', c.get('step_copies', [])))
    for c in new[-1]['checks']:
        assert c['steps'] == len(c['losses']) == 4 and c['all_parameters_and_gradients']
        assert c['heads'] == 2 and c['causal'] and c['explicit_fp32_loss']
        assert c['losses'][-1] < c['losses'][0]
    primitives = [read(RESULTS/f'attention-primitives/{case}.json') for case in ('softmax', 'bmm')]
    boundaries = read(RESULTS/'torch-migration/sdpa-boundaries.json')
    device = read(RESULTS/'torch-npu-device.json')
    assert [len(r['checks']) for r in primitives] == [2, 2]
    assert len(device['checks']) == 33
    assert all(c['status'] == 'passed' for r in primitives+[device] for c in r['checks'])
    assert len(boundaries['checks']) == 11
    assert sum(c['status'] == 'passed' for c in boundaries['checks']) == 9
    assert {c['name'] for c in boundaries['checks'] if c['status'] == 'blocked'} == {'embedding', 'native-checkpoint'}
    for name, key in (('batched-matmul', 'batched_broadcast_forward_backward'), ('sdpa', 'causal_multihead_forward_backward')):
        assert next(c for c in boundaries['checks'] if c['name'] == name)[key]
    all_results = new+primitives+[device, boundaries]
    for result in all_results:
        audit(result)
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = json.loads((fixture/'resident/manifest.json').read_text())
    assert len(manifest['kernels']) == 23
    for name, sha in manifest['source_sha256'].items():
        assert digest(fixture/name) == sha
    for name, row in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == row['elf_sha256']
        assert digest(fixture/f'resident/{name}.s') == row['assembly_sha256']
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    native = (('npunlock-probe/torch-device/backend.cpp', 'source_sha256'),
              ('npunlock-probe/torch-device/_torch_intel_npu.pyd', 'extension_sha256'),
              ('npunlock-probe/session-runtime/npunlock_session.dll', 'session_sha256'))
    for path, key in native:
        assert digest(ROOT/path) == build[key]
    assert device['extension_sha256'] == build['extension_sha256'] and device['session_dll_sha256'] == build['session_sha256']
    for stage in ('compile', 'assemble', 'link'):
        row = json.loads((RESULTS/f'sdpa-softmax-{stage}.json').read_text())
        assert row['exit_code'] == row['worker_status'] == row['tool_return'] == 0
    checks = [c for r in all_results for c in r['checks']]
    counts = {s: sum(c['status'] == s for c in checks) for s in ('passed', 'blocked', 'failed')}
    assert counts == {'passed': 59, 'blocked': 2, 'failed': 0}
    calls = sum(r['final_stats']['npu_calls'] for r in all_results)
    lines = ['# 高维matmul、SDPA与可微布局验收', '',
        '日期：2026-10-18。本地开发版；公开source-release与GitHub未改。', '',
        'Windows x64 / Core Ultra 7 258V / NPU 8086:643e / driver API 1005540 / 图编译器8.3 / Torch 2.14.1+cpu / Python 3.12.14。',
        '新增检查固定CPU单线程、关闭MKLDNN、随机种子89；SDPA参考显式选Torch MATH路径，输入与初始参数相同，训练轨迹独立。', '',
        '## 验收结论', '',
        f'- 新增13组全部通过；加上既有设备/softmax/bmm/迁移边界回归，最终59通过、2明确未支持、0失败，共**{calls:,}次NPU调用**。',
        '- 另12项离线回归通过。FP32 rtol=3e-5/atol=3e-6，FP16 rtol=.003/atol=.0005；沿用原容差，NaN位置须一致。',
        '- 迁移边界现为9通过/2未支持；Embedding与原始NPU Tensor序列化仍明确拒绝。未重跑历史全套36项迁移，不以此计算API完成度。',
        '- 所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure；各新检查无live buffer净增。', '',
        '## 实现与边界', '',
        '`torch.matmul`/`@`支持同dtype FP16/FP32、rank 2–4，只广播batch维；mm仍限定二维，bmm的相同批次契约保持。',
        '广播后每个操作数及输出须各<=1,048,576元素；这一工作区界限比理论上可串流的范围保守，超过时在图提交前拒绝。',
        '输入覆盖完整offset-zero密集存储，转置由设备DMA materialize；内部复用bmm及既有M/N/K分块。',
        '反向用FP32计算矩乘并对广播轴求和，最后转换为原输入dtype；不提供向量/dot或二阶梯度。', '',
        '标准F.scaled_dot_product_attention通过PrivateUse1/AutogradPrivateUse1注册，没有替换Torch全局函数。',
        'Q/K/V须同rank 3D或4D、同FP16/FP32 dtype，batch/head允许广播，Q/K末维和K/V序列维匹配。',
        'dropout_p=0、enable_gqa=False；scale为有限FP32可表示实数，支持正/负/零值。',
        'attn_mask须同设备完整密集浮点Tensor、dtype为query dtype或FP32，可广播到分数；支持其一阶梯度。',
        '不接受CPU/布尔mask；is_causal=True不能同时传attn_mask。非方形因果采用upper-left对齐。',
        '分数总量<=65,536元素；各广播Q/K/V及输出<=1,048,576元素，遵守现有4MiB allocation界限。', '',
        'Q/K/V、分数与概率在设备内用FP32计算，最终结果/各梯度转回相应dtype；Q/K分别预缩放以避免可避免的点积溢出。',
        '新增resident_sdpa_softmax.c在现有验证过的载体中实现safe softmax及因果偏置；第二输入是设备广播的0/查询长度元数据，CPU不生成整张因果mask。',
        '在已测有限Q/K/V与上游下，全-Inf掩码行输出和对应Q/mask梯度为零；NaN/+Inf按参考传播。解析反向复用已有softmax backward与矩乘。',
        '这是SHAVE数学路径，非FlashAttention或DPU融合内核；没有性能计时、TOPS或加速结论。', '',
        '### 多头布局缺口与修复', '',
        '首轮网络在transpose(...).contiguous()合并head时因旧copy_的autograd拒绝失败；SDPA独立前后向当时已通过。',
        '新增设备内clone及一阶梯度，默认/preserve_format保留完整密集stride，contiguous_format生成连续布局，clone拥有独立storage。',
        'INT64 clone仍仅连续存储且无梯度；不提供clone.out、channels_last或部分/重叠存储扩展。',
        '原copy_及CPU/NPU可微传输拒绝策略未放开；网络结构未为绕过失败而改写。',
        '修复后两种dtype的clone/布局独立检查与原网络四步训练通过。', '',
        '末次复查补齐clone backward的运行时锁；双线程以非连续上游同时反向，只创建一个共用permutation会话，梯度及复制计数通过。最终全套在补锁后重跑。', '',
        '## 数值与模型覆盖', '',
        'matmul每种dtype检查7对二维至四维形状，双侧/左侧/右侧梯度、密集转置与out；另测双batch轴分块和K=65,537串流。',
        'SDPA每种dtype覆盖3D/4D、batch/head广播、正/负/零scale、长短两种非方形causal及65,536分数的合法上限。',
        '另测FP32 Q/K=1e20、scale=1e-20的预缩放溢出边界，结果/梯度与CPU一致且输出有限。',
        '浮点mask覆盖广播、FP16输入+FP32 mask、全屏蔽行、NaN/Inf和mask梯度；独立Q/K/V单侧梯度与无梯度推理入口通过。',
        '布局检查覆盖3D/4D、三种memory_format、独立storage、transpose().contiguous().view()反向。',
        '24项结构拒绝均不增加图提交；matmul和SDPA保存输入的原地修改版本保护通过。', '',
        '双头因果小网络含Q/K/V和输出Linear、两层LayerNorm、GELU前馈层、残差、mean pooling与三分类head。',
        '标准SGD lr=.025、foreach=False/fused=False，CrossEntropy显式使用FP32 logits；四步逐项比较loss、全部参数梯度与更新。', '',
        '| 新检查 | 状态 | NPU调用 |', '|---|---|---:|']
    lines += [f'| {c["name"]} | {c["status"]} | {c["npu_calls"]} |' for r in new for c in r['checks']]
    lines += ['', '| dtype | 四步loss |', '|---|---|']
    lines += [f'| {c["name"]} | '+', '.join(f'{v:.9g}' for v in c['losses'])+' |' for c in new[-1]['checks']]
    lines += ['', '| 回归 | 结果 | NPU调用 |', '|---|---|---:|',
        f'| 原softmax/bmm | 4通过 | {sum(r["final_stats"]["npu_calls"] for r in primitives)} |',
        f'| 既有Device Quick | 33通过 | {device["final_stats"]["npu_calls"]} |',
        f'| 迁移boundaries | 9通过/2未支持 | {boundaries["final_stats"]["npu_calls"]} |', '',
        '计算计数区间排除了输入上传、CPU对照与显式下载比较，只允许新增标量常量上传；训练另有每步一次4字节分类错误标志读取。',
        '这是显式Tensor copy计数，不覆盖驱动图内DMA、布局搬运或编译器；不等同于物理零数据移动。', '',
        '## 未验证范围', '',
        '未认证nn.MultiheadAttention/nn.Transformer、完整LLM训练、长期收敛、AMP、GQA、布尔mask、dropout、二阶梯度或GPU+NPU协同。',
        '未重复低位完整模型、非Quick大卷积与全部优化器训练。四步小网络结果不能外推为通用Transformer支持或NPU性能优势。',
        '原22份resident ELF/汇编内容与C++/PYD/session DLL保持；新增一个ELF/汇编，23份kernel逐一通过manifest核对。',
        '没有新增依赖、模型、图片或持久图缓存；保留原历史报告。', '',
        '## 复现', '', '```powershell',
        './.conda/npunlock/python.exe -B npunlock-probe/research/build_sdpa_softmax_kernel.py',
        "foreach ($case in @('matmul','matmul-large','sdpa','sdpa-masks','layouts','contracts','training')) {",
        "    ./run_torch_npu.ps1 -Script ./torch_npu_sdpa_demo.py -ScriptArguments @('--case',$case)", '}',
        "foreach ($case in @('softmax','bmm')) {",
        "    ./run_torch_npu.ps1 -Script ./torch_npu_attention_demo.py -ScriptArguments @('--case',$case)", '}',
        './run_torch_npu.ps1 -Device -Quick',
        "./run_torch_npu.ps1 -Script ./torch_npu_migration_demo.py -ScriptArguments @('--child','boundaries','--label','sdpa')",
        './.conda/npunlock/python.exe -B npunlock-probe/research/sdpa_report.py', '```', '',
        '构建需现有外部MoviTools；运行使用原环境。审计需原始JSON/编译状态，验收后按用户空间约定清理，只保留源码、内核与文字报告。', '',
        '接口参考：[PyTorch matmul](https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html)、',
        '[PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)；实际调用参数按本机ATen schema核对。', '',
        '## SHA-256', '', '```text']
    names = set(new[0]['source_sha256']) | {path for path, key in native} | {
        'npunlock-probe/research/sdpa_report.py', 'npunlock-probe/research/build_sdpa_softmax_kernel.py',
        'npunlock-probe/research/build_resident_kernels.py', 'torch_npu_migration_demo.py', 'torch_npu_device_demo.py',
        'npunlock/tests/fixtures/npu4000/resident/sdpa-softmax.elf', 'npunlock/tests/fixtures/npu4000/resident/sdpa-softmax.s'}
    lines += [f'{name} {digest(ROOT/name)}' for name in sorted(names)]
    lines += ['```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({**counts, 'npu_calls': calls, 'resident_kernels': 23}))


if __name__ == '__main__':
    main()
