"""Audit classification hardware acceptance and retain a Markdown-only report."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'npunlock-probe/research/results'
OUT = RESULTS/'classification'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status'] == 'passed', (path, result['status'])
    return result


def audit(result):
    final = result['final_stats']
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers','cached_buffer_blocks','cached_buffer_bytes',
        'cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    for name, sha in result.get('source_sha256', {}).items():
        assert digest(ROOT/name) == sha, name
    assert all(c.get('live_buffer_delta', 0) == 0 for c in result['checks'])


def main():
    new = [read(OUT/f'{case}.json') for case in ('logsoftmax','nll','contracts','training')]
    assert all(r['source_sha256'] == new[0]['source_sha256'] for r in new)
    assert [len(r['checks']) for r in new] == [2, 2, 1, 6]
    assert all(c['status'] == 'passed' for r in new for c in r['checks'])
    guard = new[2]['checks'][0]
    assert len(guard['structural_rejections_without_submit']) == 12
    assert guard['device_checked_invalid_targets'] == 4 and guard['target_version_guard']
    for c in new[3]['checks']:
        assert len(c['losses']) == 4 and c['checkpoint_resume'] and c['independent_cpu_trajectory']
        assert c['loss_gradient_on_identical_logits']
        assert c['step_host_copies'] == [n+1 for n in c['scalar_uploads']]
    migration = [read(RESULTS/f'torch-migration/after-{case}.json') for case in ('models','sgd','updates','boundaries')]
    checks = [c for r in migration for c in r['checks']]
    supported = sum(c['status'] == 'passed' for c in checks)
    blocked = sum(c['status'] == 'blocked' for c in checks)
    assert supported >= 28 and supported+blocked == 36
    assert all(r['source_sha256'] == migration[0]['source_sha256'] for r in migration)
    batchnorm = read(RESULTS/'int64-batchnorm/models.json')
    device, training = (read(RESULTS/f'torch-npu-{case}.json') for case in ('device','training'))
    compat = read(RESULTS/'compatibility/compatibility.json')['cases']
    for r in (*new, *migration, batchnorm, device, training, *compat):
        audit(r)
    fixture = ROOT/'npunlock/tests/fixtures/npu4000'
    manifest = json.loads((fixture/'resident/manifest.json').read_text())
    for name, sha in manifest['source_sha256'].items():
        assert digest(fixture/name) == sha, name
    for name, row in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf') == row['elf_sha256'], name
        assert digest(fixture/f'resident/{name}.s') == row['assembly_sha256'], name
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    assert digest(ROOT/'npunlock-probe/torch-device/backend.cpp') == build['source_sha256']
    assert digest(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd') == build['extension_sha256']
    assert digest(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll') == build['session_sha256']
    new_calls = sum(r['final_stats']['npu_calls'] for r in new)
    migration_calls = sum(r['final_stats']['npu_calls'] for r in migration)
    old = [batchnorm, device, training, *compat]
    old_calls = sum(r['final_stats']['npu_calls'] for r in old)
    rows = ['# 有界分类训练与FP16 BatchNorm修复验收', '',
        '日期：2026-10-16。本地开发版，尚未推送GitHub；此前迁移/INT64报告保留为历史记录。', '',
        '## 结果与范围', '',
        '- 新增自定义SHAVE log_softmax、log_softmax backward、NLL packet与NLL backward四个内核；',
        '  PrivateUse1及AutogradPrivateUse1接入标准F.log_softmax、F.nll_loss/nn.NLLLoss、F.cross_entropy/nn.CrossEntropyLoss。',
        '- FP16/FP32输入；FP16通过设备内FP32计算返回原dtype，不是DPU原生FP16分类内核。',
        '- log_softmax：密集rank 1–4，仅最后一轴，总元素<=65,536；显式dtype=FP32与无梯度非别名out通过。',
        '- NLL/CE：(C)配标量标签或(N,C)配(N)标签；连续、同设备INT64硬标签；weight=None、label_smoothing=0。',
        '  支持none/mean/sum、signed INT64 ignore_index，且N*(C+4)<=65,536；不提供空输入/空间分类/soft targets/类别权重/NLL out重载。',
        '- 计算及标签合法性判断在NPU；每次NLL/CE前向同步下载4字节错误标志，发现越界抛IndexError。',
        '  标签始终按两个原始32位字通过byte DMA，不转成FP32数值；CPU仅构造形状索引和Python常量。',
        '- C++设备扩展、session DLL、既有SHAVE ELF不变；无需新增Python依赖、OpenVINO runtime、模型下载或持久图缓存。', '',
        '## SHAVE数值与布局', '',
        '完整行先减最大值、补偿求sum(exp)，再按(x-max)-log(sum)顺序写回；反向为g-exp(logp)*sum(g)。',
        'exp/log独立实现区间缩减与Taylor/atanh展开，复用既有FP32/subnormal辅助函数；未复制第三方libm源码/表。',
        '这是CPU容差验证的近似实现，不声明通用libm正确舍入、全FP32位模式或所有长度认证。',
        'ReferenceSW Power载体编译后要求一个range/一次invocation、三个完整MemRef；',
        '只接受(width,rows,1,1)，以及单行的(1,1,width,1)物理布局。若编译器拆行或改变布局则拒绝，保留既有代码窗口指纹校验。',
        'NLL选择目标位置直接取值；跳过未选中/被忽略的NaN，反向对被忽略位置直接写零，避免0*NaN。', '',
        '## FP16整网失败与修复', '',
        '初版六组分类训练中，FP16 Conv+BatchNorm的输入梯度最大绝对差为0.0118103，卷积权重梯度为0.00445557；',
        '另五组通过。没有放宽容差或把失败改记为通过。相同NPU logits上的分类loss梯度对照通过。',
        '隔离相同BN输入与incoming gradient发现CPU纯FP16保存的mean/invstd是half，而原NPU路径始终保留float。',
        '例如CPU invstd=[8.5390625,17.71875]，原NPU为[8.5355768,17.7227974]。',
        '修复共享_batch_norm：纯FP16训练时先把保存统计舍入到half再转FP32归一化/反向；',
        '方差累加、原始均值以及running mean/variance更新仍使用FP32；有FP32参数/缓冲区的路径保留FP32保存统计。',
        '修复后以原测试种子、数据、四步独立CPU轨迹和原容差重跑；六组全部通过，已有六种BN模型组合也通过。',
        '这一结果覆盖所列小模型，仍不能保证任意纯FP16深层训练轨迹一致。', '',
        '## 环境与验证口径', '',
        f'Windows x64 / Core Ultra 7 258V / NPU {new[0]["final_stats"]["hardware_device"]}；API driver_version={new[0]["final_stats"]["driver_version"]}；编译器8.3。',
        f'Python {new[0]["python"].split()[0]}、Torch {new[0]["final_stats"]["torch_version"]}，复用.conda/npunlock。CPU1线程、关闭oneDNN，分类测试随机种子53。',
        '逐项assert_close：FP16 rtol=0.003/atol=0.0005；FP32 rtol=0.00003/atol=0.000003；NaN按equal_nan核对。',
        'CPU仅为测试参照；模型上传后独立训练，不按NPU结果重设CPU参数。不测吞吐/TOPS/功耗或长时间收敛。', '',
        '## 算子、拒绝契约与模型', '',
        '| 检查组 | CPU比较次数 | NPU调用 |', '|---|---:|---:|']
    for r in new[:2]:
        for c in r['checks']:
            rows.append(f'| {c["name"]} | {c["comparisons"]} | {c["npu_calls"]} |')
    rows += ['', 'log_softmax覆盖rank1–4、宽度1/17/257/4,096/65,536、大小共同偏置、NaN/+Inf/all−Inf/partial−Inf、',
        'exp尾部−80/−90/−100/−103/−104，以及Inf/NaN incoming gradient。',
        'NLL/CE覆盖三种reduction、普通/部分/全部忽略、正ignore_index、INT64_MIN、未选中NaN、',
        'unbatched与batched total_weight，以及2,049行的FP16有效标签计数舍入边界。',
        '12种结构/参数/别名拒绝没有新增图提交；四种越界标签（含2^32、2^53+1）在设备检查后安全拒绝；',
        'backward前修改target触发saved-tensor version检查，context保持健康。', '',
        '| 模型/精度 | 四步NPU loss | 各步显式host copies | 首次标量上传 |', '|---|---|---|---|']
    for c in new[3]['checks']:
        rows.append(f'| {c["name"]} | '+ ' → '.join(f'{v:.8g}' for v in c['losses'])+
            f' | {c["step_host_copies"]} | {c["scalar_uploads"]} |')
    rows += ['', 'MLP：5→8→3；Conv：1→2卷积、默认BatchNorm、ReLU、全局池化、Linear；',
        '残差网络：三层卷积、两个默认BatchNorm、残差ReLU、全局池化和Linear。batch4，输入(4,5)或(4,1,4,4)。',
        '标准CrossEntropyLoss与SGD(lr=0.025,momentum=0.8,foreach=False,fused=False)，三步后显式CPU checkpoint恢复并继续第四步。',
        '每步检查输出/loss/输入梯度/所有参数梯度/参数/动量/全部BN buffers；另核对相同logits上的分类梯度，末尾eval。',
        '复制统计区间仅含NPU forward/backward/step，排除验证下载、输入准备和checkpoint；每步copies=首次标量上传+一次4B错误标志读取。',
        '统计不包含驱动内部DMA，不能描述成完全零拷贝或异步训练。', '',
        '## 最终计数与资源', '',
        f'- 分类11组全部passed：{new_calls:,}次真实NPU调用。',
        f'- 迁移套件{supported}组功能passed、{blocked}组blocked：{migration_calls:,}次调用；原分类轮为28/8。',
        f'- 既有默认BatchNorm模型6组、设备Quick33组、训练Quick11组、原兼容性5组passed：{old_calls:,}次调用。',
        f'- 最终验收合计{new_calls+migration_calls+old_calls:,}次真实NPU调用，不含开发失败/诊断/重跑。',
        '- 4项分类载体离线测试、3项INT64载体回归、5项源码导出回归，共12项通过。',
        '- 每组live buffer无增量，各进程无context failure；shutdown后live/cache/scalar/quarantine/session全部归零。', '',
        '仍blocked：'+ '、'.join(c['name'] for c in checks if c['status'] == 'blocked')+'。',
        '未认证CUDA运行时/扩展、torch.xpu合并、AMP、二阶梯度、多流、GPU+NPU协同、其他硬件/驱动或任意torchvision/HF模型。', '',
        '## 复现与参考', '', '```powershell',
        "foreach ($case in @('logsoftmax','nll','contracts','training')) {",
        "  .\\run_torch_npu.ps1 -Script .\\torch_npu_classification_demo.py -ScriptArguments @('--case',$case)", '}',
        ".\\run_torch_npu.ps1 -Script .\\torch_npu_batchnorm_demo.py -ScriptArguments @('--case','models')",
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_migration_demo.py',
        '.\\run_torch_npu.ps1 -Device -Quick', '.\\run_torch_npu.ps1 -Training -Quick',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_compat_demo.py',
        '$env:PYTHONPATH = "$PWD;$PWD\\npunlock\\python;$PWD\\tools"',
        '.\\.conda\\npunlock\\python.exe -B -m unittest test_torch_npu_classification test_torch_npu_int64 test_source_release',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/classification_report.py', '```', '',
        '修改分类C源码时用现有Movi环境执行build_classification_kernels.py；默认无需重建即可使用配套ELF。',
        '按用户约定，报告审计后清理原始JSON/日志/编译响应；清理后需重新跑上述硬件检查生成审计输入。',
        '数值算法与API契约参考[PyTorch LogSoftmax](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)、',
        '[SoftMaxKernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)、',
        '[LossNLL](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp)、',
        '[Normalization](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Normalization.cpp)；调研与独立实现边界见根目录REFERENCES.md。', '',
        '## 当前SHA-256', '', '```text']
    for name, sha in new[0]['source_sha256'].items():
        rows += [name, sha]
    rows += ['npunlock-probe/torch-device/backend.cpp', build['source_sha256'],
        'npunlock-probe/session-runtime/npunlock_session.dll', build['session_sha256']]
    for name in ('log-softmax','log-softmax-backward','nll-packet','nll-backward'):
        rows += [f'npunlock/tests/fixtures/npu4000/resident/{name}.elf', manifest['kernels'][name]['elf_sha256']]
    rows += ['```', '']
    (OUT/'REPORT.md').write_text('\n'.join(rows), encoding='utf-8')
    print(f'Audited 11 new, 36 migration and 55 regression groups; {new_calls+migration_calls+old_calls:,} NPU calls')


if __name__ == '__main__':
    main()
