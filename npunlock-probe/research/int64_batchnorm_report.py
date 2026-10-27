"""Audit this round's hardware results before retaining Markdown only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'npunlock-probe/research/results'
OUT = RESULTS/'int64-batchnorm'


def read(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status'] == 'passed', path
    return result


def healthy(result):
    final = result['final_stats']
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_buffer_bytes',
                                      'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))


def main():
    integer, models = (read(OUT/f'{name}.json') for name in ('integer', 'models'))
    assert integer['source_sha256'] == models['source_sha256']
    for name, digest in integer['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    assert len(integer['checks']) == 3 and len(models['checks']) == 6
    for result in (integer, models):
        healthy(result)
        assert all(c['status'] == 'passed' and c['live_buffer_delta'] == 0 for c in result['checks'])
        assert sum(c['npu_calls'] for c in result['checks']) == result['stats_before_shutdown']['npu_calls']
    carrier = integer['checks'][1]['carrier']
    assert carrier['invocations'] == carrier['ranges'] == 1
    assert carrier['contracts'] == [{'tile': 0, 'elements': 2, 'span_bytes': 8, 'dtype': 'f32'}]
    assert carrier['scheduling'] == 'single_complete_pair'
    assert carrier['kernel_elf_sha256'] == integer['source_sha256']['npunlock/tests/fixtures/npu4000/resident/int64-add.elf']
    assert len(integer['checks'][2]['rejected_without_graph_submit']) == 14
    for c in models['checks']:
        assert len(c['losses']) == 4 and c['checkpoint_resume'] and c['running_statistics_counter_eval_and_reset']
        assert c['step_host_copies'] == [s+int(c['momentum'] is None) for s in c['scalar_uploads']]
        if c['momentum'] is not None:
            assert c['step_host_copies'][2:] == [0, 0]
    migration = [read(RESULTS/f'torch-migration/after-{case}.json') for case in ('models','sgd','updates','boundaries')]
    checks = [c for r in migration for c in r['checks']]
    assert len(checks) == 36
    supported = sum(c['status'] == 'passed' for c in checks)
    blocked = sum(c['status'] == 'blocked' for c in checks)
    assert supported >= 27 and supported+blocked == 36
    for r in migration:
        healthy(r)
        for name, digest in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        assert all(c['live_buffer_delta'] == 0 for c in r['checks'])
    device = read(RESULTS/'torch-npu-device.json')
    training = read(RESULTS/'torch-npu-training.json')
    compatibility = read(RESULTS/'compatibility/compatibility.json')
    for r in (device, training, *compatibility['cases']):
        healthy(r)
    new_calls = sum(r['stats_before_shutdown']['npu_calls'] for r in (integer, models))
    migration_calls = sum(r['stats_before_shutdown']['npu_calls'] for r in migration)
    regression_calls = device['final_stats']['npu_calls'] + training['final_stats']['npu_calls'] + sum(
        r['stats_before_shutdown']['npu_calls'] for r in compatibility['cases'])
    build = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    assert build['source_sha256'] == integer['source_sha256']['npunlock-probe/torch-device/backend.cpp']
    assert build['extension_sha256'] == integer['source_sha256']['npunlock-probe/torch-device/_torch_intel_npu.pyd']
    assert hashlib.sha256((ROOT/'npunlock-probe/session-runtime/npunlock_session.dll').read_bytes()).hexdigest() == build['session_sha256']
    manifest = json.loads((ROOT/'npunlock/tests/fixtures/npu4000/resident/manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/'npunlock/tests/fixtures/npu4000'/name).read_bytes()).hexdigest() == digest
    assert manifest['kernels']['int64-add']['elf_sha256'] == carrier['kernel_elf_sha256']
    lines = ['# INT64存储与默认BatchNorm迁移验收', '', '日期：2026-10-16。本地开发版，尚未推送GitHub。', '',
        '## 本轮实现', '',
        '- C++设备存储新增INT64，保留4MiB单allocation上限；INT64最多524,288元素、rank 0–4。FP16/FP32上限不变。',
        '- INT64支持同dtype同步上传/下载、连续NPU复制/clone、fill/zero和单元素item。NPU内复制/填充使用raw-word DMA。',
        '- fill接受范围内Python int或有限整数值float；不提供小数截断、整数/浮点cast、通用整数算术或NPU非连续复制。',
        '- rank-0 INT64 counter.add_(Python int, alpha=1)由SHAVE unsigned32低字加法、高字加法和进位实现，模2^64回绕。',
        '  F32载体仅传输两组原始位；未把INT64数值转成FP32。计数操作可在grad模式执行，整数不参与autograd。',
        '- 默认BatchNorm2d的num_batches_tracked随模型迁移至NPU；使用标准nn模块，未替换forward、修改torch.cuda或使用CPU张量算术回退。', '',
        '## SHAVE布局诊断', '',
        '初版采用普通Power载体时，本机编译器生成8次调用，每次仅一个32位字，无法完成跨字进位；数值检查检测到失败。',
        '修复为ReferenceSW单tile完整双字载体；加载前验证一组range/一次invocation、三个完整双字MemRef。',
        f'最终内核代码{carrier["code_bytes"]}字节，载体代码窗口{carrier["window_bytes"]}字节；仅替换受指纹校验的代码窗口。',
        '没有改写驱动任务指针、DMA地址或同步协议。编译器若再次拆分双字，离线/运行时契约会明确拒绝。', '',
        '## 环境与验证口径', '',
        f'- Windows x64 / Core Ultra 7 258V / NPU {integer["final_stats"]["hardware_device"]} / API driver_version={integer["final_stats"]["driver_version"]}。',
        f'- Python {integer["python"].split()[0]} / Torch {integer["final_stats"]["torch_version"]} / 图编译器8.3；现有.conda/npunlock，未安装依赖或下载模型。',
        '- 重建PyTorch C++扩展与新增SHAVE计数内核；已有浮点SHAVE ELF和session DLL未改动。',
        '- 整数用torch.equal/item做精确比较；没有用浮点相对容差接受大整数误差。',
        '- 浮点CPU参考1线程、关闭oneDNN、随机种子47；FP16 rtol=0.005/atol=0.0008，FP32 rtol=0.0001/atol=0.00001。',
        '  每个模型有相同初始权重的独立CPU训练轨迹，不按NPU结果重设CPU参数。', '',
        '## 整数检查', '',
        '- 存储往返：0、±1、2^24+1、2^32−1、±(2^53+1)、INT64_MIN/MAX，以及含FP32 NaN/Inf字模式的大整数。',
        '- 五种rank、CPU非连续输入显式pack、工厂/复制/clone/zero、524,288元素最大容量通过逐位比较。',
        '- 11对进位/负数/符号回绕/NaN字模式加法精确通过；从2^53起连续20次加1得到2^53+20。',
        '- 每次计数更新为一次SHAVE加法加一次NPU DMA写回；14个非法形状/dtype/转换/参数/复制场景在没有新增图提交时拒绝。', '',
        '## 默认BatchNorm模型训练、恢复与状态', '',
        'Conv2d(1→2,3×3)、BatchNorm2d(2)、ReLU、全局池化、Linear(2→2)；输入(2,1,4,4)，MSE，SGD lr=0.01/momentum=0.8/foreach=False/fused=False。',
        '六组均使用默认track_running_stats=True。每组3步训练后，显式CPU checkpoint保存参数、running mean/variance、INT64计数与优化器动量，',
        'weights_only=True加载到新NPU模型，继续第4步；输出、loss、输入/参数梯度、参数、动量和全部buffers与CPU对照通过。',
        'eval使用running buffers且计数不变；reset_running_stats恢复mean=0、variance=1、counter=0。', '',
        '| 精度 | momentum | affine | 四步NPU loss | 各步显式host copies | 其中首次标量上传 | NPU调用 |',
        '|---|---|---|---|---|---|---:|']
    for c in models['checks']:
        lines.append(f'| {c["dtype"]} | {c["momentum"]} | {c["affine"]} | '+
                     ' → '.join(f'{x:.8g}' for x in c['losses'])+
                     f' | {c["step_host_copies"]} | {c["scalar_uploads"]} | {c["npu_calls"]} |')
    lines += ['',
        '复制区间只包含NPU前向/反向/step，排除输入准备、验证下载和checkpoint搬运。',
        '固定momentum=0.1第三/第四步的显式host payload copies为0。momentum=None按PyTorch Python控制流每次前向读取8字节计数，',
        '且1/步数系数变化会首次上传新标量；它不是零拷贝模式。统计不包含驱动内部DMA，本轮不测吞吐或性能收益。', '',
        '## 最终验收与回归', '',
        f'- 新增9组真实NPU检查通过：{new_calls:,}次调用；新增3项离线单任务/双字载体检查与5项源码导出回归通过。',
        f'- 普通迁移套件：{supported}组功能通过、{blocked}组明确未支持，共{migration_calls:,}次NPU调用（INT64修复前为25/11）。',
        f'- 原有设备33组、训练11组、兼容性5组通过：{regression_calls:,}次调用。',
        f'- 最终验收总计：{new_calls+migration_calls+regression_calls:,}次真实NPU调用，不含开发诊断与失败重跑。',
        '- 全部进程无context failure，各检查live buffer无增量；shutdown后live/cache/scalar/quarantine/session归零。', '',
        '## 尚未支持与下一步', '',
        ('有界CrossEntropyLoss在后续分类轮次通过，详见 ../classification/REPORT.md。' if supported >= 28 else
         'INT64 target可以存储和复制，但CrossEntropyLoss仍缺稳定log_softmax与NLL前后向；本轮没有打通分类训练。'),
        'INT64并非通用整数计算后端，INT8量化推理与该存储能力是不同接口。BatchNorm仍受原4D/通道/样本数/容量契约限制。',
        '当前迁移仍明确拒绝：'+ '、'.join(c['name'] for c in checks if c['status'] == 'blocked')+'。',
        '没有认证CUDA运行时/扩展、AMP、多流、GPU+NPU协同、其他设备/驱动或长时间训练；后续扩展范围见对应轮次报告。', '',
        '## 复现', '', '```powershell',
        '.\\build_torch_npu_device.ps1',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_batchnorm_demo.py -ScriptArguments "--case","integer"',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_batchnorm_demo.py -ScriptArguments "--case","models"',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_migration_demo.py',
        '.\\run_torch_npu.ps1 -Device -Quick', '.\\run_torch_npu.ps1 -Training -Quick',
        '.\\run_torch_npu.ps1 -Script .\\torch_npu_compat_demo.py',
        '$env:PYTHONPATH = "$PWD;$PWD\\npunlock\\python"',
        '.\\.conda\\npunlock\\python.exe -B test_torch_npu_int64.py',
        '.\\.conda\\npunlock\\python.exe -B tools/test_source_release.py',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/int64_batchnorm_report.py', '```', '',
        '通常使用随源码提供的已校验ELF；修改计数C源码后，先用已有Movi环境执行build_integer_kernel.py，或完整build_resident_kernels.py。',
        '按用户约定清理原始JSON/日志后，重算审计需先重新执行验收生成数据。', '',
        '## 最终代码指纹（SHA-256）', '', '```text']
    for name, digest in integer['source_sha256'].items():
        lines += [name, digest]
    lines += ['npunlock-probe/session-runtime/npunlock_session.dll', build['session_sha256'], '```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Audited 9 new and 85 migration/regression checks; {new_calls+migration_calls+regression_calls:,} NPU calls')


if __name__ == '__main__':
    main()
