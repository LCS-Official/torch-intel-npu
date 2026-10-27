"""Audit ordered LayerNorm statistics and retain broader mixed-model failures."""
import json
from classification_report import ROOT, RESULTS, audit, digest, read

OUT=RESULTS/'layernorm-mixed'


def main():
    before=json.loads((OUT/'diagnosis.json').read_text())
    new=[json.loads((OUT/f'{case}.json').read_text()) for case in ('replay','stress','contracts','training','training-fp32-loss')]
    assert [len(r['checks']) for r in new]==[1,22,1,2,2]
    assert all(r['source_sha256']==new[0]['source_sha256'] for r in new)
    assert new[0]['checks'][0]['input_sha256']==before['input_sha256']
    assert before['comparisons']['actual_npu']['bad']==1
    assert new[0]['checks'][0]['gradient_max_abs']==[0.,0.,0.]
    assert all(c['status']=='passed' for r in new[:3] for c in r['checks'])
    assert all(c['status']=='failed' and 'step=' in c['error'] and 'parameter=' in c['error'] for c in new[3]['checks'])
    assert new[4]['checks'][0]['status']=='passed' and new[4]['checks'][0]['steps']==4
    assert new[4]['checks'][1]['status']=='failed' and 'step=' in new[4]['checks'][1]['error']
    regression=[read(RESULTS/f'norm-activation/{case}.json') for case in ('layernorm','gelu','contracts','training')]
    assert [len(r['checks']) for r in regression]==[3,4,1,6]
    assert all(c['status']=='passed' for r in regression for c in r['checks'])
    migration=read(RESULTS/'torch-migration/layernorm-mixed-boundaries.json')
    assert sum(c['status']=='passed' for c in migration['checks'])==6
    assert sum(c['status']=='blocked' for c in migration['checks'])==5
    for result in new+regression+[migration]: audit(result)
    probe=read(OUT/'probe.json')
    trace=read(OUT/'training-trace.json')
    assert probe['fma_samples']==8202 and probe['fma_finite_bits_equal']
    assert all(c['statistics_bit_equal'] for c in probe['statistics'])
    assert trace['same_input_and_seed_strict_affine_passed']
    assert all(row['max_abs']==0 for row in trace['same_input_and_seed_layernorm'].values())
    assert trace['classification_components']['backward_on_identical_saved_output']['max_abs']==0
    for result in (probe,trace):
        final=result['final_stats']
        assert final['closed'] and not final['context_failed']
        assert all(final[k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_buffer_bytes','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    fixture=ROOT/'npunlock/tests/fixtures/npu4000'
    manifest=json.loads((fixture/'resident/manifest.json').read_text())
    for name,sha in manifest['source_sha256'].items(): assert digest(fixture/name)==sha
    for name,row in manifest['kernels'].items():
        assert digest(fixture/f'resident/{name}.elf')==row['elf_sha256']
        assert digest(fixture/f'resident/{name}.s')==row['assembly_sha256']
    origin=json.loads((ROOT/'provenance/layernorm-moments.json').read_text())
    assert digest(ROOT/origin['license_file'])==origin['license_sha256']
    assert digest(ROOT/'.conda/npunlock/Lib/site-packages'/origin['installed_header'])==origin['installed_header_sha256']
    build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    for name,key in (('backend.cpp','source_sha256'),('_torch_intel_npu.pyd','extension_sha256')):
        assert digest(ROOT/'npunlock-probe/torch-device'/name)==build[key]
    assert digest(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll')==build['session_sha256']
    checks=[c for r in new+regression+[migration] for c in r['checks']]
    counts={s:sum(c['status']==s for c in checks) for s in ('passed','blocked','failed')}
    calls=sum(r['final_stats']['npu_calls'] for r in new+regression+[migration,probe,trace])
    rows=['# 混合LayerNorm统计修复与整网精度边界','',
        '日期：2026-10-18。本地开发版；公开source-release和GitHub未改动。前一轮报告保留为历史记录。','',
        '## 结论','',
        '- 原FP16输入/FP32 affine的4096宽LayerNorm失败已修复；同一输入与梯度重放，输入/weight/bias梯度均与CPU逐位一致。',
        '- 22组独立混合LayerNorm扩大检查通过，包含宽度2、7、15、16、17、33、257、576、4096、65,536，',
        '  两个种子、可选bias，以及17×4096跨块；保存均值与逆标准差逐位一致，梯度沿用原严格阈值。',
        '- 既有LayerNorm/GELU四类14组回归全部通过，含原三组LayerNorm与六组同精度小网络四步训练。',
        '- 更广混合整网检查仍有三组严格失败，**不认证任意混合训练或AMP**；失败见下表，不计作通过。',
        '- 默认GELU none、FP16 Linear/激活、FP32 LayerNorm参数、显式FP32 CrossEntropy的四步SGD独立CPU对照通过。',
        '  这是一项明确配置的有限检查，不把显式FP32损失当作FP16损失的修复或后端自动策略。','',
        '## 原因与修复','',
        f'同一输入/上游梯度指纹：`{before["input_sha256"]}`。原失败为1/4096个weight梯度：',
        f'CPU={before["at_1228"]["expected_weight_gradient"]}，NPU={before["at_1228"]["actual_weight_gradient"]}。',
        'CPU均值=-0.026748105883598328，原NPU=-0.02674810215830803；',
        'CPU逆标准差=1.0068116188049316，原NPU=1.006811499595642。',
        '两种合理统计顺序的微小差异跨过FP16中间累加的舍入边界，导致FP32参数梯度出现一次half ULP跳变。',
        'CPU按分组Welford与级联合并，本项目原本按两遍补偿归约；原NPU统计并非简单的低精度错误。',
        '新增SHAVE内核对混合LayerNorm采用8个FP32 lane、每16步分组及级联合并，保持本机CPU参考的求值顺序。',
        '该排序改编自PyTorch moments_utils.h，保留BSD许可；普通FP32/纯FP16归约路径与既有内核保持不变。','',
        '本机Movi编译器将__builtin_fmaf生成未定义外部fmaf调用，不能直接链接到这个裸SHAVE环境；',
        '这不证明硬件绝无相关指令，只说明当前工具链/运行时没有可直接使用的入口。',
        '因此自行实现整数乘积、对齐sticky位、精确抵消与一次RNE编码，模拟融合结果的数值语义。',
        '没有把分开的FP32乘加冒充融合；也没有声称它是SHAVE硬件FMA加速。',
        '对8,202组随机位模式/抵消/overflow-cancellation/subnormal/特殊值，与系统UCRT fmaf比较，有限输出逐位一致；',
        '不据此宣称全部binary32组合或NaN payload正确舍入已证明。','',
        '## 契约与实现位置','',
        '- 标准nn.LayerNorm/F.layer_norm/native_layer_norm入口不变。混合输入为FP16，weight为FP32，可选同dtype bias。',
        '- normalized_shape须匹配非空尾部维度；混合归一化宽度<=65,536，总存储沿用rank 1–4/1,048,576元素及4MiB界限。',
        '- 大批次按完整行分块，复用原ReferenceSW载体的MemRef、布局和代码窗口指纹检查；统计在NPU生成并由byte DMA提取。',
        '- 单元素归一化沿用既有路径；仅有FP32 bias的混合组合仍明确拒绝，CPU参考自身反向也会类型报错。',
        '- 参数梯度仍按单线程CPU参考经过FP16逐行累加；没有改变参数/动量dtype，没有在CPU计算数据或自动开启AMP。',
        '- 八lane顺序针对本机Torch 2.14.1+cpu/AVX2参考，不保证其他CPU向量宽度、线程配置或后端逐位一致。',
        '- 新增layernorm-stats-f32和仅用于验证的layernorm-fma-test-f32 ELF；C++扩展、session DLL和原有18份resident ELF不变。','',
        '## 整网边界诊断','',
        '诊断网络为Linear(16,16)→混合LayerNorm(16)→GELU→Linear(16,3)，固定seed127，batch4。',
        '第一步四层前向与CPU逐位一致，但分类损失的反向开始产生half梯度差异。',
        f'相同logits的保存log_softmax最大差={trace["classification_components"]["saved_log_softmax"]["max_abs"]}；',
        f'分类损失输入梯度最大差={trace["classification_components"]["full_ce_gradient"]["max_abs"]}。',
        '使用完全相同的保存log_softmax时，CPU/NPU反向逐位一致；相同LayerNorm输入/上游梯度下的输入、weight、bias梯度也逐位一致。',
        '这把该样例的整网差异定位到分类损失保存值的半精度中间舍入，而非已修复LayerNorm的局部导数。',
        '本机CPU半精度log_softmax在sum(exp)与log(sum)处保存half中间量，现有NPU分类路径保留FP32统计。',
        'CPU的65,536类全零half输入甚至产生-inf，而FP32统计路径可保持有限结果；本轮没有为追求逐位相同而改写分类精度策略。','',
        '两种损失dtype与两种GELU的独立CPU轨迹均按原阈值检查，FP32 affine梯度仍使用rtol=3e-5/atol=3e-6；',
        'FP16 Tensor使用rtol=.003/atol=.0005，没有将FP32参数梯度放宽到half容差。','',
        '| 损失dtype | GELU | 状态 | 结果 |','|---|---|---|---|']
    for result in new[3:]:
        for c in result['checks']:
            detail=', '.join(f'{v:.8g}' for v in c['losses']) if c['status']=='passed' else c['error'].replace('\n',' ').replace('|','/')
            rows.append(f'| {"FP32" if result["fp32_loss"] else "FP16"} | {c["name"].split("-")[-1]} | {c["status"]} | {detail} |')
    rows+=['','表中step从0计数，step=2表示第3步。','',
        '显式FP32损失的调用（模型Linear为half、LayerNorm参数显式float）：','',
        '```python','logits = model(inputs)','loss = torch.nn.functional.cross_entropy(logits.float(), labels)','loss.backward()','```','',
        'tanh版本即使使用FP32损失仍有严格轨迹失败，不能只靠改损失dtype保证全部混合网络一致。',
        '当前只将默认GELU none的这组四步配置记为通过；不宣称长期收敛、模型质量或其他网络已认证。','',
        '## 验收与资源','',
        f'保留的修复后验收与附加ELF/逐层诊断合计**{calls:,}次NPU调用**；主检查{counts["passed"]}组通过、{counts["blocked"]}组明确未支持、{counts["failed"]}组严格失败。',
        '另有8,202组融合运算位模式、3组独立统计位模式与一次相同输入/梯度诊断通过；修复前诊断不计入上述调用数。',
        '新检查逐组live buffer增量为零；所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。',
        '12项离线回归通过（INT64/分类载体、源码导出），没有新增Python依赖、模型、图片或持久图缓存。',
        '本轮不重复既有49组基础设备/训练回归或全部AdamW状态测试；相关实现未改，前轮结果仍为历史记录。',
        '未评估新整数Welford路径的性能收益，不能据此声称算力或吞吐提升。','',
        '## 复现','', '```powershell',
        '.\\.conda\\npunlock\\python.exe -B npunlock-probe/research/build_layernorm_kernel.py',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\layernorm_mixed_probe.py',
        "foreach ($case in @('replay','stress','contracts','training')) {",
        '    .\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\layernorm_mixed_acceptance.py -ScriptArguments @(\'--case\',$case)',
        '}',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\layernorm_mixed_acceptance.py -ScriptArguments @(\'--case\',\'training\',\'--fp32-loss\')',
        '.\\run_torch_npu.ps1 -Script .\\npunlock-probe\\research\\layernorm_mixed_training_trace.py',
        '```','',
        '两个混合整网命令含严格失败而返回非零，其他检查不因此被记作通过。审计还需重跑norm_activation四类及migration boundaries，',
        '然后执行本报告脚本。修复前diagnosis.json需旧后端或从本报告读取其历史数据；不能用新代码伪造修复前基线。',
        '原始JSON/日志/编译响应在报告审计后清理，只保留源码、内核与报告。','',
        '## 参考与许可','',
        '[PyTorch moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h)的排序已直接改编；',
        '本地头文件与许可指纹见[来源记录](../../../../provenance/layernorm-moments.json)，BSD条款见[PyTorch许可](../../../../LICENSES/PyTorch.txt)。',
        '[LogSoftmaxKernelImpl.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)用于核对CPU中间dtype，没有移植其SIMD实现。','',
        '## 指纹（SHA-256）','', '```text']
    names=list(new[0]['source_sha256'])+['npunlock-probe/research/layernorm_mixed_moments.py',
        'npunlock-probe/research/layernorm_mixed_probe.py','npunlock-probe/research/layernorm_mixed_training_trace.py',
        'npunlock-probe/research/build_layernorm_kernel.py','npunlock-probe/research/build_resident_kernels.py',
        'npunlock-probe/research/layernorm_mixed_report.py','provenance/layernorm-moments.json','LICENSES/PyTorch.txt',
        'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd',
        'npunlock-probe/session-runtime/npunlock_session.dll']
    names += [f'npunlock/tests/fixtures/npu4000/resident/{n}.elf' for n in ('layernorm-stats-f32','layernorm-fma-test-f32')]
    rows += [name+' '+digest(ROOT/name) for name in dict.fromkeys(names)]
    rows += ['```','']
    (OUT/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    print(f'Audited {counts}; {calls:,} NPU calls')


if __name__=='__main__': main()
