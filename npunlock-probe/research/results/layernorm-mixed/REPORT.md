# 混合LayerNorm统计修复与整网精度边界

日期：2026-10-18。本地开发版；公开source-release和GitHub未改动。前一轮报告保留为历史记录。

## 结论

- 原FP16输入/FP32 affine的4096宽LayerNorm失败已修复；同一输入与梯度重放，输入/weight/bias梯度均与CPU逐位一致。
- 22组独立混合LayerNorm扩大检查通过，包含宽度2、7、15、16、17、33、257、576、4096、65,536，
  两个种子、可选bias，以及17×4096跨块；保存均值与逆标准差逐位一致，梯度沿用原严格阈值。
- 既有LayerNorm/GELU四类14组回归全部通过，含原三组LayerNorm与六组同精度小网络四步训练。
- 更广混合整网检查仍有三组严格失败，**不认证任意混合训练或AMP**；失败见下表，不计作通过。
- 默认GELU none、FP16 Linear/激活、FP32 LayerNorm参数、显式FP32 CrossEntropy的四步SGD独立CPU对照通过。
  这是一项明确配置的有限检查，不把显式FP32损失当作FP16损失的修复或后端自动策略。

## 原因与修复

同一输入/上游梯度指纹：`02b355bf987bffbfc31709448d1e644963e6365dfcfe42a5b8f664fed58486ac`。原失败为1/4096个weight梯度：
CPU=-0.1368408203125，NPU=-0.13671875。
CPU均值=-0.026748105883598328，原NPU=-0.02674810215830803；
CPU逆标准差=1.0068116188049316，原NPU=1.006811499595642。
两种合理统计顺序的微小差异跨过FP16中间累加的舍入边界，导致FP32参数梯度出现一次half ULP跳变。
CPU按分组Welford与级联合并，本项目原本按两遍补偿归约；原NPU统计并非简单的低精度错误。
新增SHAVE内核对混合LayerNorm采用8个FP32 lane、每16步分组及级联合并，保持本机CPU参考的求值顺序。
该排序改编自PyTorch moments_utils.h，保留BSD许可；普通FP32/纯FP16归约路径与既有内核保持不变。

本机Movi编译器将__builtin_fmaf生成未定义外部fmaf调用，不能直接链接到这个裸SHAVE环境；
这不证明硬件绝无相关指令，只说明当前工具链/运行时没有可直接使用的入口。
因此自行实现整数乘积、对齐sticky位、精确抵消与一次RNE编码，模拟融合结果的数值语义。
没有把分开的FP32乘加冒充融合；也没有声称它是SHAVE硬件FMA加速。
对8,202组随机位模式/抵消/overflow-cancellation/subnormal/特殊值，与系统UCRT fmaf比较，有限输出逐位一致；
不据此宣称全部binary32组合或NaN payload正确舍入已证明。

## 契约与实现位置

- 标准nn.LayerNorm/F.layer_norm/native_layer_norm入口不变。混合输入为FP16，weight为FP32，可选同dtype bias。
- normalized_shape须匹配非空尾部维度；混合归一化宽度<=65,536，总存储沿用rank 1–4/1,048,576元素及4MiB界限。
- 大批次按完整行分块，复用原ReferenceSW载体的MemRef、布局和代码窗口指纹检查；统计在NPU生成并由byte DMA提取。
- 单元素归一化沿用既有路径；仅有FP32 bias的混合组合仍明确拒绝，CPU参考自身反向也会类型报错。
- 参数梯度仍按单线程CPU参考经过FP16逐行累加；没有改变参数/动量dtype，没有在CPU计算数据或自动开启AMP。
- 八lane顺序针对本机Torch 2.14.1+cpu/AVX2参考，不保证其他CPU向量宽度、线程配置或后端逐位一致。
- 新增layernorm-stats-f32和仅用于验证的layernorm-fma-test-f32 ELF；C++扩展、session DLL和原有18份resident ELF不变。

## 整网边界诊断

诊断网络为Linear(16,16)→混合LayerNorm(16)→GELU→Linear(16,3)，固定seed127，batch4。
第一步四层前向与CPU逐位一致，但分类损失的反向开始产生half梯度差异。
相同logits的保存log_softmax最大差=0.0009765625；
分类损失输入梯度最大差=0.0001220703125。
使用完全相同的保存log_softmax时，CPU/NPU反向逐位一致；相同LayerNorm输入/上游梯度下的输入、weight、bias梯度也逐位一致。
这把该样例的整网差异定位到分类损失保存值的半精度中间舍入，而非已修复LayerNorm的局部导数。
本机CPU半精度log_softmax在sum(exp)与log(sum)处保存half中间量，现有NPU分类路径保留FP32统计。
CPU的65,536类全零half输入甚至产生-inf，而FP32统计路径可保持有限结果；本轮没有为追求逐位相同而改写分类精度策略。

两种损失dtype与两种GELU的独立CPU轨迹均按原阈值检查，FP32 affine梯度仍使用rtol=3e-5/atol=3e-6；
FP16 Tensor使用rtol=.003/atol=.0005，没有将FP32参数梯度放宽到half容差。

| 损失dtype | GELU | 状态 | 结果 |
|---|---|---|---|
| FP16 | none | failed | step=0, parameter=1.weight, fp32_loss=False: Tensor-likes are not close!  Mismatched elements: 10 / 16 (62.5%) Greatest absolute difference: 6.103515625e-05 at index (3,) (up to 3e-06 allowed) Greatest relative difference: 0.0022371364757418633 at index (4,) (up to 3e-05 allowed) |
| FP16 | tanh | failed | step=0, parameter=1.weight, fp32_loss=False: Tensor-likes are not close!  Mismatched elements: 5 / 16 (31.2%) Greatest absolute difference: 6.29425048828125e-05 at index (5,) (up to 3e-06 allowed) Greatest relative difference: 0.03206996992230415 at index (5,) (up to 3e-05 allowed) |
| FP32 | none | passed | 1.1035312, 1.086705, 1.0571959, 1.0187119 |
| FP32 | tanh | failed | step=2, parameter=1.weight, fp32_loss=True: Tensor-likes are not close!  Mismatched elements: 1 / 16 (6.2%) Greatest absolute difference: 3.0517578125e-05 at index (11,) (up to 3e-06 allowed) Greatest relative difference: 0.0009671180159784853 at index (11,) (up to 3e-05 allowed) |

表中step从0计数，step=2表示第3步。

显式FP32损失的调用（模型Linear为half、LayerNorm参数显式float）：

```python
logits = model(inputs)
loss = torch.nn.functional.cross_entropy(logits.float(), labels)
loss.backward()
```

tanh版本即使使用FP32损失仍有严格轨迹失败，不能只靠改损失dtype保证全部混合网络一致。
当前只将默认GELU none的这组四步配置记为通过；不宣称长期收敛、模型质量或其他网络已认证。

## 验收与资源

保留的修复后验收与附加ELF/逐层诊断合计**17,911次NPU调用**；主检查45组通过、5组明确未支持、3组严格失败。
另有8,202组融合运算位模式、3组独立统计位模式与一次相同输入/梯度诊断通过；修复前诊断不计入上述调用数。
新检查逐组live buffer增量为零；所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。
12项离线回归通过（INT64/分类载体、源码导出），没有新增Python依赖、模型、图片或持久图缓存。
本轮不重复既有49组基础设备/训练回归或全部AdamW状态测试；相关实现未改，前轮结果仍为历史记录。
未评估新整数Welford路径的性能收益，不能据此声称算力或吞吐提升。

## 复现

```powershell
.\.conda\npunlock\python.exe -B npunlock-probe/research/build_layernorm_kernel.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_probe.py
foreach ($case in @('replay','stress','contracts','training')) {
    .\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_acceptance.py -ScriptArguments @('--case',$case)
}
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_acceptance.py -ScriptArguments @('--case','training','--fp32-loss')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\layernorm_mixed_training_trace.py
```

两个混合整网命令含严格失败而返回非零，其他检查不因此被记作通过。审计还需重跑norm_activation四类及migration boundaries，
然后执行本报告脚本。修复前diagnosis.json需旧后端或从本报告读取其历史数据；不能用新代码伪造修复前基线。
原始JSON/日志/编译响应在报告审计后清理，只保留源码、内核与报告。

## 参考与许可

[PyTorch moments_utils.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/moments_utils.h)的排序已直接改编；
本地头文件与许可指纹见[来源记录](../../../../provenance/layernorm-moments.json)，BSD条款见[PyTorch许可](../../../../LICENSES/PyTorch.txt)。
[LogSoftmaxKernelImpl.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)用于核对CPU中间dtype，没有移植其SIMD实现。

## 指纹（SHA-256）

```text
torch_intel_npu.py 29d81705ddc2d020008088133d6a333d7a52688ff9be0ee8b554fe6a3a8a3922
torch_npu_autograd.py ab1513ac423b0468dc01399d44855a83efa24227e8c368bc2f55db7d0e31235e
npunlock-probe/research/layernorm_mixed_acceptance.py e5318060a13b87351f34d58516849594be527837c13dee85c95416de3377a3b9
npunlock-probe/research/layernorm_mixed_diagnose.py 316bc3cb389f844ccc5b581053d6d86dd7f0307a3054c4afe7c1303bcf42f9fc
npunlock/tests/fixtures/npu4000/resident_layernorm.c d581488a7c057a6c7f84339423757f36d07511ce805a22ec51ed150c74077b01
npunlock/tests/fixtures/npu4000/resident/manifest.json f381add6d42cd58731f10d5832f20bd0737d06fa42b77f1655e9ee121837aee1
npunlock-probe/research/layernorm_mixed_moments.py ed8842cda11b1bd7f6f6c728ad0cfd4bdbde09aa70bbe8fb5d0a1874b96ce0fe
npunlock-probe/research/layernorm_mixed_probe.py 50fba5e5ebf23b8e0e439429e4ec68621b6ab7862f5688425f08150bf0ae0b8a
npunlock-probe/research/layernorm_mixed_training_trace.py be77711fa5e5d87ec5338b492cb76d8f715c81eaca86a6e65c91c1eb6b391d50
npunlock-probe/research/build_layernorm_kernel.py 3574bb585c4ceb091f7640230877ea99dfcede47af39f6e33835b3a5747051d9
npunlock-probe/research/build_resident_kernels.py 6dcab8629107a1fe9e19ab322321b61976c15b826d0c795187b67786e94c95c7
npunlock-probe/research/layernorm_mixed_report.py 81cbbcff41230b3b880cb0f04cb4a20afecb33acd4d81af725244102d9adbc6d
provenance/layernorm-moments.json 0c099e88880f129b6aab4b910b7611ec0c84b92cea0cddf46ea9ac11b47afc71
LICENSES/PyTorch.txt 0fc92425d130d9c87a5eb7427082e3afdbff6ef169bc678e15abba4c2d37c934
npunlock-probe/torch-device/backend.cpp 006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock-probe/torch-device/_torch_intel_npu.pyd b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/session-runtime/npunlock_session.dll 0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock/tests/fixtures/npu4000/resident/layernorm-stats-f32.elf f882a5423014374cb56595b22cc7f090931dd4a07d215b002d09492a899d9f3f
npunlock/tests/fixtures/npu4000/resident/layernorm-fma-test-f32.elf 410a5fb09355552274b8d93f1f380ffd09cf2a3e8e1006719e6f50ffb6e761a6
```

## 验收后清理

删除32份原始JSON与编译响应，共249,157字节（243.32KiB）。
删除时核对81份保留源码、内核、宿主组件、许可/来源、文档和报告的SHA-256完全一致，随后补充本段清理统计。
保留两个新增ELF及汇编、C源码、manifest、构建/诊断/验收/报告脚本；原有18份resident ELF与C++/DLL/PYD保持不变。
未安装依赖、下载模型、生成图片或持久图缓存；公开source-release的Git工作区保持干净，未推送GitHub。
