# 分类损失FP64对照与32步混合训练边界

日期：2026-10-18。本地开发版；本轮新增验证与报告，生产算子/内核保持上一轮版本，未推送GitHub。

## 结论

- 10组分类精度/容量检查通过，包括原none/tanh两组网络logits、随机形状、偏移/范围及宽类别边界。
- 原两例显式FP32损失得到的FP16输入梯度，均与FP64参考最终舍入到FP16的结果相同。
- 本机CPU原生FP16保存统计有额外舍入；NPU保持FP32中间统计。原生half两侧结果不同，不能据此直接认定NPU算错。
- 32步none/tanh网络都保持有限数值并拟合固定四样本；严格独立CPU轨迹分别在第6/第5步失败。
  **两组32步检查标记strict_failed，未计为严格通过。** 四步通过的上一轮结论不推广为长期逐项一致。

## 对照口径

硬件/软件沿用Windows x64、Core Ultra 7 258V、NPU 8086:643e、driver_version=1005540、Torch 2.14.1+cpu。
CPU单线程、关闭oneDNN。所有路径使用完全相同的已量化FP16 logits，再分别运行CPU/NPU的原生half或显式FP32损失。
独立参考是这些logits转成FP64后的稳定log_softmax/CrossEntropy及其导数，没有拿CPU half作数学真值。
CPU FP64只用于离线核对，不进入NPU运行时或训练数据计算。
FP32前向保留rtol=3e-5/atol=3e-6；FP16前向与FP16输入梯度沿用rtol=.003/atol=.0005。
显式FP32损失依然向FP16 logits返回FP16梯度，因此对FP64真值有最终half舍入误差，不等于整条反向全用FP32。

## 原两例的FP64误差

| GELU来源 | 路径 | log_softmax最大绝对差 | 输入梯度RMS差 | 与FP64舍入后half梯度不同元素 |
|---|---|---:|---:|---:|
| none | cpu-half-loss | 0.000530841336 | 3.24374837e-05 | 4 / 12 |
| none | cpu-fp32-loss | 5.90283019e-08 | 2.28638554e-05 | 0 / 12 |
| none | intel_npu-half-loss | 0.000445721164 | 2.97352593e-05 | 3 / 12 |
| none | intel_npu-fp32-loss | 1.21578818e-07 | 2.28638554e-05 | 0 / 12 |
| tanh | cpu-half-loss | 0.000513034968 | 3.18534695e-05 | 5 / 12 |
| tanh | cpu-fp32-loss | 8.34295806e-08 | 2.47568747e-05 | 0 / 12 |
| tanh | intel_npu-half-loss | 0.000513034968 | 3.22112833e-05 | 4 / 12 |
| tanh | intel_npu-fp32-loss | 8.34295806e-08 | 2.47568747e-05 | 0 / 12 |

原生half梯度的CPU/NPU优劣随样例改变：none的NPU误差较小，tanh的CPU RMS误差略小，不能宣称NPU始终更准确。
显式FP32损失在这两个样例中消除了除最终half舍入外的输入梯度差异。
这支持保留现有FP32统计策略；本轮没有为模仿CPU half溢出而改写生产算子，也未将原FP16损失整网失败改记为通过。

## 大类别边界

| 场景 | CPU half | NPU half | NPU FP32 | FP64参考 |
|---|---:|---:|---:|---:|
| 65,532类全零logits的CrossEntropy | +inf | 11.09375 | 11.09029388 | 11.09029385 |
| 65,536类全零logits的log_softmax每项 | -inf | -11.09375 | -11.09035492 | -11.09035489 |

65,532类CE使用现有NLL packet的额外4列，恰好达到65,536元素任务上限；65,536类只检查log_softmax。
本机CPU last-dimension实现把sum(exp)和log(sum)保存为scalar_t，FP16可能在求和时溢出。
NPU内部保留FP32统计，最终half值只是把有限结果舍入到FP16；此例不需要降低NPU精度来兼容CPU异常。

## 32步训练

网络仍为Linear(16,16)→混合LayerNorm(16)→GELU→Linear(16,3)，SGD lr=.01/momentum=.8；
FP16 Linear/激活、FP32 LayerNorm参数、两侧都显式使用FP32 CrossEntropy。seed127与原先初始化/数据顺序相同，
CPU/NPU各自连续更新，不重置CPU权重，不放宽原参数梯度/权重阈值。严格差异被记录后继续训练，用于观察有限性和学习行为。

| GELU | 初始NPU loss | 32次更新后CPU / NPU loss | 固定训练集准确率 | 首次严格失败（1起算） |
|---|---:|---:|---:|---|
| none | 1.10353124 | 0.180186361 / 0.180121198 | 100% | 第6步，gradient:1.weight，1个元素，最大差7.62939453125e-06 |
| tanh | 0.940976202 | 0.0871895924 / 0.0871895924 | 100% | 第5步，gradient:1.weight，1个元素，最大差1.52587890625e-05 |

32步说明该固定四样本例子能正常学习，不能证明泛化、真实数据集训练或任意长时间稳定性。
后续的严格轨迹偏差尚未逐层定位；不能仅凭最终loss接近就宣称已修复，也不能把该偏差直接等同于发散。
下一项有针对性的工作是重放第5/6步，在相同输入与上游梯度下核对局部导数及优化器状态，避免盲目改精度。

## 验收和资源

最终保留验收共**18,453次NPU调用**：精度探针549次，32步训练17,904次。
10组精度检查通过；2组32步严格轨迹未通过，其有限学习检查通过但不替代严格验收。
所有检查live buffer增量为零；两个进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。
未改变torch_intel_npu.py、autograd、20份resident ELF/汇编、manifest和C++/DLL/PYD；无需安装依赖或重编内核。
没有新增模型、图表或持久图缓存；未进行性能计时，也未重复与本轮无关的设备/卷积/低位回归。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\classification_precision_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_stability_probe.py
.\.conda\npunlock\python.exe -B npunlock-probe/research/classification_precision_report.py
```

第二个命令按strict_failed返回非零，同时保留完整32步结果；不要忽略该返回码。
报告脚本要求重跑后的两个原始JSON并核对源码/内核/资源指纹；验收后按空间约定删除JSON，历史报告保留。

## 参考

[PyTorch LogSoftmaxKernelImpl.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h)
用于核对CPU half保存中间统计的类型；本机实际安装头文件指纹列于下方。main链接只作导航，没有移植CPU SIMD代码。
原统计/更新修复见[混合LayerNorm报告](../layernorm-mixed/REPORT.md)与[更新规则报告](../mixed-training-rounding/REPORT.md)。

## SHA-256

```text
.conda/npunlock/Lib/site-packages/torch/include/ATen/native/cpu/LogSoftmaxKernelImpl.h 26165581a12ae3cdc9a501c57a0916b5239cb1901075c0a60225b5abcd4e6d57
npunlock-probe/research/classification_precision_probe.py 1fed9a081c3a98e9cdce105b6ecb1e3f03d56b41fada4e8155615c9878274ddf
npunlock-probe/research/classification_precision_report.py 2802ceee5a6eae00f46917a80b0660877853cf50fa5d30d78a5f370e9f792c55
npunlock-probe/research/mixed_training_stability_probe.py 5ff4467fa3fdd3d3d27446f8dc58d646b04b5e1210c17d3ec8166d0cb8bfab9e
npunlock-probe/session-runtime/npunlock_session.dll 0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock-probe/torch-device/_torch_intel_npu.pyd b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/torch-device/backend.cpp 006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock/tests/fixtures/npu4000/resident/manifest.json f381add6d42cd58731f10d5832f20bd0737d06fa42b77f1655e9ee121837aee1
npunlock/tests/fixtures/npu4000/resident_classification.c fcfb57bae8381b1875e11dec36505fdefd6fda3a858a5fbc7aca1c45ddfd607c
torch_intel_npu.py 0523715f8c418584baf28e75f239b8c3575a814a4f28cd32b03ba2f29067f55f
torch_npu_autograd.py ab1513ac423b0468dc01399d44855a83efa24227e8c368bc2f55db7d0e31235e
```

## 验收后清理

删除两份原始JSON，共108594 字节。清理前后79份保留文件的SHA-256一致，随后追加本段和空间记录。
保留三份研究/审计脚本、报告和原生产内核；没有新增模型、图片、持久缓存或Python依赖，公开source-release保持干净。

