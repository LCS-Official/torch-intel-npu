# 第5/6步混合训练差异：SGD舍入归因与独立验证集

日期：2026-10-18。本地开发版；本轮扩展研究/验收脚本并更新文档，生产后端与内核保持不变，未推送GitHub。

## 结论

- 原none第6步、tanh第5步的严格梯度差异已完成归因：SGD参数更新的融合/分步乘加及half尾部中间舍入足以解释这两个失败点。
- 两种CPU参照共96次逐层重放，相同输入/权重/上游梯度下前向与局部导数均逐位相同。
- 144次相同状态SGD重放中，动量逐位一致；NPU参数更新均与分步FP32乘加一致，CPU均与本机向量/标量尾部规则一致。
- 只将CPU参照改为NPU的分步更新顺序，两个网络各六步的前向、梯度、参数全部相同；NPU仍运行原torch.optim.SGD。
- 标准CPU参照的严格失败保留，未放宽阈值；对照实验不能当作标准CPU轨迹通过。
- 新增64训练/64独立验证样本的32步合成分类实验：两侧验证准确率均为75%，预测完全相同；该实验也有严格轨迹失败，不能替代严格验收。

## 原失败重放

沿用seed127、batch4、Linear(16,16)→LayerNorm(16)→GELU→Linear(16,3)，FP16 Linear/激活、FP32 LayerNorm参数，
两侧显式FP32 CrossEntropy；SGD lr=.01/momentum=.8/foreach=False/fused=False，CPU单线程并关闭oneDNN。
分别连续运行六步；CPU/NPU从相同初始化独立更新，不将CPU参数重置为NPU状态。
同输入局部重放只用于诊断：把NPU层输入、参数、上游梯度复制给一个CPU shadow，核对该层实际输出与导数。

| GELU | 首次严格失败（1起算） | LayerNorm weight梯度最大差 | 样本CPU / NPU |
|---|---|---:|---|
| none | 第6步 | 7.62939453125e-06 | index 0：0.0146636962890625 / 0.0146560668945312 |
| tanh | 第5步 | 1.52587890625e-05 | index 9：0.0211334228515625 / 0.0211181640625 |

原FP32参数梯度rtol=3e-5/atol=3e-6，FP16 Tensor rtol=.003/atol=.0005不变。
同输入/上游梯度下LayerNorm/GELU/Linear的局部前后向均相同；该样例不支持把失败归因于LayerNorm局部导数或GELU实现。

## 最早分叉与算术顺序

两种GELU都从第2步的FP32 LayerNorm bias参数更新出现微小差异；相同状态的动量仍逐位相同。
标准CPU向量add_使用FMA，对alpha*momentum+parameter只进行一次最终FP32舍入；当前NPU先执行FP32 mul，再执行FP32 add。
这是FP32运算顺序不同，没有将FP32参数降为FP16。后续FP16参数的小张量/标量尾部还会出现CPU额外的half乘积舍入。
CPU half向量主体先转FP32再FMA；本机AVX2每次处理32个half，尾部走标量规则。
诊断用UCRT fmaf核对融合结果，用独立的CPU乘法/加法核对分步结果；没有把普通乘加冒充FMA。

第2步的同状态bias样本：

| GELU | 参数/index | CPU融合结果 | NPU分步结果 | 绝对差 |
|---|---|---:|---:|---:|
| none | 1.bias/8 | 0.0008726196247152984 | 0.0008726195665076375 | 5.82076609135e-11 |
| tanh | 1.bias/13 | 0.0001993255718844011 | 0.0001993255573324859 | 1.45519152284e-11 |

原标准参照共72次同状态参数更新，16次有数值差异，但每次局部更新均在既有单算子容差内；
对应24次FP32更新全部匹配CPU FMA，48次FP16更新全部匹配向量FMA主体加half标量尾部；NPU全部匹配分步FP32更新。
小差异经过后续half舍入和反向传播，可以越过原FP32参数梯度阈值；不能从单步参数接近推断任意长度轨迹相同。

## 改变CPU更新规则的对照

保留相同网络、数据、NPU实现及NPU标准SGD，只将独立CPU参照的参数更新改为：

```text
alpha = round_to_parameter_dtype(-lr)
parameter = round_to_parameter_dtype(FP32(parameter) + FP32(momentum) * FP32(alpha))
```

CPU动量仍按既有乘法与加法更新，未读取NPU值来覆盖CPU轨迹；同状态shadow副本只参与核对。
在两个网络共12步中，CPU/NPU四层前向、上游梯度、全部参数梯度及更新后参数均逐位相同。
这项受控变化足以消除原第5/6步分叉，提供更新舍入是其来源的证据；它是诊断对照，不是新NPU模式或标准CPU兼容性修复。
该结论限定于这两个六步重放，不把全部模型/其他输入的差异都归为舍入。

## 独立验证集

另用seed131的默认GELU混合网络，固定seed601生成一个线性teacher、64训练样本及64未参与训练的样本。
FP16输入16维/3分类，batch8，四个epoch共32次更新，SGD lr=.03/momentum=.8；CPU/NPU独立训练，使用原标准优化器。
teacher与数据在CPU准备，属于实验输入生成；模型前后向与更新在NPU执行，CPU参数更新对照未用于这个实验。

| 指标 | CPU | NPU |
|---|---:|---:|
| 初始验证loss | 1.14187634 | 1.14187634 |
| 最终验证loss | 0.6174545288 | 0.6174795628 |
| 初始验证准确率 | 39.0625% | 39.0625% |
| 最终验证准确率 | 75.00% | 75.00% |

最终64个预测全部相同；验证loss绝对差2.50339508057e-05，logits最大差0.0029296875。
该实验原严格轨迹首次在第4步gradient:1.weight超差，状态仍为strict_failed。
没有用新的宽阈值把它改记为严格通过；该新输入上的逐项偏差未做完整局部归因。
验证集提升说明这个合成任务未观察到预测质量下降，不认证真实数据集、任意模型、长期训练或AMP。

输入/teacher及初始化指纹：

```text
train ca5e24d083c32942b6cbb0a0804227eed96099d6b1e706ec20cd6308ffec2d05
train_labels d58feb2e534e50729c4525f31e08388ef6fb37d60340b3182a3b4fb29e68b7f2
heldout 9583d453e27ffbab20504314f08c5ebd64cd91c2ce5b4805d0ae2c339f5373b6
heldout_labels 0da963ddc3ae1df3fc1e1cc345e4783f1655bd77401dd937b68e1cde0e40f6ed
teacher cfd7737c220c99764b336bfcb094283491afcaffa67da0483213e717bfbdd4a3
initial_parameters e44bcfb9a66ef87b9c55f26fcde92dc56e491d73d5eed2d32c3007d98c2f9697
```

## 验收与资源

| 入口 | 结果 | NPU调用 |
|---|---|---:|
| 标准CPU六步归因 | 重现原两组严格失败；48层局部/72状态重放完成 | 3304 |
| 分步CPU对照六步 | 两个网络参数/梯度相同；不计作标准CPU通过 | 3304 |
| 原默认四步入口回归 | 原两组通过 | 2168 |
| 64/64独立验证集 | 学习/验证提升；严格轨迹失败保留 | 10298 |

最终验收共**19,074次NPU调用**。所有进程shutdown后live/cache/scalar/quarantine/session归零，无context failure。
生产Python后端/autograd、20份resident ELF及汇编/manifest、C++/DLL/PYD均未改；未新增依赖、模型、图表或持久图缓存。
本轮未重复设备/卷积/低位模型全回归，也未进行性能计时；没有为匹配CPU逐位结果而引入额外软件FMA生产路径。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py -ScriptArguments @('--steps','6')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py -ScriptArguments @('--steps','6','--unfused-reference')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_training_rounding_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\mixed_validation_probe.py
.\.conda\npunlock\python.exe -B npunlock-probe/research/mixed_step_report.py
```

六步诊断返回零表示重现了预期差异并完成核对，原标准轨迹仍有严格失败；独立验证集命令按strict_failed返回非零。
审计需四份新生成JSON；验收后按空间约定清理，保留本报告与脚本，旧报告作为历史记录。

## 参考

[PyTorch add ufunc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ufunc/add.h)区分标量表达式与向量fmadd；
本机vec256_float.h/vec256_half.h、Loops.h、Half.h及sgd.py用于核对实际CPU控制流，指纹列于下方。
[PyTorch数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)指出求值顺序和平台变化可使数学等价计算产生不同浮点结果。
本轮不是仅凭该说明排除问题，而是以上述相同状态重放与改变CPU舍入顺序的实验完成归因；未移植CPU SIMD源码。

## SHA-256

```text
.conda/npunlock/Lib/site-packages/torch/include/ATen/cpu/vec/vec256/vec256_float.h 959df6fc22c28f021524043bc644b17b251279316ce9f78ed288cd7e6d789046
.conda/npunlock/Lib/site-packages/torch/include/ATen/cpu/vec/vec256/vec256_half.h 6092beff1546e33ec878e0eea589356ca305e5ec58aa016814679379d60a49d5
.conda/npunlock/Lib/site-packages/torch/include/ATen/native/cpu/Loops.h 6c83d56976eb9b7d8e92627cc24d7226f5a2a493efd3c17a76471351feb2ec7e
.conda/npunlock/Lib/site-packages/torch/include/torch/headeronly/util/Half.h b820c7f8267d3647edfd8666abbcd1cbe7e74c5ac37a3b93c9b120440b5cfe06
.conda/npunlock/Lib/site-packages/torch/optim/sgd.py 34476f526f3c48d40ebfa86eef556e5d6d740239edc0bb061ea4786d6ccda275
npunlock-probe/research/mixed_step_report.py fbeeef902ad9bc809514db023bd4b38b81155276ad8d18dbfd5f78ca7048cc90
npunlock-probe/research/mixed_training_rounding_probe.py 2f97500946927c5934402c5687bb0b40e97bf2a4421bf4d4cf58710871d11a61
npunlock-probe/research/mixed_validation_probe.py 744c88a9a614c21ace7a32401868b2fc303e89e89d2db1b537f238ea3809a886
npunlock-probe/session-runtime/npunlock_session.dll 0121f3efb7c50d3f38a3b5c6c529af5bab2f6980d44b2a7e9896d7a4b07cd5c9
npunlock-probe/torch-device/_torch_intel_npu.pyd b072e438e153adb1026604e12125e94c749dd57e4a3a59d06eb647debf330a6e
npunlock-probe/torch-device/backend.cpp 006cad067bf8c16782be59b58b9b5d1dfd34244c99329a38854273dd39e6a924
npunlock/tests/fixtures/npu4000/resident/manifest.json f381add6d42cd58731f10d5832f20bd0737d06fa42b77f1655e9ee121837aee1
npunlock/tests/fixtures/npu4000/resident_gelu.c 8c97064a8e9a772ef4729b73a3d1d9e0aa0355a4a60e631368d0c13dd43e8938
torch_intel_npu.py 0523715f8c418584baf28e75f239b8c3575a814a4f28cd32b03ba2f29067f55f
torch_npu_autograd.py ab1513ac423b0468dc01399d44855a83efa24227e8c368bc2f55db7d0e31235e
```

## 验收后清理

删除4份原始JSON，共600843 字节。清理前后83份保留文件的SHA-256一致，随后追加本段与空间记录。
保留扩展诊断、新验证/审计脚本和报告；生产后端、原20份resident ELF/汇编/manifest及原生组件未改。无新增模型、图表、依赖或持久图缓存。

