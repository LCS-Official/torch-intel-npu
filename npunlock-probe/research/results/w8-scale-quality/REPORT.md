# S51：W8 scale／低秩联合校准与输入敏感精度边界

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；使用已有本地SmolLM2-135M-Instruct。

完成默认关闭的受限scale／rank32联合校准。90个FFN投影的校准集局部重建平方误差合计下降0.888910%，但真实NPU内部整网通过数243→242/290，整个FFN候选还在最后一层被既有局部精度门槛拒绝。**本轮没有整网质量改善结论，不替换生产默认或原W8+rank32基线。**

共享实际输入的进一步诊断表明：该最坏坐标的-0.375差值通过attention边界传入，三个末层参数配置都重现同一峰值；不是新scale或SiLU单独造成的差异。此结论只针对本次输入和坐标，没有把全部W8质量问题归因于attention或认定驱动错误。

## 方法和预先固定的范围

- 重建S44/S50的40条CPU FP16 greedy轨迹到64个消费token；编号i%5!=4的32条、2,048行激活用于拟合，另外8条、290个预测位置用于内部NPU选择。拟合不读取独立回归、WikiText、PPL或生成结果。
- 原W8代码仍来自既有GPTQ适配；INT8 codes完全不变，rank固定32。只允许FP16逐行scale在原值±2%内移动，并交替重新拟合已有低秩补偿，两轮固定。
- 固定两个候选：gate-up调整60个gate/up投影；ffn调整全部90个FFN投影。attention投影、图结构、压缩格式和推理入口保持原实现。
- 复用已有PackedWeight、LowRankCorrection和fit_lowrank_correction；新scale步骤自行实现一维最小二乘，不复制新的第三方代码，不新增库或OpenVINO运行时依赖。

对输入X和固定INT8代码C，令U=XCᵀ（CPU FP64），T为原权重FP32投影再舍入Half的输出，D为现有两次FP32累加／Half舍入的低秩补偿。每行解：

```text
s* = [sum(U * (T-D)) + 0.01 * sum(U²) * s_old] / [1.01 * sum(U²)]
```

先将s*限制在初始scale的±2%，再检查最近FP16值及其相邻两值；以实际分阶段Half主投影、补偿和相加的校准SSE逐行保留较优scale。随后重新拟合rank32因子，只有该投影总SSE下降才接受因子更新。零能量保持原scale；输入、范围、格式、有限值和FP16溢出均有检查。

保证的是每次更新后**每个投影的校准集总SSE不增加**，不是每行误差都下降，也不是未知输入或整网质量保证。scale变化后部分重新拟合的因子被拒绝，原因子可以保留。

## CPU局部拟合

| 投影 | 原SSE | 新SSE | 下降 | 改变scale的行数 |
|---|---:|---:|---:|---:|
| gate | 559.257849 | 559.030076 | 0.040728% | 1565/46080 |
| up | 663.217514 | 662.648788 | 0.085753% | 3209/46080 |
| down | 4162.436950 | 4115.366420 | 1.130841% | 585/17280 |
| 合计 | 5384.912313 | 5337.045285 | 0.888910% | 5359/109440 |

88/90个投影SSE下降，另2个保留原值。scale实际范围为原值的0.9978237214–1.0023966447倍。gate-up的60个scale矩阵、56组因子改变；ffn的88个scale矩阵、69组因子改变。表中的绝对SSE包含不同层的输出能量，不能把下降量直接当作最终logits贡献比例。

基线W8准备约100.74秒，原rank32因子约19.71秒；两次新拟合约35.99／42.06秒，更新后的参数NPZ逐字节相同。以上均为本机CPU离线准备时间，不是NPU推理速度。

## 图与默认行为控制

对两个候选、30层、B1/S64整层前向及B1/S1 decode，共120组图检查：XML与基线逐字节相同，常量区长度相同；所有210个主权重矩阵的INT8 codes相同。只改变已声明的scale／补偿常量，不增加图节点、工作区格式或新的SHAVE ELF。

实际内部基线与gate-up的预热调用都保持30次提交、0新session、0显式host copy。一次诊断计时分别0.050646秒和0.050065秒；各只计一次，不构成速度提升结论。ffn在局部检查被拒绝，未进行其整网预热计时。

## 内部真实NPU对照

原始logits质量门槛保持相对L2<0.03且KL(reference||actual)<0.02；候选晋级还须通过位置更多，平均／最大L2、最大KL和相同next-token数量均不变差。原局部检查也保持relative L2<0.003、max_scaled_error<0.01。

| 配置 | 通过位置 | 平均logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| gate-up | 242/290 | 2.064265% | 13.430156% | 0.004749349 | 286/290 |
| ffn | 未完成整网评分 | — | — | — | — |

ffn在首条轨迹4的第30层（编号29）触发局部门槛：L2=0.0006528815，max_scaled_error=0.0135399299；L2低于局部门槛，但scaled max高于0.01，因此停止，不能把缺失位置算为通过或继续晋级。

第一次运行暴露选择器对局部拒绝处理不完整。已补齐：已知的局部精度拒绝记录为不合格候选；不健康设备、未知异常、不完整的成功记录仍令研究失败。候选的参数、拟合代码和所有门槛未变。重新拟合的NPZ逐字节相同，基线／gate-up的全部位置与局部检查逐项复现；ffn第二次仍在同一层以相同误差被拒绝。最终冻结选择为baseline。

## 同一真实前缀下的末层诊断

先用ffn候选在NPU运行轨迹4的前29层，保留实际中间Tensor；随后baseline、gate-up、ffn三种第30层均使用该相同输入。三个方案的实际attention残差／归一化边界哈希相同，输入在诊断前后不变。导出图、分段普通FFN、融合整层及down控制尾段在相应比较中逐项相等，排除本次观测改图造成输出变化。

| 末层配置 | 整层scaled max | 同实际输入FFN scaled max | SHAVE SiLU后整层scaled max | 最坏坐标actual / reference |
|---|---:|---:|---:|---:|
| baseline | 0.013540070 | 0.003527205 | 0.013540070 | -5.9375 / -5.5625 |
| gate-up | 0.013539894 | 0.003527189 | 0.013539894 | -5.9375 / -5.5625 |
| ffn | 0.013539930 | 0.003527190 | 0.013539930 | -5.9375 / -5.5625 |

三个末层都在同一0起始坐标(batch=0, token=9, feature=199)出现峰值：actual=-5.9375、CPU分阶段reference=-5.5625，相差-0.375。换回原末层参数或使用已有SHAVE FP32 SiLU均没有消除这个峰值。单独FFN同实际输入检查的scaled max约0.003527，仍在0.01局部门槛内。

进一步固定设备norm1，重放CPU attention；再分别使用实际／参考attention残差和norm2运行CPU FFN，构造有符号输出差值。最坏坐标四项为：

| 边界项 | 该坐标的输出差值 |
|---|---:|
| norm1变化经CPU attention／FFN传播 | 0 |
| attention实际残差与同norm1的CPU attention之差经CPU FFN传播 | -0.375 |
| norm2在同实际残差下的差异 | 0 |
| FFN在同实际残差／norm2下的设备算术差异 | 0 |

四项精确重建-0.375，三个末层配置一致。attention残差自身相对L2约0.01095%、scaled max约0.001903，之后在该输入的FFN中被放大。norm1／norm2的同输入相对L2分别约0.000653%／0.000333%，本坐标的上述传播项为0。

这是对一个最坏坐标的控制重放，不能推广为所有坐标、其他轨迹或整个模型的误差归因。尚未在attention内部区分Q/K/V投影、QK、softmax、PV和输出投影；没有认定某个硬件原语违反精度契约。

## 验证与边界

- 8个正式NPU进程，共1,839次记录提交；6个完成预定检查，2个按原局部精度门槛拒绝候选。全部正常shutdown，context_failed=false，live buffers、缓存session/block和隔离字节归零。没有挂起、强制终止或设备重置。
- 12项新增离线检查覆盖scale约束、已知偏移改善、不可变参数、零能量、非法输入／FP16溢出、CPU模型参数替换、失败候选排除及峰值坐标。完整工作区293项检查通过（255+32+6）；公开副本280项检查，279通过、1项因缺少嵌套source-release副本按原规则跳过，该检查已在完整工作区通过。
- 120组图控制通过；228份fixture C/ELF/汇编/manifest与父分支及公开副本逐字节相同。生产后端、原生C/C++、SHAVE ELF、运行DLL/PYD均未修改，无新依赖、环境、模型下载、图表或持久图缓存。
- 30层decoder均在NPU；embedding、末端norm/lm_head、校准和参考计算仍为显式CPU边界，没有CPU推理回退。
- 候选未晋级，本轮未执行独立前向/KV、PPL和自由生成认证；整网decode没有新增质量认证，decode控制仅验证图结构。未新增INT4/NF4、低位训练或任意HF模型支持。
- 本机运行树含S46损失版本保护，它不参与此次冻结decoder；公开分支沿用S50父分支生产后端。实际源码／运行指纹按下表记录。
- 基于[PR #6](https://github.com/LCS-Official/torch-intel-npu/pull/6)保存独立研究；旧PR、main及v0.2.0 tag保留原内容。

## 后续方向

固定INT8 codes后，仅移动scale／重拟合低秩因子的这一方案没有带来整网收益。优先在相同实际输入上拆分attention内部误差，确认产生峰值的路径；后续校准若继续，应考虑实际设备投影输出而非只拟合理想CPU重建。任何新方案仍需先冻结内部选择，再进行独立质量／PPL／生成检查，本轮不据诊断结果修改已冻结候选。

## 复现

需要父分支S50及既有模型／运行环境；所有NPU命令串行。prepare／fit拒绝覆盖已有结果。ffn的internal命令预计以非零状态结束，应核对生成JSON中的healthy=true和local_guard_failure；这不是驱动崩溃。未知失败不可忽略。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_scale
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_quality.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_quality.py -ScriptArguments @("fit")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_quality.py -ScriptArguments @("internal","--profile","baseline")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_quality.py -ScriptArguments @("internal","--profile","gate-up")
try {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_quality.py -ScriptArguments @("internal","--profile","ffn")
} catch {
  $result=Get-Content npunlock-probe\research\results\w8-scale-quality\internal-ffn.json -Raw | ConvertFrom-Json
  if ($result.status -ne "failed" -or -not $result.healthy -or -not $result.local_guard_failure) { throw }
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_quality.py -ScriptArguments @("select")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_scale_audit.py
```

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_scale_quality.py | `89cbc5a302ae28ce51e3e29e91204fbbf4ab0c6fdb5f2309b9efb929b65dd0db` |
| npunlock-probe/research/w8_scale_audit.py | `881bbce33cc5c4cdc8a7385ed68171bcd062fa29063dad5039f507606cae8d7e` |
| test_torch_npu_w8_scale.py | `e59bde009317dfea0604a770d97f2699630349f2caf96b405628adcd278ec30f` |
| torch_npu_lowbit.py | `1c579ad4f2de9528d3fbd407c30df52bcfd29ac11b63cc202ffacd9c9137b01d` |
| torch_npu_lowbit_correction.py | `585642159d23366afcff326b95db8189d4d960509e68e2ab7a6bb6f94d44aec5` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时scale-fit.npz | `bf83484623cf3ed6ad0d0712ed2e99def50c7533625d57e70ea802b12d34a0ba` |

| 临时进程记录 | 状态 | 提交数 | SHA-256 |
|---|---|---:|---|
| initial/internal-baseline | passed | 330 | `b93e5e3b250cd8c2b95529a3110cd0ec7c293cc66fc290fa6ec0edf93b6ab5e3` |
| initial/internal-gate-up | passed | 330 | `fde1d14b3ea948190559f80baeaed0c42a6051966ffe15c944df5270d5daa72a` |
| initial/internal-ffn | failed | 90 | `0816f3858b86d5dfa34eb80c78445c108fe764ac653e88ee5f9c2ecef469850c` |
| internal-baseline | passed | 330 | `557533eb9b2b7f5975ef7742607eb70210fbd52ce725ffa5ba29f0a78da44a7d` |
| internal-gate-up | passed | 330 | `f8f2685838726e1de8c34dff0609f7be0778750ac58780727e277896777115d1` |
| internal-ffn | failed | 90 | `ed83673322defe2379240b020c6f5de283958972c65c9babeb84040909e131fe` |
| initial/diagnostic | passed | 168 | `c36f1b509e10da951c0f89fe5676b3562ad204ef7725ffffa937305ec2d89ade` |
| diagnostic | passed | 171 | `3f3a040e2ed0e8d53d551579fc465e3b91ad6d32490f3c488ab616e766bdd803` |

## 清理

报告审计后清理本轮原始NPZ、JSON、日志与重复诊断快照；保留实现、测试、运行组件与本报告。实际清理统计随后追加。

本轮删除40份临时NPZ、JSON、日志与两份重复诊断源码快照，共139445076 字节（132.985 MiB）。清理前后核对1949份源码、内核、文档、公开副本与运行组件SHA-256，全部一致，随后追加本清理记录及空间台账。研究目录仅保留REPORT.md；原模型、环境和全局缓存保留，本轮测试进程均已退出。
