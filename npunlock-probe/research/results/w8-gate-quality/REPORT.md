# S50：gate/up/SiLU拆分与SHAVE激活原型

2026-10-28，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；沿用已有SmolLM2-135M-Instruct。

本轮分离gate/up投影、SiLU和门控乘法。gate/up误差的传播与SiLU局部差异量级相近，乘法差异明显更小。改用已有SHAVE FP32 SiLU及设备乘法后，20个局部FFN取样都更接近CPU参考。

整网内部原型通过243→246/290，平均与最大L2略降，但最大KL略升，预热提交30→660次。**未满足预先定义的晋级条件，保留生产默认及原W8+rank32实验基线**；严格激活路径仅作为研究原型，不宣称整网质量或速度提升。

## 固定范围与观测控制

- 重建S44的40条CPU轨迹、W8权重和rank32因子，NPZ与历史基线哈希相同；没有重拟合S48候选或调整权重。
- 固定内部轨迹4/19、层0/6/14/24/29。forward为B1/S64；decode先真实prefill63个token，再处理最后一个token。共20个层取样，诊断不改变后续层原路径输入。
- 实际attention残差和SHAVE RMSNorm输出均在NPU产生，诊断只在内存使用中间Tensor，不保存激活/logits。没有CPU推理回退。
- 从原FFN导出15个输出：gate/up/down各自的主矩乘、低秩中间值、补偿输出、校正输出，以及SiLU、门控乘积、最终输出；保留decompression与Swish版本。native上限64参数，当前2输入+15输出在范围内。
- 每项使用相同实际前驱输入。主W8分别比较精确code×scale权重/FP64累加、Half舍入权重/FP32累加；其他节点比较实际操作数，不把前缀误差混入本算子。不能仅凭两个主矩乘参考推断硬件采用哪种舍入。
- 为排除观测改图的影响，核对诊断图=普通FFN、分段FFN=融合层、down控制尾段=诊断图。20处均逐元素数值相等，decode的30个旧KV/位置数组保持原值。

## 同输入局部误差

表中为每阶段自身CPU输出范数归一化的相对L2，不能直接比较为最终logits贡献比例。

| 路线 | 阶段 | 平均相对L2 | 最大相对L2 |
|---|---|---:|---:|
| forward | gate_main_exact | 0.025484% | 0.026336% |
| forward | up_main_exact | 0.025437% | 0.026922% |
| forward | down_main_exact | 0.031216% | 0.057425% |
| forward | silu_same_input | 0.032162% | 0.036754% |
| forward | multiply_same_input | 0.000807% | 0.001655% |
| forward | shave_silu_same_input | 0.000017% | 0.000090% |
| decode | gate_main_exact | 0.025452% | 0.028055% |
| decode | up_main_exact | 0.024966% | 0.026642% |
| decode | down_main_exact | 0.029313% | 0.053914% |
| decode | silu_same_input | 0.035688% | 0.045648% |
| decode | multiply_same_input | 0.000000% | 0.000000% |
| decode | shave_silu_same_input | 0.000000% | 0.000000% |

gate/up/down的两次补偿和相加另有同输入记录。main指压缩主矩乘；SiLU使用同一个gate，乘法使用同一组SiLU/up。SHAVE FP32 SiLU及Half转换都在NPU执行。SiLU差异可降低，但未证明厂商实现违反精度契约，也未确认FTZ或驱动错误。

## 门控乘积的有符号分解

```text
reference  = half(silu_fp32(cpu_gate)) * cpu_up
strict     = half(silu_fp32(device_gate)) * device_up
observed   = device_silu * device_up
total      = device_gated - reference
projection = strict - reference
activation = observed - strict
multiply   = device_gated - observed
```

乘法边界均转Half。20处满足total=projection+activation+multiply，重建误差为0。统一以reference门控乘积范数作分母；这些是有符号向量的范数，可相互抵消，不能把范数直接相加当作误差占比。

| 路线 | 总误差均值 | gate/up传播均值 | SiLU局部均值 | 乘法局部均值 |
|---|---:|---:|---:|---:|
| forward | 0.051615% | 0.038043% | 0.036207% | 0.000807% |
| decode | 0.056534% | 0.034517% | 0.042500% | 0.000000% |

projection是gate/up投影差异通过SiLU/乘法后的传播，activation是同gate/up下SiLU差异的作用；两者均需保留，不能将整个门控误差只归因于SiLU。

## 设备局部反事实

将实际gate换用已有SHAVE FP32 SiLU，转Half并以已有设备乘法生成门控值；原权重、rank32因子、主投影与残差保持。down尾段先用原门控值重现原输出，才输入替代门控值。

| 路线 | 原FFN/CPU参考L2均值 | 替代激活后L2均值 | 改善取样 |
|---|---:|---:|---:|
| forward | 0.024835% | 0.022002% | 10/10 |
| decode | 0.033216% | 0.024343% | 10/10 |

局部改善不等于整网质量改善，因此接着进行内部整网原型对照。

## 内部整网对照与晋级

原型拆为attention、RMSNorm、gate/up前段、SHAVE FP32 SiLU及Half乘法、down尾段，这些decoder阶段均在NPU；embedding、末端norm/lm_head和CPU参考仍为显式宿主边界。未改变压缩权重或补偿因子。首条内部轨迹30层检查导出图=原融合层、gate/up前段=导出值、down控制尾段=原输出，共120项逐元素相等。另30项staged CPU检查满足原relative L2<0.003、scaled max<0.01。

固定内部轨迹4/9/14/19/24/29/34/39，共290位置，两路线各独立进程。每位置门槛仍raw logits L2<0.03且KL<0.02。运行前已在代码定义：候选须更多通过位置，且平均L2、最大L2、最大KL均不变差，token一致数不下降，否则保留基线。

| 路线 | 通过 | 平均raw L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.0419% | 13.3917% | 0.0040497 | 286 |
| strict | 246/290 | 2.0394% | 12.9089% | 0.0041673 | 286 |

候选多通过3处，但最大KL从0.0040497升至0.0041673，故未通过保守晋级规则。两者最大KL都低于原0.02门槛；拒绝的是不劣化要求，并非声称KL超过原门槛。没有根据结果放松规则。

## 代价与资源

| 路线 | 单次预热提交 | 单次诊断计时 | 新session | 显式host上传/下载 |
|---|---:|---:|---:|---:|
| baseline | 30 | 0.049413秒 | 0 | 0 |
| strict | 660 | 1.996055秒 | 0 | 0 |

原型提交22倍，观察计时约2.00秒对0.049秒。各只计一次完整预热调用，含Python/图调度，非隔离背景负载后的性能基准；不代表硬件SiLU本身慢相同倍数。当前拆图代价明显，不作为默认速度优化。预热输出与前次调用逐项相等，编译与host copy计数不增。

## 验证与边界

- 四个正式NPU进程共7,130次记录提交，全部正常shutdown；context_failed=false，live buffers、cached blocks/sessions和隔离字节归零。默认1536图/512MiB不变。提交数包含实际记录的冷启动参数准备等调用，不只计算模型层数。
- 本地281项离线检查通过：根目录243（含9项新拓扑/元数据/有符号误差/晋级测试）、Python API32、发布工具6。独立公开分支268项检查：267通过、1项因缺少嵌套source-release副本按原规则跳过；该对照在完整工作区已通过。
- 审计核对20个取样的480项局部比较、20次有符号重建、120项原型控制和30项staged对照，复算580个内部质量位置与冻结选择。
- 228份fixture C/ELF/汇编/manifest与父分支字节相同。没有新SHAVE ELF、C源或宿主DLL/PYD构建；运行时编译的是已有算子的诊断/原型图。没有新依赖、环境、模型、图表或持久图缓存。
- 候选未晋级，本轮不运行独立前向/KV质量、PPL或生成，也不引用旧候选语言结果证明新原型。decode仅是10个局部取样，未接入严格激活整网KV。未新增INT4/NF4、低位训练或任意模型认证。
- 本机运行树包含S46分类损失版本修复，它不参与本次冻结decoder路径；公开分支沿用S49父分支的生产后端，实际本机运行指纹按下表记录。
- 新研究基于[PR #5](https://github.com/LCS-Official/torch-intel-npu/pull/5)，保留旧PR与v0.2.0 tag。生产默认、模型图和质量门槛未改。
- 后续优先研究投影量化与校准误差；SiLU精度路径只有在质量和融合代价均可接受后再考虑生产接入，本轮局部改善不能推广为整网提升。

## 复现

需要父分支S49/S48工具、已有模型及运行环境。prepare拒绝覆盖已有权重，清理后需重新准备，各NPU命令串行。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_gate
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_gate_quality.py -ScriptArguments @("prepare")
foreach ($route in @("forward","decode")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_gate_quality.py -ScriptArguments @("probe","--route",$route)
}
foreach ($mode in @("baseline","strict")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_gate_experiment.py -ScriptArguments @("internal","--mode",$mode)
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_gate_experiment.py -ScriptArguments @("select")
```

## 源码与运行身份

| 文件/产物 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_gate_quality.py | `bc7e56ab7b6bb19291fb107b876f467e264f2cca409f02980494e1777d0e40b2` |
| npunlock-probe/research/w8_gate_experiment.py | `e712e5c1a2dde8567dd2cff7b3b47f05e281136c394ff7ef06b0e5668fa67017` |
| test_torch_npu_w8_gate.py | `e63d7db221e4d545072931dbb5f3a34d669b09eb7f4ffb71b1bb446b4986e4c7` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时 weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时 corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时 probe-forward.json | `0835a2735fe402b94faa033288948eae0ea9c509b973e3be6ad3b09dd0d4df35` |
| 临时 probe-decode.json | `be923052c3f1f7dc5130a1d15e01577a237c7bb73c1730793f325d61759b2895` |
| 临时 activation-internal-baseline.json | `5099285962034d401fccc2de2f79e6d68818a92b6149245ca70765f65c9d059a` |
| 临时 activation-internal-strict.json | `62d9724f628c3b12474cb47666e0613f723113fccd190ea8b401ab2af6a88ea9` |
| 临时 activation-selection.json | `0405dd1b4ae41cd4cd9560574696c17acfbe306a83f7e8b14575e217065f7155` |

## 清理与交付

保留两个研究驱动、9项新离线测试与本报告；分支保存历史。原始NPZ、JSON与日志在报告审计后清理，统计随后追加。

本轮删除22 份临时NPZ、JSON和日志，共116141648 字节（110.761MiB）。清理前后核对1945份源码、内核、文档、公开副本与运行组件SHA-256，全部一致，随后追加本清理记录及空间台账。研究目录仅保留REPORT.md；原模型、环境和全局缓存保留，本轮测试进程均已退出。
