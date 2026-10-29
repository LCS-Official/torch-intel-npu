# S53：真实NPU Q/K/V输出校准与整网对照

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；使用已有本地SmolLM2-135M-Instruct。

本轮把S52的诊断转为参数校准：只更新现有Q/K/V rank32补偿的输出矩阵，以真实NPU主投影和低秩中间值为观测、原FP16投影为目标。三个固定强度均降低校准代理SSE，但没有通过原整网晋级规则；生产默认保留基线。

0.50强度的平均logits L2由2.041913%降至2.016721%，最大L2/KL也降低、token一致数286→287，但原质量门槛通过数243→241/290。1.00强度通过数仍243/290，平均L2及最大KL略升。不能据部分指标改善宣布低位整网质量问题解决。

## 固定范围与拟合目标

- 重新生成原40条CPU FP16 greedy轨迹、W8权重及rank32因子；两份NPZ均与S44/S52历史指纹逐字节相同。32条index%5!=4的轨迹用于拟合，每个矩阵2048行；8条index%5==4只用于内部筛选。没有把内部轨迹混入拟合，也没有新模型／语料下载。
- 真实NPU以原baseline参数推进全部30层、B1/S64前缀；每层从相同NPU-normalized输入观测Q/K/V的main、reduced、corrected输出。目标是该输入经原FP16权重的CPU FP32线性计算再Half舍入，属于局部投影目标；它不直接优化原模型整网输出。
- 90个Q/K/V投影只更新已有rank32补偿的up矩阵；INT8 codes、FP16 scales、down矩阵、其他四类投影、参数字节数、推理图拓扑与默认softmax均不变。
- 基于已有固定输入基的岭回归思路，用FP64解 B 的最小二乘：`||Z B^T - (Y_fp16 - M_npu)||²/m + lambda ||B-B_prior||²`；`lambda=0.01*mean(diag(Z^T Z/m))`，并保护零能量输入。
- 三种强度alpha=.25/.50/1.00，保存`Half(B_prior+alpha*(B_fit-B_prior))`。只有CPU代理`Half(M_npu+Half(Z_npu B^T))`的校准SSE严格下降，才接受该矩阵；保持两次Half边界，非有限或溢出拒绝。
- 所有拟合解算属于CPU准备，实际推理和整网对照中的decoder计算仍走NPU；Embedding、末端norm/lm_head及参考指标是已有显式CPU边界。没有把CPU校准输出回填为模型推理结果。

## 观测控制与校准结果

复用S52保留decompression元数据的26输出图；全部32条轨迹、30层逐项检查导出attention=原融合attention，以及导出残差经原FFN=原融合整层。共**1,920项精确控制**逐元素相同。原baseline NPU输出推进后续层，三个拟合强度共享同一套观测前缀；不以候选输出改变其他候选的校准数据。

| 输出因子强度 | 矩阵 | 原代理SSE | 新代理SSE | 下降 | 接受矩阵 |
|---|---:|---:|---:|---:|---:|
| 0.25 | 90 | 579.668145758 | 578.728970786 | 0.162019% | 90 |
| 0.50 | 90 | 579.668145758 | 578.064761213 | 0.276604% | 90 |
| 1.00 | 90 | 579.668145758 | 577.555808809 | 0.364405% | 90 |

原实际NPU corrected输出的校准SSE合计579.668131671；使用真实main/reduced、CPU重算up的代理基线为579.668145758，两者非常接近但不是逐位相同。上述下降是**已观测输入上的CPU代理拟合收益**，未把它表述为重新执行候选后的完整NPU校准SSE或整网质量收益。原始激活仅在内存使用，没有保存Tensor／logits。

## 完整30层内部NPU验证

沿用S51原执行器、局部门槛、8条内部轨迹和290个位置；整网对照目标为原CPU FP16完整模型。原质量门槛保持logits相对L2<0.03、KL(reference||actual)<0.02。各方案首条轨迹另做全部30层的同输入分阶段CPU检查，保持局部L2<0.003、scaled max<0.01。

| 路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| qkv-0.25 | 240/290 | 2.041305% | 12.863544% | 0.003819845 | 286/290 |
| qkv-0.50 | 241/290 | 2.016721% | 13.121292% | 0.003801033 | 287/290 |
| qkv-1.00 | 243/290 | 2.051543% | 12.966214% | 0.004060413 | 286/290 |

原规则要求通过位置更多，且平均／最大L2、最大KL和next-token一致数均不变差。内部排序的新候选是1.00强度，但它没有超过基线通过数，平均L2和最大KL也变差，最终冻结选择仍为baseline。0.25/0.50虽降低多项误差指标，也没有满足通过数要求；没有放宽门槛或更换选择规则。

按原门槛逐位置对照：

| 强度 | 原失败→通过 | 原通过→失败 | 净通过变化 |
|---|---:|---:|---:|
| 0.25 | 4 | 7 | -3 |
| 0.50 | 4 | 6 | -2 |
| 1.00 | 5 | 5 | 0 |

这不是全部位置一致变好。例如0.50在轨迹4/位置31的L2为0.031244→0.029141，跨过门槛；轨迹19/位置55则从0.025785→0.031281，由通过变为失败。不能把这些变化都归因于贴近阈值的显示舍入；真实逐位置退化保留在评价中。

四种配置的120个逐层局部检查全部通过，未触发精度保护、异常中止或设备错误；原基线完整复现S51/S52的243/290及全部摘要值。

## 提交、资源与数值范围

| 路径 | 单次预热提交 | 单次诊断计时 | 新session | 显式host copy |
|---|---:|---:|---:|---:|
| baseline | 30 | 0.051897秒 | 0 | 0 |
| qkv-0.25 | 30 | 0.051680秒 | 0 | 0 |
| qkv-0.50 | 30 | 0.051113秒 | 0 | 0 |
| qkv-1.00 | 30 | 0.050745秒 | 0 | 0 |

所有候选保持30次提交、相同参数与图预算；重复输出逐项相等。每种计时仅来自一次预热调用，不作为受控性能基准，不声称这些小差别构成加速。五个正式NPU进程共**6,180次提交**，全部健康关闭，context_failed=false；shutdown后live buffers、缓存session/block、隔离字节全部归零。缓存默认1536图／512MiB，五个进程均无淘汰。校准Tensor峰值12,045,824字节，四个内部进程均4,787,712字节；该数值不包含全部驱动／模型驻留内存。

## 验证、身份与限制

- 新增10项离线测试：闭式岭回归解、真实main／reduced输入依赖、两次Half边界、固定down／rank、三种blend、零能量与原值保留、非法输入／溢出拒绝、32/8集合隔离、参数读取身份／形状以及复用执行器在成功和失败后的恢复。
- 完整工作区315项检查通过（277+32+6）；公开副本302项检查，301通过、1项原有嵌套source-release检查按原规则跳过，该项在完整工作区通过。
- 228份fixture C/ELF/汇编/manifest与父分支、工作区及公开副本逐字节相同；S52所列10份源码／运行时／模型指纹保持原值。没有生产后端、C/C++、SHAVE ELF、DLL/PYD、依赖、环境或模型下载变化。
- 本轮只验证B1/S64完整30层前向的内部轨迹；没有新独立前向、KV decode、PPL或自由生成认证，未扩展INT4/NF4或低位训练。候选未晋级，未继续消耗独立验收集。
- 此实验的局部拟合使用NPU前缀，但目标仍是同输入FP16投影。它不会自动补偿之前层的轨迹误差；下一轮若要继续，应把原模型整网logits／KL作为拟合目标，使用新的独立样本检验泛化，而非继续依赖局部SSE下降。此方向仍需实验，不能保证收益。
- 本机运行树含S46分类损失版本保护，不参与冻结decoder；公开研究分支沿用S52生产后端。新研究基于[PR #8](https://github.com/LCS-Official/torch-intel-npu/pull/8)。

## 复现

需父分支S52及现有环境／本地模型。首次prepare拒绝覆盖；每条NPU命令串行执行：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_qkv
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_qkv_quality.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_qkv_quality.py -ScriptArguments @("calibrate")
foreach ($profile in @("baseline","qkv-0.25","qkv-0.50","qkv-1.00")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_qkv_quality.py -ScriptArguments @("internal","--profile",$profile)
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_qkv_quality.py -ScriptArguments @("select")
```

本次选择baseline，独立评测入口按原规则拒绝未晋级配置。原始NPZ／日志清理后需重新prepare和calibrate；保留源码与本报告。

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_qkv_quality.py | `add3cbc212e6045a6bdedb825450de2a24ae7f4655ae860690bfd2c91f9a79e0` |
| test_torch_npu_w8_qkv.py | `b49d505566908e6a923f5ef172e13ccb5008c891be13c41081fd27b17f01a44b` |
| npunlock-probe/research/w8_attention_quality.py | `d1ebe9b1acd4fd9e429bed98d0561a30a8018c29cfef6ac66ab9a967d99152a3` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_npu_lowbit_correction.py | `585642159d23366afcff326b95db8189d4d960509e68e2ab7a6bb6f94d44aec5` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时qkv-fit.npz | `2c8459ad3e0ca3b12be87c034aaed411e9f8d333f2f211afd02801611f45097f` |

| 临时进程记录 | 提交数 | SHA-256 |
|---|---:|---|
| qkv-fit | 4860 | `587fdc08562438c355639774fcaea80362ca6c4d20c472482ccdeeb83690a6e4` |
| internal-baseline | 330 | `80e351f6397fde0a066703ed8f7dbec318faa879c3e35a54d14a0796d9d18dfa` |
| internal-qkv-0.25 | 330 | `7e1f163f32be1e02840df818214124a0832e426ae57445401a41b023e52beb9c` |
| internal-qkv-0.50 | 330 | `d3551c8e3d7166d8801ce236124954b5c4ae38d101e8834075be198c29e3eeea` |
| internal-qkv-1.00 | 330 | `d336caa1ad7bbbd6d9363baf459ea9273f40568d4ceb0ebc862281cfb1d223cf` |

## 清理

2026-10-29实际删除27份本轮临时NPZ、JSON和日志（含身份／审计／清理核对清单），共121855632字节（116.211MiB）。清理前后1,963份保留源码、内核、文档、公开副本与运行组件SHA-256全部一致，随后补充本段及存储台账。研究结果目录仅保留REPORT.md；原模型、环境和全局缓存保留，五个正式测试进程均已退出。
