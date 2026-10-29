# S57：末层精度守卫、SiLU 与解压权重舍入诊断

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；沿用本地SmolLM2-135M-Instruct与W8＋rank32研究配置。

本轮原样复现[S56](../w8-native-selection/REPORT.md)的两次zipper末层精度拒绝，并定位到同一个`(batch=0, token=61, feature=17)`坐标。按固定顺序重放，原0.5差异出现在SiLU阶段。基线换用SHAVE激活／乘法后，该层scaled max从0.0106282降至0.0047944，进入原0.01门槛；**旧候选仍为0.0102929，未修复**。

进一步上传CPU精确计算的激活乘积，候选仍失败。在该输入和坐标，NPU down主投影为-583.5，精确`INT8 code × FP16 scale`权重参照为-584.0，先解压到FP16的参照也为-583.5。低秩delta相同，残差565.0相加后成为-18.5／-19.0的差别。因此，单独替换SiLU不足以解决候选的精度边界；后续应让拟合代理更贴近实际的FP16权重解压语义，并保持原FP16 teacher与验收门槛。

本轮是问题定位与受控反事实，**没有修改生产默认、原守卫或模型参数，没有宣称低位整网质量已解决**。低秩选项仍默认关闭。

## 范围、重建与参照

- 重建相同的W8编码、FP16 scale及rank32输入／输出因子；两份基线NPZ、S55/S56第二轮拟合种子NPZ的SHA均与历史一致。仅为复现旧参数，未尝试新学习率、校准集或候选。
- 基线及固定旧候选`tail-e2-a100`各测试两条已查看的提示：S56的zipper，以及原内部index4作为原守卫通过的对照。无新的独立质量集，没有重选或重训。
- 各配置运行自身原前29层，取得末层真实NPU输入；**同一配置内**的所有分段／反事实使用相同末层输入，并由哈希核对。两个配置的前缀参数不同，不将它们称为共享同一个输入。
- 独立局部CPU参考仍使用原实现：FP32归一化／attention中间计算与明确Half边界，W8主投影按精确`code × scale`权重的FP64乘加后转Half，低秩两次FP32乘加各转Half。没有换参照以改变原通过标准。
- NPU执行全部模型前缀、被观测的层、DPU子图及SHAVE反事实算术。CPU用于固定模型/token、独立参考及明确标记的诊断输入；不是CPU回退的模型执行路径。

## 原故障与对照

原局部门槛为同输入relative L2<0.003、scaled max<0.01。scaled max分母为`abs(reference) + 每行RMS(reference)`；表中只记录最坏scaled坐标。

| 配置／提示 | 最坏坐标 | actual / reference | relative L2 | scaled max | 超0.01元素数 |
|---|---|---:|---:|---:|---:|
| baseline/zipper | (0, 61, 17) | -17.0 / -17.5 | 0.0006576719 | 0.0106282039 | 1 |
| baseline/original-4 | (0, 9, 17) | -48.25 / -48.75 | 0.0006518046 | 0.0070544155 | 0 |
| tail-e2-a100/zipper | (0, 61, 17) | -18.5 / -19.0 | 0.0006463178 | 0.0102929156 | 1 |
| tail-e2-a100/original-4 | (0, 27, 17) | -22.75 / -22.5 | 0.0006520955 | 0.0050488333 | 0 |

zipper的峰值位于从0开始计数的token位置61，是被评分后缀中的换行符。两个配置各只有一个元素越scaled门槛；`max_abs=4.0`出现在其他坐标，不能把它当作这里的0.5差异。对照提示的最坏坐标和误差来源不同，说明不是任何输入都会在相同位置触发拒绝。

前29层均保留原局部检查，末层也完整记录失败。后续只运行明确的诊断子图，不继续整网质量计分；本轮没有新的有效268位置NPU通过率。

## 固定坐标的累积重放

复用S52 attention工具及S50 FFN观测图，分别导出Q/K/V、RoPE、QK、softmax、PV、gate/up、SiLU、乘积和down链。切图输出、原段图、融合整层及原输入重放必须逐项相同，防止观测本身改变计算。

固定在各自原最坏坐标，依次用实际NPU中间结果替换CPU参考中的同名边界。下表为**有符号增量**，同一行精确相加等于原NPU与参考的差值；这是固定替换顺序下的分解，不是唯一因果归因，也不能把有符号增量绝对值当作全Tensor误差占比。

| 配置／提示 | norm1 | attention | norm2 | gate-up | SiLU | multiply | down | residual-add |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline/zipper | +0 | +0 | +0 | +0 | +0.5 | +0 | +0 | +0 |
| baseline/original-4 | +0 | +0.25 | +0 | +0 | +0 | +0 | +0.25 | +0 |
| tail-e2-a100/zipper | +0 | +0 | +0 | +0 | +0.5 | +0 | +0 | +0 |
| tail-e2-a100/original-4 | +0 | -0.25 | +0 | +0 | +0 | +0 | +0 | +0 |

两份zipper的0.5差值都在SiLU替换时出现；同输入attention QK、softmax、PV及组合FP32替换均未降低该峰值。对照中，基线的0.5由attention和down各0.25重建，候选的-0.25来自attention；因此不推广为“attention没有误差”或“所有末层问题都来自SiLU”。

同输入原DPU SiLU与显式CPU FP32 SiLU再转Half存在约0.00018的相对L2差异。原乘法差异远小于激活差异，但反事实更换的输入会改变后续舍入，不能仅凭原路径重放预测改动后的设备结果。

## SHAVE 激活／乘法反事实

在原实际gate/up上运行已有SHAVE FP32 SiLU、转Half及设备乘法，再进入原down DPU尾图；保留相同残差、权重和低秩因子。此处同时更换激活与乘法，单独的SiLU定位由前面的累积重放提供。

| 配置／提示 | 原scaled max | SHAVE反事实scaled max | 原峰值坐标的新输出 |
|---|---:|---:|---:|
| baseline/zipper | 0.0106282039 | 0.0047944011 | -17.5 |
| baseline/original-4 | 0.0070544155 | 0.0049726218 | -48.5 |
| tail-e2-a100/zipper | 0.0102929156 | 0.0102929156 | -18.5 |
| tail-e2-a100/original-4 | 0.0050488333 | 0.0050488333 | -22.75 |

基线的局部拒绝被消除，候选没有。两种对照也都处于原门槛内。本轮没有据此切换默认或运行自由生成／PPL；S50对所有层使用SHAVE激活的原型曾出现内部质量退化和额外提交，本轮不重复将局部改善当作整网改善。

## 下游投影：分开激活精度与权重解压

为理解候选为何仍失败，第二个驱动只保留down链，导出主投影、低秩reduced／delta、校正相加及残差相加。原输入下五个导出结果与整层对应结果逐项一致。固定三种输入：

1. `original`：原DPU激活／乘积。
2. `shave-product`：上述SHAVE激活及设备乘积。
3. `cpu-product`：相同实际gate/up在CPU显式FP32 SiLU→Half→Half乘法，作为诊断输入上传；down计算继续在NPU上进行。

SHAVE与CPU严格激活的全Tensor逐项一致比例为基线99.993896%、候选99.995929%，仍有少量Half差异。上传CPU乘积后，候选峰值及以下标量不变，排除了这些SHAVE残余差异作为该峰值的解释。

| 配置／输入 | NPU main | 精确权重main | FP16解压权重main | NPU delta | NPU最终输出 | 同输入CPU最终输出 |
|---|---:|---:|---:|---:|---:|---:|
| baseline/original | -582.0 | -582.0 | -582.0 | 0.1973876953125 | -17.0 | -17.0 |
| baseline/shave-product | -582.5 | -582.5 | -582.5 | 0.1981201171875 | -17.5 | -17.5 |
| baseline/cpu-product | -582.5 | -582.5 | -582.5 | 0.1981201171875 | -17.5 | -17.5 |
| tail-e2-a100/original | -583.5 | -583.5 | -583.5 | 0.1953125 | -18.5 | -18.5 |
| tail-e2-a100/shave-product | -583.5 | -584.0 | -583.5 | 0.1944580078125 | -18.5 | -19.0 |
| tail-e2-a100/cpu-product | -583.5 | -584.0 | -583.5 | 0.1944580078125 | -18.5 | -19.0 |

两组所有表中残差均为565.0，原完整局部参考分别为-17.5／-19.0。候选严格输入下，delta约0.194458，小于这一主投影数值处的Half半ULP0.25；校正相加仍分别舍入至-583.5／-584.0，最终残差抵消保留0.5差异。基线的三个main参照在这个坐标一致，SHAVE／CPU严格输入均恢复原参考输出。

源码中的`torch_npu_lowbit.weight_graph`明确将INT8转换为f16，再以f16乘scale，随后进入输出f16的MatMul。精确`code×scale`参考与这一中间解压舍入存在数值语义差别。候选的NPU标量与FP16解压参照一致；全矩阵main的相对L2也更接近FP16解压参照（约0.000175，对精确权重约0.000544），但仍非逐项相同，不能将这次观测当成整个编译器／DPU累加过程的精确重建或任意模型的根因证明。

这解释了一个重要边界：在原输入上“后段贡献为0”，并不保证更换激活后后段仍贡献0。原候选main的三个值都是-583.5；严格激活输入改变后，精确与FP16解压两种参考落到了相邻Half值。前面单路径的累积重放和真正的NPU替换因此给出不同结果，两份记录均保留。

## 判断与下一步

该复现不支持直接修改生产SiLU默认或放宽scaled门槛。基线有可消除的激活路径差异，但固定旧候选还存在解压权重／投影舍入边界；整网量化泛化也仍未通过S56的内部晋级。

下一轮优先建立**显式模拟FP16权重解压的拟合代理**，先对末层down路径做小范围试点，再用实际NPU核对。在该代理下重新评估低秩残差拟合目标，原独立FP16 teacher、原局部守卫和整网晋级规则维持。需要另留未查看的提示验证；本轮两条提示只承担调试和回归职责。暂不扩大模型、精度种类或把大矩乘改为慢速SHAVE FP32。

## 资源与回归

- 4个正式诊断进程共**3,260次提交**，均完成自身诊断断言并健康shutdown；context_failed=false，活跃buffer、缓存session／block及隔离字节归零，无挂起、强制结束或驱动重置。
- 120个原模型局部检查中118通过，zipper两处原拒绝如实记录。诊断进程完成不表示原模型全部通过精度门槛。
- 54项切图／观测／CPU参考一致性控制逐项相同。四组原末层场景和两组三输入down尾图均验证原输入观测不扰动结果。
- 默认1536图／512MiB预算未改，四份记录淘汰均为0。主诊断后端Tensor峰值8,060,420字节，down诊断7,912,960字节，不含全部模型／驱动内存。没有独立吞吐或功耗基准。
- 新增10项离线检查，覆盖FFN累积替换、Half边界、非法输入、固定坐标有符号重建、切图导出与剩余依赖保护。完整工作区**364项检查通过**（326+32+6）；公开副本351项，350通过、1项原有嵌套source-release检查跳过，该项在完整工作区通过。
- 228份C／ELF／汇编／manifest fixture与父提交、root及公开副本逐字节相同；12份旧源码／运行时／模型身份不变。未修改生产后端、内核、DLL/PYD、依赖、环境或模型权重，未下载新模型。
- 公开分支`research/w8-final-layer-audit`基于[S56 PR #12](https://github.com/LCS-Official/torch-intel-npu/pull/12)。完整工作区保留不参与冻结decoder的S46分类损失版本保护；公开副本沿用父分支生产后端。main、旧PR及v0.2.0 tag保持原值。
- 仅覆盖同一个135M模型、B1/S64、W8＋rank32的这些已查看输入。没有新增INT4／NF4、整网质量通过率、PPL、KV decode、自由生成、长期训练或跨平台认证。

## 复现

使用已有项目环境和本地模型；首次prepare同时重建旧数据。文件已存在时拒绝覆盖，拟合只是重建S55/S56相同种子。硬件命令串行；诊断脚本记录原局部拒绝，但不执行其后的整网质量评分。未知异常／控制不一致将使诊断进程失败。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_final_layer
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_final_layer_audit.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_final_layer_audit.py -ScriptArguments @("fit")
foreach ($profile in @("baseline","tail-e2-a100")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_final_layer_audit.py -ScriptArguments @("diagnose","--profile",$profile)
}
foreach ($profile in @("baseline","tail-e2-a100")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_final_layer_down.py -ScriptArguments @("--profile",$profile)
}
```

本机重建的三份NPZ与历史相同；JSON中的时间、来源元数据不承诺跨运行逐字节复刻。清理后需要重新prepare／fit；中间激活只在内存中使用，磁盘仅临时保存权重／因子、token和标量记录。

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_final_layer_audit.py | `2d631c7264979af1c3c5a328aef35e32a8501df5fea6b42ed1b5bf94383fd95f` |
| npunlock-probe/research/w8_final_layer_down.py | `879942aa7e11e3fa14a1a4894afe11e544f346050c7cf0cd165996a2341b7012` |
| test_torch_npu_w8_final_layer.py | `ce923a0321432b960b2818bda0db10579cb2d925c0417a7f18701bea4f48c940` |
| npunlock-probe/research/w8_native_selection.py | `d83a9f8d2c2a21c3ee53d6375bed40be0a070bd78c5467a3e7f3ca6bfb7ddd6e` |
| npunlock-probe/research/w8_tail_quality.py | `0d4a61bea978c16b6e630e3504d8a5d9dc6eb78d936cf93f25f397922a487b61` |
| npunlock-probe/research/w8_tail_audit.py | `98a05e20d4c1650bef21315d5d130279d08665fb7d055e2cc8fc8bb130eeac58` |
| npunlock-probe/research/w8_logits_quality.py | `a865d2805c0aa87e1e29c6f534c1a51d76023f9b992e5c02fa48171e43bb7d25` |
| npunlock-probe/research/w8_attention_quality.py | `d1ebe9b1acd4fd9e429bed98d0561a30a8018c29cfef6ac66ab9a967d99152a3` |
| npunlock-probe/research/w8_gate_quality.py | `bc7e56ab7b6bb19291fb107b876f467e264f2cca409f02980494e1777d0e40b2` |
| npunlock-probe/research/w8_scale_audit.py | `881bbce33cc5c4cdc8a7385ed68171bcd062fa29063dad5039f507606cae8d7e` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时logits-fit.npz | `73e5fbde2b5afe651ffa84e55b3336a23438d94c325cbd384acc724f619e6863` |
| 临时fresh.json | `5b5a24c777585383c8a2c0cad3b843f00dd7c21bad40cea7ca3dc483f5abedbe` |
| 临时logits-fit.json | `13eeb7e7bcb35565be704dd60c91002bb606c709a9206ce675a03b6159769d7f` |
| 临时audit-summary.json | `0e996fb1a776f7919e6ccdff2fb451d0432b2928391b099ba3842e38d2ae9216` |

| 临时NPU记录 | 提交 | SHA-256 |
|---|---:|---|
| diagnostic-baseline | 1516 | `05a7653c7e5a4ea42785468a2d2f432209c6f7fc5c8459196ea7b8567106d853` |
| diagnostic-tail-e2-a100 | 1516 | `2939881e7e08f715dbe4b313ffeea5303d94cde79a3269ff8b20d8dd2513e51d` |
| down-baseline | 114 | `75721fb48ccb967a13e3ad4c662a1e17bcd08398ea2775bcf2fb5149fc26545c` |
| down-tail-e2-a100 | 114 | `7ce331dfda259fce7a99ed960da2727c8692f3ba080c7ea7609dc89a4a46bc12` |

## 清理

2026-10-29末次删除本轮结果目录30份临时NPZ、JSON和日志（含重建参数、token、诊断／审计及清理核对清单），共134,595,615字节（128.360MiB）。清理前后1,991份保留源码、内核、文档、公开副本及运行组件SHA-256全部一致，随后补充本段及存储台账。结果目录仅保留REPORT.md；模型、环境和全局缓存保留，4个正式NPU诊断进程均已退出。
