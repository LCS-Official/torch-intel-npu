# S52：Attention分段数值诊断与FP32 softmax原型

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；使用已有本地SmolLM2-135M-Instruct。

S51已将一个可复现的末层峰值缩小到attention边界。本轮继续拆分：在该坐标的固定顺序重放中，Q/K/V投影和softmax各贡献-0.1875，合计-0.375；其他项在此坐标为0。已有NPU FP32 softmax替换将该样本整层scaled max从0.013540070降到0.007103981。

**局部改善没有变成整网收益。** 完整30层、8条内部轨迹的通过数243→242/290，平均L2略降，但最大L2与最大KL均升高；预热提交30→240。候选没有通过原晋级规则，生产默认及原W8+rank32基线保持不变。

## 固定范围与参照

- 重新准备S44/S51的40条CPU FP16 greedy轨迹及W8/rank32参数；32条校准轨迹仍只用于重建原参数。W8权重、rank32因子和S51 scale-fit NPZ均与历史哈希逐字节一致，没有重新选择scale或拟合新权重。
- focus使用S51的ffn校准候选运行轨迹4的前29层，取实际NPU中间输入，再以baseline参数评估第30层。B1/S64，9个query heads、3个KV heads、head_dim64。
- cross固定baseline/ffn两种前缀、内部轨迹4/19、层14/29，共8个设置；其中一个与focus重复，合计9次取样。两个提示有共享系统前缀，早期位置并非独立随机样本。
- 局部参照使用**相同W8代码、scale和rank32参数**的CPU分阶段计算；它衡量设备数值差异，不等同于W8相对原FP16模型的量化损失。
- 整网内部参照另为官方CPU FP16完整模型，保持原始logits L2<0.03、KL(reference||actual)<0.02。所有局部与整网门槛保持原值。

## 观测与分段控制

从原attention IR导出26个输出：Q/K/V/O各自的主矩乘、低秩中间值、补偿和校正值，加上RoPE／重复后的Q/K/V、QK、缩放、掩码、softmax、PV、展平attended和残差。保留原decompression元数据，检查原scale=0.125、axis=3、因果mask及矩乘转置方向；不匹配的图拒绝诊断。

原mask的负无穷位置必须逐项相同，有限score单独统计误差；不把Inf减Inf产生的NaN混入指标。所有激活仅在内存使用，不保存原始Tensor或logits。

9个取样的导出图=普通attention=融合attention，四种tail控制（QK、masked、probs、attended边界）以及分段FFN=融合整层均逐元素相等，共63项。输入在诊断前后哈希不变。另有30层整网原型的150项精确控制；合计**213项图／分段控制**通过。focus与cross中的重复样本，全部局部比较、重放和四种NPU替换结果逐项一致。

## 同输入阶段误差

下表是focus样本各阶段自身输出范数归一化的相对L2。Q/K/V主投影参照未提前舍入的code×FP16 scale及FP64累加；QK/PV参照FP32累加再Half舍入，softmax参照FP32计算再Half舍入。各行不是最终logits贡献比例。

| 阶段 | 原设备路径相对L2 | 已有NPU FP32替换相对L2 |
|---|---:|---:|
| q-main | 0.0002269040124 | — |
| k-main | 0.0002776879119 | — |
| v-main | 0.000265573588 | — |
| qk | 1.174655335e-05 | 1.220318427e-05 |
| softmax | 0.0002408511937 | 1.294454677e-07 |
| pv | 7.763442227e-06 | 2.742305469e-06 |

FP32替换仍不保证与CPU逐位相同，矩乘归约次序也会影响舍入；本例QK的FP32替换相对L2甚至略高。Q/K/V另有主权重提前Half舍入的对照、补偿矩乘和相加记录，不能仅凭这些数据推断驱动内部的累加格式。9个取样共243项同输入阶段比较；没有认定硬件或驱动违反精度契约。

## 已知坐标的有符号重放

与S51相同，坐标(batch=0, token=9, feature=199)的NPU输出=-5.9375，CPU分阶段参照=-5.5625，差值=-0.375。按顺序逐段替换实际中间值，再用CPU参照继续O投影／残差及FFN，记录每次输出变化：

| 替换边界 | 该坐标的有符号输出变化 |
|---|---:|
| norm1 | +0.0000000 |
| qkv | -0.1875000 |
| rope-repeat | +0.0000000 |
| qk | +0.0000000 |
| scale | +0.0000000 |
| mask | +0.0000000 |
| softmax | -0.1875000 |
| pv | +0.0000000 |
| o-projection | +0.0000000 |
| residual-add | +0.0000000 |
| remaining-norm2-ffn | +0.0000000 |
| 合计 | -0.3750000 |

每次重放都使用同一层输入、权重和显式舍入边界；CPU余段只用于归因，不回填到NPU推理。11项差值精确重建已观测的-0.375。该分解依赖替换顺序及所选坐标，不能推广为全模型“各占一半”的误差比例；0项也不表示该算子处处无误差。

## 真实NPU替换对照

四个固定替换均调用已有intel_npu设备算子：QK FP32、softmax FP32、PV FP32，以及三者组合。QK/PV各保持原Half输出边界；softmax用Half masked scores→FP32 softmax→Half概率。原权重／残差和后续FFN不变，所有替换算术在NPU；CPU只有常量、参考与读回检查。

| 替换 | 原坐标相对CPU输出差值 | 整层相对L2 | 整层scaled max |
|---|---:|---:|---:|
| baseline | -0.37500 | 0.000651849 | 0.013540070 |
| qk-f32 | -0.37500 | 0.000651869 | 0.013540070 |
| softmax-f32 | -0.18750 | 0.000639212 | 0.007103981 |
| pv-f32 | -0.37500 | 0.000651732 | 0.013540070 |
| all-f32 | -0.21875 | 0.000639855 | 0.007898374 |

单独FP32 softmax消除了该坐标一半的差值；QK或PV替换没有消除该峰值。组合替换在本例也没有优于单独softmax。以上是固定输入的替换实验，不是整网或生成质量提升结论。

## 交叉取样

下表均为baseline参数层在相应实际NPU前缀输入上的计算，与FP32 softmax替换比较。层号0起始。

| 前缀参数 | 轨迹 | 层 | 原→新相对L2 | 原→新scaled max |
|---|---:|---:|---:|---:|
| baseline | 4 | 14 | 0.000018264 → 0.000017834 | 0.001398987 → 0.001639622 |
| baseline | 4 | 29 | 0.000651805 → 0.000647469 | 0.007054416 → 0.007054416 |
| baseline | 19 | 14 | 0.000018574 → 0.000018139 | 0.001489872 → 0.001639622 |
| baseline | 19 | 29 | 0.000639015 → 0.000637479 | 0.008197622 → 0.008197622 |
| ffn | 4 | 14 | 0.000016236 → 0.000016096 | 0.001232224 → 0.001077009 |
| ffn | 4 | 29 | 0.000651849 → 0.000639212 | 0.013540070 → 0.007103981 |
| ffn | 19 | 14 | 0.000017312 → 0.000015300 | 0.001376127 → 0.001376127 |
| ffn | 19 | 29 | 0.000656132 → 0.000646750 | 0.013540070 → 0.007719408 |

FP32 softmax在8/8设置上降低L2；scaled max为3处下降、3处相同、2处升高。最坏取样scaled max从0.013540070降到0.008197622，但不能称为每个指标都改善，也不能把相关的8个设置当作8项独立模型能力评测。QK/PV替换分别3/8、6/8设置L2下降，scaled max均无改善；组合替换8/8 L2下降，但同样有2处scaled max升高。

## 完整30层内部原型

原型仅改变softmax路径：SHAVE norm1→DPU Q/K/V及masked scores／V前段→已有SHAVE FP32 softmax并转Half→DPU PV/O／残差尾段→原融合FFN。原W8代码、scale、rank32及其他模型数学路径保持。

首条内部轨迹30层逐项核对前段masked scores／V、原attention输出、原概率tail控制及FFN分段，共150项精确控制；30个原型逐层CPU局部检查也满足L2<0.003、scaled max<0.01。随后7条轨迹只做整网对照。模型embedding、末端norm/lm_head及参考指标仍是显式CPU边界。

固定内部轨迹4/9/14/19/24/29/34/39，共290个位置；候选还必须通过位置更多，且平均／最大L2、最大KL和next-token一致数均不变差，才可继续晋级。

| 路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| softmax-f32 | 242/290 | 2.040853% | 13.533334% | 0.004162213 | 286/290 |

平均L2略降，但通过数少1、最大L2与最大KL升高；冻结选择为baseline。这里的失败是预设不劣化晋级规则，不表示最大KL超过原0.02门槛。所有正式进程均完成，不存在本轮候选异常中止或设备错误。

## 代价与资源

| 路径 | 单次预热提交 | 单次诊断计时 | 新session | 显式host copy |
|---|---:|---:|---:|---:|
| baseline | 30 | 0.050314秒 | 0 | 0 |
| softmax-f32 | 240 | 1.918259秒 | 0 | 0 |

提交增至8倍，计时约1.92秒对0.050秒。各只测一次预热调用，包含拆图、类型转换、Python／驱动调度等，不是隔离负载后的性能基准，也不表示softmax硬件算术本身慢相同比例。重复输出逐项相等。默认1536图／512MiB预算不变，四个进程graph cache evictions均为0。

4个正式NPU进程，共**9,073次记录提交**，全部正常shutdown，context_failed=false，live buffers、缓存session/block和隔离字节归零。没有挂起、强制终止或驱动重置。

## 验证与限制

- 12项新离线测试覆盖图拓扑／边界、decompression元数据、tail依赖、独立标量attention参照、RoPE/GQA布局、Inf掩码、原晋级规则及已知拒绝／未知失败区分。
- 完整工作区305项检查通过（267+32+6）；公开副本292项检查，291通过、1项因缺少嵌套source-release副本按原规则跳过，该检查在完整工作区通过。
- 228份fixture C/ELF/汇编/manifest与父分支及公开副本逐字节相同；S51记录的10份源码／运行时／模型指纹保持原值。无生产后端、C/C++、SHAVE ELF、DLL/PYD或构建变化，无新增依赖、环境、模型下载、图表或持久图缓存。
- 本轮只有有限B1/S64前向取样及内部整网验证；原型未接入KV decode。候选未晋级，未做独立前向、PPL／生成认证，未扩展INT4/NF4或低位训练。
- 局部目标为CPU分阶段W8参照，整网目标为原CPU FP16模型；更接近前者不保证更接近后者。本轮不据局部峰值降低改写默认质量结论。
- 本机运行树含S46分类损失版本保护，它不参与此次冻结decoder；公开分支沿用S51父分支生产后端，实际运行指纹如下。
- 研究分支基于[PR #7](https://github.com/LCS-Official/torch-intel-npu/pull/7)，保留旧PR、main及v0.2.0 tag。

## 后续方向

优先研究以真实NPU Q/K/V投影输出为依据的校准，联合处理原权重量化残差与设备数值差异，并以原FP16模型整网输出作为最终目标。沿用当前softmax默认路径和同等参数／图预算，先验证是否有稳定整网收益，再考虑额外精度路径的融合与效率。此次两个-0.1875只定位了一个已知坐标，不能替代完整模型质量验证。

## 复现

需要父分支S51及已有模型／运行环境。prepare串行重建原权重、因子和scale拟合结果，拒绝覆盖已有准备；各NPU命令也须串行。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_attention
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_attention_quality.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_attention_quality.py -ScriptArguments @("focus")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_attention_quality.py -ScriptArguments @("cross")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_attention_experiment.py -ScriptArguments @("internal","--mode","baseline")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_attention_experiment.py -ScriptArguments @("internal","--mode","softmax-f32")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_attention_experiment.py -ScriptArguments @("select")
```

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_attention_quality.py | `d1ebe9b1acd4fd9e429bed98d0561a30a8018c29cfef6ac66ab9a967d99152a3` |
| npunlock-probe/research/w8_attention_experiment.py | `8e4d16d9626a82666f24ea59ca25e646a51c4a8c0e69237c3ec997c4257db939` |
| test_torch_npu_w8_attention.py | `3d8471866c259440323d4ca976d26d53c05e7844a6c59e027d56945efd344c58` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_npu_lowbit.py | `1c579ad4f2de9528d3fbd407c30df52bcfd29ac11b63cc202ffacd9c9137b01d` |
| torch_npu_lowbit_correction.py | `585642159d23366afcff326b95db8189d4d960509e68e2ab7a6bb6f94d44aec5` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时scale-fit.npz | `bf83484623cf3ed6ad0d0712ed2e99def50c7533625d57e70ea802b12d34a0ba` |

| 临时进程记录 | 提交数 | SHA-256 |
|---|---:|---|
| focus | 761 | `1cd481c6fdb26ee970b77d7fc857444e6f4e46f43cef115e1a37b7011e6afc08` |
| cross | 5612 | `08edff4d5851adda14763dd056b98287d7410fcee30c068d55c6bae2628d81a5` |
| internal-baseline | 330 | `d8217fecb6da1e7228d0392186afc40578d6bfb3500afb5ac37a5c857e8073ff` |
| internal-softmax-f32 | 2370 | `b543e832d77831a9ab7a8d0314ef088cd47641992bf2a4430e314deadce7ada5` |

## 清理

报告审计后清理本轮原始NPZ、JSON和日志；保留实现、测试、运行组件及本报告。实际清理统计随后追加。

2026-10-29实际删除24份本轮临时NPZ、JSON和日志（含清理核对清单），共127830589字节（121.909MiB）。清理前后1,957份保留源码、内核、文档、公开副本与运行组件SHA-256全部一致，随后追加本段及存储台账。研究结果目录仅保留REPORT.md；原模型、环境和全局缓存保留，四个正式测试进程均已退出。
