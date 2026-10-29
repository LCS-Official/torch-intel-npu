# S55：高误差位置约束、整批校准与新提示泛化诊断

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；已有本地SmolLM2-135M-Instruct。

本轮新增最高误差位置与逐位置退化惩罚，将整网拟合校准提示32→48，并冻结另一组8条新提示。最终扩展至全部210个既有rank32输出矩阵，采用整批梯度与实际校准分数回溯。两次校准目标下降1.64%／9.26%，第二次最大校准L2下降约30%、最大KL下降约12%；这些收益未泛化到NPU验收。

NPU内部通过仍243/290，上轮回归集246→245/262，新提示245→242/266；最大KL均变差。第一次更新还触发原末层局部守卫，第二次可完整执行但未通过晋级。保留默认baseline，不启用候选。新提示shadow/62的KL在CPU／NPU基线均超过原0.02门槛，显示量化／校准泛化问题，不能单归因到驱动。

## 数据与固定边界

- 原32条拟合轨迹仍只重建原W8编码、FP16 scale及全部rank32因子；两份基线NPZ与历史指纹逐字节相同。编码、scale、rank和输入因子没有改动。
- 整网拟合使用原32条加16条新校准提示，共48条、1,657个后续位置，B1/S64。先验证原32的indices及新增40..55，再按“两条原有、一条新增”交错；与原内部8条严格隔离。
- 原内部8条／290位置仅用于冻结候选。S54已经看过的8条提示明确作为regression／262位置，另8条新提示作为fresh／266位置。新增校准、回归、新提示之间做文本和tag检查，与原40条、HOLDOUT/FRESH及S45提示做文本去重。
- 三组提示及原FP16 greedy token轨迹在首次拟合前冻结。CPU方案调整只查看48条校准结果；所有NPU组和CPU泛化对照都在最终拟合后进行。候选冻结后未按回归／新提示重排、重训或修改选择。
- 新提示这轮已经被查看，后续不能继续充当未查看验收集。262／266是相关的teacher-forced位置，不是同等数量的独立文本。

## 目标与优化方法

令每个位置的`e`为原始logits相对L2的平方，`k=KL(teacher||student)`。每条提示取`ceil(10%*位置数)`个最大e及最大k（分别排序，至少1个位置）。固定的budget是原W8＋rank32 CPU参考在相同校准位置的(e,k)，拟合中detach，不更新。

```text
mean(e + 0.1*k + 2*relu(e-0.03^2))
+ 0.5*mean(top10%(e)) + 0.5*mean(top10%(k))
+ 2*mean(relu(e-e_baseline)) + 2*mean(relu(k-k_baseline))
```

最后加1e-5相对输出因子Frobenius偏移作为梯度正则；表中校准分数为以上主目标，不含该微小正则。实现使用`maximum(excess,0)`，在budget恰相等时取对称子梯度，前向数值与ReLU一致。训练惩罚没有修改原L2／KL验收门槛，没有中心化logits。

先在CPU尝试仅Q/K/V输出矩阵的逐条AdamW更新，两轮结果：

| CPU开发方案 | 初始目标 | 第一次 | 第二次 |
|---|---:|---:|---:|
| 原顺序，lr=1e-6 | 0.001238042029 | 0.001339440901 | 0.001278205792 |
| 交错，lr=2e-7 | 0.001238042029 | 0.001326211556 | 0.001321702478 |

两种方案综合目标均退化，未做NPU筛选或新提示验收。这两种开发方案使用ReLU默认边界梯度，最终整批版本改为对称子梯度；同时改变了可调参数范围与更新方式，不能把收益单独归因于某一项。随后在首次真机评价前，将可调参数从Q/K/V的90个输出矩阵扩至全部210个既有输出矩阵（4,976,640个参数），继续冻结全部INT8编码／scale、rank32输入基和其他权重。这里只增加CPU准备时可调自由度；NPU参数数量、字节数和推理算子拓扑不变。

最终方案固定两轮整批梯度：每轮48条的梯度平均，loss scale512，FP32 master梯度还原后norm clip1；前向每次都用Half输出因子。沿负梯度尝试相对原输出因子Frobenius范数的0.001／0.00025／0.0000625／0.000015625步幅；只接受严格更低的实际Half校准分数，否则恢复全部master。两轮均在第一档接受，记录完整proposal。

| 参数状态 | 校准主目标 | 平均L2 | 最大L2 | 最大KL |
|---|---:|---:|---:|---:|
| baseline | 0.001238042029 | 1.196062% | 6.485161% | 0.004848611 |
| tail-e1 | 0.001217754960 | 1.179528% | 5.657619% | 0.004320973 |
| tail-e2 | 0.001123347702 | 1.149589% | 4.551994% | 0.004264116 |

最终96次反向、两次整批更新耗时约92.678秒，另有两种被淘汰的CPU开发方案；均为本机准备过程记录，不是性能基准。teacher／student激活、logits及校准budget只在内存使用。CPU可微参考的下降不是NPU质量保证。

## NPU内部筛选与数值守卫

原局部门槛仍为同输入分阶段CPU参照L2<0.003、scaled max<0.01。tail-e1在第30层scaled max=0.01012098416686058、relative L2=0.0006864631432108581，按原规则拒绝，未继续计算该候选的整网通过率。设备健康关闭，不是驱动失败；没有放宽到0.0102或跳过此坐标。

原整网位置门槛仍为logits L2<0.03及KL<0.02，原晋级规则还要求通过数更多且平均／最大L2、最大KL、token一致数均不变差。

| 内部路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| tail-e2 | 243/290 | 2.028494% | 13.225685% | 0.004538313 | 285/290 |

唯一完整的新候选tail-e2被冻为诊断candidate，selected仍baseline。其通过数没有增加，最大KL及token一致数变差，未晋级；随后回归／新提示仅作成对诊断，不重选参数。

## NPU回归与新独立提示

| 集合／路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| regression/baseline | 246/262 | 1.500357% | 4.837565% | 0.004043723 | 257/262 |
| regression/candidate | 245/262 | 1.485053% | 5.100930% | 0.004789775 | 257/262 |
| fresh/baseline | 245/266 | 1.602766% | 8.233340% | 0.034045011 | 263/266 |
| fresh/candidate | 242/266 | 1.614233% | 7.966191% | 0.035698410 | 263/266 |

S54回归baseline的246/262及所有摘要值原样复现。候选在该组平均L2略降，但通过数少1、最大L2／KL升高；新组通过数少3、平均L2和最大KL升高，最大L2略降，token一致数不变。校准集上的尾部收益没有成为稳定泛化收益。

固定64-token数值协议中，部分其他提示含首个EOS之后的teacher-forced位置，不能当作正常自由生成；shadow轨迹在64-token输入内没有生成EOS，以下峰值不属于EOS后续段。

新集合baseline在shadow/62的KL=0.0340450108，candidate=0.0356984101，均高于原0.02；该位置L2也超限。新组baseline有21个L2超限、2个KL超限；candidate有24个L2超限、2个KL超限，KL超限位置包含在L2超限中。没有把“推理进程完成”当成质量通过。

## CPU整网参考对照：区分泛化与设备差异

候选冻结后，以相同token、原FP16 teacher及相同参数，在CPU W8＋rank32参考上另做六组对照。仅统计指标，不回填NPU结果；assert未导入torch_intel_npu。

| CPU路径／集合 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL |
|---|---:|---:|---:|---:|
| baseline/internal | 246/290 | 2.002960% | 13.091786% | 0.003829703 |
| baseline/regression | 246/262 | 1.510489% | 5.538071% | 0.004023504 |
| baseline/fresh | 246/266 | 1.572197% | 7.363860% | 0.033708718 |
| candidate/internal | 245/290 | 2.029757% | 12.961422% | 0.003840984 |
| candidate/regression | 243/262 | 1.515736% | 5.403499% | 0.005027992 |
| candidate/fresh | 248/266 | 1.602078% | 7.714130% | 0.032097068 |

CPU参考在内部／回归组也出现通过数和部分指标退化，说明问题不只存在于NPU路径。新提示shadow/62的CPU基线KL=0.0337087177，候选=0.0320970677，两者均超0.02。至少存在模型／校准泛化缺口，不能把全部差异归为NPU驱动错误。

另一方面，新组CPU候选通过数246→248，NPU却245→242，KL的改善方向也相反，说明CPU目标无法替代真实设备验收。本轮没有将这两类问题唯一归因至某个算子，也没有证明某个方案已修复。

## 资源、回归与范围

7个正式NPU进程共**2,070次提交**：6个完整通过执行与资源检查，1个按原局部精度规则拒绝。全部7个均健康shutdown，context_failed=false，live buffers、缓存session/block及隔离字节归零，没有挂起、强制终止或驱动重置。7×30=210项局部检查，其中209通过、1项拒绝，不写为210项全部通过。

六个完整进程的重复整网调用仍30次提交，无新session或显式host拷贝，输出逐项一致。默认1536图／512MiB预算不变，七个进程淘汰均为0；完整进程Tensor峰值4,787,712字节，拒绝进程4,713,984字节，不包含全部模型／驱动内存。本轮没有独立吞吐或功耗基准。

- 14项新增离线检查覆盖L2／KL方向、top10%取整、逐位置budget、对称子梯度、teacher／budget detach、非法输入、210输出因子冻结边界、校准交错顺序及三组文本隔离／旧模块目录副作用恢复。
- 完整工作区340项检查通过（302+32+6）；公开副本327项检查，326通过、1项原有嵌套source-release检查跳过，该项在完整工作区通过。
- CPU泛化审计另完成6组对照，检查全部210个输入因子不变、选择／源码身份不变、数据与位置数完整、logits有限且无NPU调用。
- 228份fixture C/ELF/汇编/manifest与父分支、root、公开副本逐字节相同；S54报告所列7份源码／运行时／模型身份不变。未改生产后端、C/C++、SHAVE ELF、DLL/PYD、依赖或环境，未下载新模型。
- root仍有S46分类损失版本保护，它不参与冻结decoder；公开分支沿用S54生产后端。研究基于[PR #10](https://github.com/LCS-Official/torch-intel-npu/pull/10)。
- 仅覆盖同一个135M模型、B1/S64和W8＋rank32的teacher-forced前向。未新增INT4/NF4、KV decode、PPL／自由生成、长训练或其他模型／平台认证。

## 后续优先级

优先定位shadow后段的高KL分布差异，扩大校准文本覆盖，并让真实NPU整网指标参与校准候选检查。CPU参考已经显示泛化退化，设备指标又可能反转，不能只凭进一步降低CPU损失启用默认。继续调整时需另留未查看的新独立集。本轮保留完整工具与反例，不推广为已解决低位整网质量。

## 复现

使用已有本地模型和项目环境。prepare拒绝覆盖已有数据；硬件命令串行，tail-e1是本机已记录的预期精度拒绝，复现时保留失败记录后继续其余命令。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_tail
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_tail_quality.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_tail_quality.py -ScriptArguments @("fit")
foreach ($profile in @("baseline","tail-e1","tail-e2")) {
    try {
        .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_tail_quality.py -ScriptArguments @("internal","--profile",$profile)
    } catch {
        $r = Get-Content -Raw "npunlock-probe/research/results/w8-tail-quality/internal-$profile.json" | ConvertFrom-Json
        if ($profile -ne "tail-e1" -or -not $r.healthy -or -not $r.local_guard_failure -or -not ($r.local_guard_failure.relative_l2 -ge 0.003 -or $r.local_guard_failure.max_scaled_error -ge 0.01)) { throw }
    }
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_tail_quality.py -ScriptArguments @("select")
foreach ($group in @("regression","fresh")) {
    foreach ($label in @("baseline","candidate")) {
        .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_tail_quality.py -ScriptArguments @("evaluate","--group",$group,"--label",$label)
    }
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_tail_audit.py
```

三组数据在首次CPU试验前冻结，来源指纹对应当时源码；之后只调整拟合方法，没有修改提示／token。最终源码重跑会记录最终来源，不声明全部JSON／NPZ跨运行逐字节相同。原始数据清理后需重建准备与拟合。

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_tail_quality.py | `0d4a61bea978c16b6e630e3504d8a5d9dc6eb78d936cf93f25f397922a487b61` |
| npunlock-probe/research/w8_tail_audit.py | `98a05e20d4c1650bef21315d5d130279d08665fb7d055e2cc8fc8bb130eeac58` |
| test_torch_npu_w8_tail.py | `ab87878ab83f221738df2a6f6334d4b11803cf78c6b543be048f7aa58942c80c` |
| npunlock-probe/research/w8_logits_quality.py | `a865d2805c0aa87e1e29c6f534c1a51d76023f9b992e5c02fa48171e43bb7d25` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时logits-fit.npz | `73e5fbde2b5afe651ffa84e55b3336a23438d94c325cbd384acc724f619e6863` |
| 临时tail-data.json | `a85f2ad0afeba1e2e24e2fbedaff64dea03a96ec42763a093cc3787fe1e0dddd` |
| 临时regression.json | `66f67a1c2e899ce555f0b95df9a87bdac6d31b1b5c438a0497daf4eb4fbfcf6c` |
| 临时fresh.json | `1c9338e179cb99116157db5ae9e1b7dc95052693ddfd5b5dd7db7551fda676ec` |
| 临时logits-fit.json | `c34e6de0a94b2f9960fb4add44c8e174eb4ccb28fce45751d5e6928f65992ed2` |
| 临时selection.json | `b563ecf079a5c3ea9b48be730f64dc57b90301814a381455c4d478287d5cfa40` |
| 临时cpu-generalization.json | `4310ee251bf851f18ccb9faf2f1b1ca8250509736199041827b7dab7ec92bf52` |

| 临时NPU记录 | 提交数 | SHA-256 |
|---|---:|---|
| internal-baseline | 330 | `e8474fb323402e05326451a28082bc80505b0616d727d0022e875d0fe97d6e7c` |
| internal-tail-e1 | 90 | `cd19dd0e750c58289ec6e0bc5361b09b9cb91d2b0760c1797fe0c42ddf71208f` |
| internal-tail-e2 | 330 | `34db3b69479e66d928b0eb3ea49175203fa0bc17518e2411ed70a156fa8100d9` |
| regression-baseline | 330 | `d9b61a2a626f46db2be53bc34c9cfdba1e1adda9d5a7a677243ebd8ce8ab3b57` |
| regression-candidate | 330 | `814f6e7be5f866c02cfc4b0170b818b829b2a8c79177feb40a8a8e9109ca08bd` |
| fresh-baseline | 330 | `db36af697258fbfb00d841c4ab60394c7a31f059e299855ea61663b3a20f96f3` |
| fresh-candidate | 330 | `411cdbdbd60a8e237b330a13b00e93a01386f483906a1f4d99c509092ed43c7a` |

## 清理

2026-10-29末次删除本轮结果目录45份临时NPZ、JSON和日志（含两种淘汰的CPU试验、身份／审计／清理核对清单），共142559141字节（135.955MiB）。清理前后1,977份保留源码、内核、文档、公开副本及运行组件SHA-256全部一致，随后补充本段与存储台账。结果目录仅保留REPORT.md；模型、环境和全局缓存保留，7个正式NPU进程均已退出。
