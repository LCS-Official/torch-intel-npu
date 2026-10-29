# S54：整网logits／KL蒸馏校准与新样本验证

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；使用已有本地SmolLM2-135M-Instruct。

本轮从S53的局部投影SSE转向原FP16模型的完整logits／KL目标。离线CPU只拟合现有Q/K/V rank32输出因子；实际30层decoder的质量在NPU上验收。第二轮候选内部通过243→244/290，8条新提示通过246→248/262，但最大KL升高，新样本最大L2也升高。候选保留为研究结果，未通过原晋级规则，不替换生产默认。

## 固定数据与参数边界

- 重建原W8编码／FP16 scale和rank32基线，weights.npz、corrections.npz与历史指纹逐字节相同；32条原校准轨迹用于拟合，原8条内部轨迹仅用于冻结候选。
- 拟合前另外冻结8条新提示及原CPU FP16 greedy轨迹：sand、compass、parcel、moon、library、pattern、circuit、moss。全部B1/S64，按各自prefix末位置起计，共262个teacher-forced位置；与原40条提示、HOLDOUT/FRESH及S45新提示做文本去重检查。8条提示不是262个互相独立的样本。
- 新提示的原始logits不进入拟合，内部筛选后才首次进行新样本NPU比较；看过新样本后没有重新拟合、重排epoch或修改选择。它们在后续轮次已属于被查看集合，不应继续充当未查看验收集。
- 仅90个Q/K/V矩阵、921,600个既有输出因子参数参与CPU优化；INT8编码、scale、rank32输入基、其他四类投影、Embedding／norm／lm_head权重及全部推理结构固定。原有210个补偿仍使用rank32。
- 推理权重保持FP16输出因子。拟合使用外部FP32 master，**每次前向先转Half再参与原两次Half边界**；没有把FP32 master直接当作NPU推理参数。CPU转换的梯度传回master，优化器只持有90个master。

## 整网目标与校准过程

对原提示后续位置，以原CPU FP16完整模型logits为teacher；student是已有W8＋rank32 CPU可微参考。没有中心化logits，也没有改变原验收门槛。目标为：

```text
mean(relative_logits_L2_squared + 0.1 * KL(teacher || student)
     + 2 * relu(relative_logits_L2_squared - 0.03^2))
+ 1e-5 * ||B-B_prior||_F^2 / ||B_prior||_F^2
```

固定两个epoch、每轮32条轨迹单条更新，AdamW无weight decay；损失固定缩放512，FP32 master梯度还原后裁剪norm至1。优化只发生在CPU准备进程，NPU负责真实前向验收。原模型／低位参考保留Half前向，梯度是此参考的梯度，不能据CPU训练目标下降推断NPU必然改善。

初始lr=1e-4的开发试验仅看校准目标，两个epoch分别退化至0.002927099350和0.001511068787，均高于原0.000189254005，未进行NPU或独立样本验收。随后仅依据校准结果将lr调小至2e-6，仍固定两个epoch；没有试探内部／新样本后再调参。

| 固定参数状态 | 同一32条校准轨迹的平均目标 | 相对初始下降 |
|---|---:|---:|
| 原baseline | 0.000189254005 | — |
| logits-e1 | 0.000182434441 | 3.603392% |
| logits-e2 | 0.000172657976 | 8.769182% |

最终64次更新耗时约45.768秒；高步长开发试验另64次约45.514秒。这些是本机单次CPU准备记录，未作为性能基准。目标值为32条轨迹逐条平均的复合损失，不是原logits L2通过率。所有teacher／student激活和logits只在内存使用。

## 原内部集合：冻结候选

完整30层、原8条内部轨迹、290个位置，保持logits L2<0.03、KL(reference||actual)<0.02。各方案首条轨迹还按原同输入分阶段CPU参考做30项局部检查，L2<0.003且scaled max<0.01。

| 路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| logits-e1 | 243/290 | 2.011227% | 13.533424% | 0.004562495 | 286/290 |
| logits-e2 | 244/290 | 2.023981% | 13.289991% | 0.004263846 | 286/290 |

按原排序冻结新候选logits-e2；它修复5个原失败位置，同时新增4个失败位置，净增1。其最大KL由0.004049684升至0.004263846，未满足原“通过数更多且平均／最大L2、最大KL、token一致数均不变差”的晋级规则，所以selected仍为baseline。最大KL仍低于原绝对0.02门槛；本次拒绝的关键是相对基线不退化约束，未把它写成KL绝对越界。

## 新提示的独立NPU前向对照

candidate固定为logits-e2，即使原内部未晋级，仍作为诊断对象与baseline成对测量；新结果不重新选择参数，也不改变原selected。新集合首次查看发生于选择冻结后。

| 路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 246/262 | 1.500357% | 4.837565% | 0.004043723 | 257/262 |
| candidate | 248/262 | 1.499420% | 5.253115% | 0.004537756 | 257/262 |

修复library/36、circuit/32、circuit/44、moss/32四个原失败位置，新增compass/44、moon/51两个失败位置，净增2。平均L2仅小幅降低，最大L2由4.837565%升至5.253115%，最大KL也升高；token一致数未变。这是有限teacher-forced前向的混合结果，没有证明每个提示／位置都改善，也不构成PPL、自由生成或长上下文认证。

## 实际执行与验证

五个正式NPU进程均完成，全部健康shutdown，context_failed=false；live buffers、缓存session/block及隔离字节归零。每个进程330次提交，共**1,650次提交**；5×30=**150项原局部检查**通过，没有精度保护拒绝、驱动重置或挂起。各进程图缓存淘汰均为0，默认1536图／512MiB预算不变。

所有内部配置及新样本的重复整网调用仍30次提交，无新session、无显式host上传／下载，重复输出逐项相同。参数数量／字节数及执行图拓扑不变；没有新增推理算子或独立性能基准。Embedding及末端norm/lm_head／参考指标继续采用已有显式CPU边界。

- 11项新增离线测试覆盖复合损失与teacher KL方向、原始logits整体偏移、target detach、输入／非有限检查、Half前向与master梯度、更新后重新绑定、冻结参数／输入基、溢出保护和执行器恢复。
- 读取旧提示定义后恢复自有实验输出目录；测试模拟遗留模块导入修改全局OUT，避免实验数据写入旧结果目录。
- 完整工作区326项检查通过（288+32+6）；公开副本313项检查，312通过、1项原有嵌套source-release检查按规则跳过，该项在完整工作区通过。
- 228份fixture C/ELF/汇编/manifest与父分支、工作区、公开副本逐字节相同；S53报告所列9份现存源码／运行时／模型指纹不变。无生产后端、C/C++、SHAVE ELF、DLL/PYD、依赖、环境或模型下载变化。
- 本轮只研究同一个135M模型、B1/S64及既有W8＋rank32格式。未新增INT4/NF4、KV decode、PPL／生成、长时间低位训练或通用NPU训练后端认证。
- root运行树含S46分类损失版本保护，不参与此次冻结decoder；公开分支沿用S53生产后端。研究基于[PR #9](https://github.com/LCS-Official/torch-intel-npu/pull/9)。

## 下一步

整网目标在内部及新样本各净增1／2个通过位置，说明此方向有有限收益；当前平均目标仍不足以限制少数高误差位置。若继续，应在校准集扩大提示覆盖、加入高分位／高KL位置约束，并另备未查看的独立样本。当前证据支持继续研究，尚不足以替换默认或宣布低位整网质量已解决。

## 复现

已有本地模型和项目环境下，从仅有REPORT.md的结果目录重新生成；硬件命令串行：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_logits
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_logits_quality.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_logits_quality.py -ScriptArguments @("fit")
foreach ($profile in @("baseline","logits-e1","logits-e2")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_logits_quality.py -ScriptArguments @("internal","--profile",$profile)
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_logits_quality.py -ScriptArguments @("select")
foreach ($label in @("baseline","candidate")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_logits_quality.py -ScriptArguments @("fresh","--label",$label)
}
```

本轮新提示在高步长试验前冻结，fresh.json来源指纹属于当时源码；之后仅改步长，未改提示或token。最终源码重跑会在元数据中记录最终来源，因此不声明整份JSON／NPZ跨运行逐字节一致。表中指纹记录本次实际运行身份。

## 身份与临时记录

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_logits_quality.py | `a865d2805c0aa87e1e29c6f534c1a51d76023f9b992e5c02fa48171e43bb7d25` |
| test_torch_npu_w8_logits.py | `7fe701b5be3b2a9623940c3c965e357d49490949498388f1bec8c8135b0773d1` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时logits-fit.npz | `4c7b24ce7d22c1514cf669b8df8d61ac676612360138290bc99407128dbae5b7` |
| 临时fresh.json | `360b199ec2afc197f6ef35a6fa79f0387ce49094da3c719b92573fe6be667653` |
| 临时logits-fit.json | `1f099bf7da4279a8cf13d00473f779524129754adee4c499c93bc056dc0577f4` |
| 临时selection.json | `ae0076f2889ef3739bf371351292bd14b0d85c9ccf7c0c8c9154d078bdd5d614` |

| 临时进程记录 | 提交数 | SHA-256 |
|---|---:|---|
| internal-baseline | 330 | `9a3f70096fe61d601b18c20aee99fe39acfab367e6704bc361516c9ba8fde5d9` |
| internal-logits-e1 | 330 | `acb0acfc59ff432eca833b49d548fd7812edffb63f558575831c4f9ddfb102ce` |
| internal-logits-e2 | 330 | `99ebee6710c13c7310a845fcb3711c0da219926f5e828ec002c8bc82482543b9` |
| fresh-baseline | 330 | `76a565dab88d4c7ab4c98aeca2614ff44a01cc7dca2d0263a5a5d0275475705e` |
| fresh-candidate | 330 | `25630bd96c38189590e9f2dcc2b08f40c8eec79e12a9473e3c65f7351bc9ce14` |

## 清理

2026-10-29末次清理本轮结果目录下34份临时NPZ、JSON和日志（含核对清单及被淘汰的高步长试验），共123103478字节（117.401MiB）。清理前后1,969份保留源码、内核、文档、公开副本及运行组件SHA-256全部一致，随后补充本段与存储台账。结果目录仅保留REPORT.md；模型、环境和全局缓存保留，五个正式NPU测试进程均已退出。
