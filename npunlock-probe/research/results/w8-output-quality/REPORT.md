# S48：量化前缀上的FFN down与整块输出校准

2026-10-27，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3；Torch2.14.1+cpu、Transformers5.18.0。使用已有本地SmolLM2-135M-Instruct，完整30层decoder。

本轮实现并比较8种校准候选，加入输出残差回归、固定低秩输入基的约束回归及其独立数学测试。内部最佳候选input-local只重新校准30个FFN down投影。它在NPU前向多通过1处，却使平均/最大误差和最大KL变差；KV通过数不变，NPU文本PPL恶化约0.01686%，8条自由生成与基线相同。因此**不替换原rank32实验方案，不改变生产默认或原始质量门槛**。

这轮交付的是可复现校准工具和完整的否定性评估，而不是低位质量已修复或语言能力提升的声明。前一轮单层FP16保护研究见[PR #3](https://github.com/LCS-Official/torch-intel-npu/pull/3)。

## 协议、输入与边界

- 原40条FP16 greedy轨迹、S44逐行W8 GPTQ权重及rank32基线重新准备；W8和基线因子NPZ与S44字节相同。固定32条/2048行拟合，另8条/290个位置选候选。所有新拟合均只访问32条拟合轨迹。
- 实际量化前缀输入来自CPU精确W8参考：INT8 code乘FP16 scale不预先舍入，FP64主矩乘后转Half；补偿使用原FP32累加、Half中间结果。它不是从真实NPU读取的校准激活。NPU仅用于冻结候选后的独立验收。
- Teacher为未修改的官方FP16模型；手动分层回放先与官方B8/S64全模型输出逐位核对。拟合按B8分组推进量化前缀；内部选择和原长度独立CPU评测为B1。
- 六个非down投影、全部INT8 codes/scales、rank32和因子形状保持原值。每种候选只替换30个down投影的因子；投影参数字节净增加0。没有新增推理算子、原生接口或训练路径。
- 初始4个候选只在内部评测后发现整块拟合退化，才增加固定输入基的4个候选；该适应性决定发生在独立阶段之前，并保留首轮标量记录。最终按失败数、最大raw L2、平均raw L2排名，同分优先基线，再按标签。
- 原17例/142位置与S45的16例/128位置均为已有历史独立集合，不作为新盲测。选定input-local后不依据CPU、NPU、文本或生成结果重选。
- 每位置标准始终为raw logits relative L2<0.03且KL(reference || actual)<0.02；去均值误差和token一致性不替代原门槛。

## 校准目标与约束

令X为量化前缀下的门控后FFN输入，R1为本层attention之后的学生残差，HT为teacher本层输出，Wq为不预舍入的W8权重，E=W-Wq：

```text
Ylocal = X E.T
Yblock = HT - R1 - FP16(X Wq.T)
Yalpha = (1-alpha) Ylocal + alpha Yblock
```

input-local复用原fit_lowrank_correction，仅将激活换为该候选逐层产生的量化输入。block族alpha固定0.25/0.50/1.00，求rank32的C，使||X C.T-Yalpha||²/m + lambda||C-E||²最小；lambda为1%平均输入能量。用Cholesky白化和FP64 SVD求解，撤销白化后重新平衡两组因子，再存为FP16。该平衡处理已用零激活/原权重误差先验测试核对。

basis族固定原A0，令Z=FP16(linear_FP32(X,A0))，只拟合输出因子B：||Z B.T-Yalpha||²/m + lambda||B-B0||²；lambda为1%平均Z能量。测试local及alpha=0.25/0.50/1.00。A0字节严格不变，避免重新选择1536维输入基；这组约束仍未带来内部整网改善。两类实现均自行编写，沿用S44的LowRankCorrection接口和已有来源记录。

## 内部筛选（B1，290位置）

| 候选 | 拟合集末层hidden L2 | 内部通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|---:|
| baseline | — | 246/290 | 2.0030% | 13.0918% | 0.0038297 | 286 |
| input-local | 0.7330% | 247/290 | 2.0281% | 13.3565% | 0.0038133 | 286 |
| block-0.25 | 0.5070% | 184/290 | 3.1784% | 17.9957% | 0.0340314 | 283 |
| block-0.50 | 0.3740% | 183/290 | 3.2343% | 23.1131% | 0.0339331 | 283 |
| block-1.00 | 0.3114% | 221/290 | 2.4001% | 16.3527% | 0.0107582 | 285 |
| basis-local | 0.7331% | 244/290 | 2.0391% | 12.3865% | 0.0037346 | 285 |
| basis-0.25 | 0.7132% | 236/290 | 2.1797% | 14.7168% | 0.0044687 | 286 |
| basis-0.50 | 0.6932% | 219/290 | 2.5344% | 16.4875% | 0.0071066 | 286 |
| basis-1.00 | 0.6151% | 186/290 | 3.0539% | 18.3143% | 0.0087660 | 287 |

拟合hidden误差是32条轨迹合并后的向量范数比；内部logits指标是8条未拟合轨迹的逐位置统计，两种量不可直接相减。整块方案在拟合激活上误差较小，却在内部整网退化，表现为过拟合；固定输入基也未解决。input-local仅多通过1处，而且内部平均和最大误差略升，选择它只意味着按预定排序进入独立检验。FP16包装控制290/290通过且误差逐位为0。

## 独立CPU完整模型

| 集合 | 方案 | 通过 | 平均raw L2 | 最大L2 | 最大KL | 相同token |
|---|---|---:|---:|---:|---:|---:|
| regression | baseline | 126/142 | 1.7505% | 10.1581% | 0.0212464 | 141 |
| fresh | baseline | 111/128 | 1.6415% | 6.0134% | 0.0072343 | 128 |
| regression | candidate | 126/142 | 1.7196% | 9.9508% | 0.0213116 | 141 |
| fresh | candidate | 112/128 | 1.6077% | 5.9964% | 0.0071497 | 128 |
| regression | fp16 | 142/142 | 0.0000% | 0.0000% | 0.0000000 | 142 |
| fresh | fp16 | 128/128 | 0.0000% | 0.0000% | 0.0000000 | 128 |

CPU下平均误差略降，历史回归通过数不变，新集合多通过1处；最大KL仍超过0.02。FP16控制270/270与未包装模型逐位一致。

## 真实NPU前向与KV

前向使用64-token右侧补零因果计算，只计原有效位置；KV使用bird/cat/fox/repeat32共98个位置，one-token prefill后分组decode。CPU embedding、末端norm/lm_head是显式适配器边界，全部30层decoder与KV在NPU执行，没有CPU计算回退。前向和KV分别报告。

| 路线/集合 | 方案 | 通过 | 平均raw L2 | 最大L2 | 最大KL | 相同token |
|---|---|---:|---:|---:|---:|---:|
| forward/regression | baseline | 126/142 | 1.6581% | 9.7826% | 0.0198278 | 141 |
| forward/fresh | baseline | 113/128 | 1.6319% | 6.3928% | 0.0067631 | 127 |
| forward/regression | candidate | 127/142 | 1.7098% | 10.6301% | 0.0212879 | 141 |
| forward/fresh | candidate | 113/128 | 1.6338% | 6.6327% | 0.0062594 | 127 |
| forward/regression | fp16 | 142/142 | 0.3165% | 1.1532% | 0.0001050 | 142 |
| forward/fresh | fp16 | 128/128 | 0.3084% | 1.5434% | 0.0002107 | 128 |
| KV | baseline | 82/98 | 1.9752% | 10.1119% | 0.0198410 | 97 |
| KV | candidate | 82/98 | 1.8832% | 9.7444% | 0.0198467 | 97 |

候选KV修好cat/48和repeat32/52，又新增cat/46和repeat32/45两个失败，总通过数仍为82/98。原16个失败中仍有14个，另有2个新失败；这不构成质量缺口已修复。

三组forward首例的90个逐层同输入检查全部满足原staged CPU relative L2<0.003、scaled max<0.01；FP16整网控制270/270通过。两组KV每条轨迹重放末步，均30次调用且不新增session或host upload/download，输出逐位一致。

## 固定文本PPL与自由生成

沿用S44同一WikiText-2 raw test revision b08601e04326c79dfdd32d625aee71d232d685c3的8段固定切片；标准库取回临时文本并校验revision、行号、完整性和文本SHA，无数据依赖安装。每段257个token、7个64输入窗口/步长32，共56窗、2048个目标恰好计一次，FP64 log-softmax求和得到token加权NLL/PPL。不是完整WikiText测试集成绩。

| 平台 | 方案 | NLL/token | PPL |
|---|---|---:|---:|
| CPU | fp16 | 3.329926874 | 27.936298767 |
| CPU | baseline | 3.333513475 | 28.036675007 |
| CPU | candidate | 3.333468260 | 28.035407361 |
| NPU | baseline | 3.333532058 | 28.037196037 |
| NPU | candidate | 3.333700652 | 28.041923332 |

CPU候选PPL降低0.00452%，但NPU候选PPL提高0.01686%，方向相反，均幅度很小。本轮不推断统计显著性或模型能力提升。NPU基线28.037196037与S44复现一致。CPU侧的小幅收益不能当作设备侧收益。

自由生成沿用8条固定提示、greedy、最多32个新token及相同EOS。没有教师强制生成，以下比较完整生成token序列，不能代替任务准确率。

| 提示 | CPU候选=CPU基线 | NPU候选=NPU基线 | NPU候选=CPU FP16 |
|---|---|---|---|
| divide | 相同 | 相同 | 相同 |
| planet | 相同 | 相同 | 相同 |
| format | 相同 | 相同 | 相同 |
| square | 相同 | 相同 | 相同 |
| paperclip | 相同 | 相同 | 相同 |
| shadow | 相同 | 相同 | 不同 |
| spanish | 相同 | 相同 | 相同 |
| clock | 相同 | 相同 | 不同 |

两组NPU文本各对首窗全部30层做staged CPU数值检查。每条生成另重放最后一步并核对91个输出/缓存数组以及最初90个缓存数组未被修改；16组重放均通过，且不新增session或host upload/download。

## 数值对齐与后续方向

新候选CPU平均误差/PPL略优，NPU却出现前向误差/PPL退化，说明当前校准收益对参考实现与真实设备的数值路径敏感。局部staged检查通过只证明误差处于原容差内，不能保证这种微小校准收益会跨30层保留，也没有证明驱动或某个内核存在错误。

只读统计中，down因子的subnormal数量为基线2156/1474560、候选2184/1474560；up为10111/552960和10086/552960。它们近似，不能据此把退化归因于FTZ或因子动态范围。下一轮宜先逐阶段对齐实际NPU的down GEMM、两次补偿GEMM及残差相加，再决定是否重平衡因子或使用设备实测激活校准；不继续仅靠扩大系数搜索。

## 验证与修改范围

- 7个最终真机进程共44,130次NPU提交，全部正常shutdown；context_failed=false，live buffers、cached blocks/sessions和隔离字节归零。默认1536图/512MiB不变，7组均无图淘汰、设备错误或超时；未新增承压生命周期或其他驱动认证。
- 完整工作区265项离线检查通过：根目录227（含9项新数学/路由测试）、Python API32、发布工具6。独立公开分支252项检查：251通过，1项因没有嵌套source-release副本而按原规则跳过；该对照在完整工作区已通过。
- 独立标量审计复算4426份质量位置记录的原始门槛/通过数/均值/极值/token计数，并验证5组文本的280份窗口记录、目标计数与NLL/PPL；首轮4候选重跑统计一致。
- 228份既有fixture ELF/汇编/C源/manifest与公开main字节相同；原session DLL/PYD保持S45指纹。没有生产后端或图算法修改，没有新依赖、环境、模型下载或内核/宿主重建。
- 本机运行树含独立S46损失版本修复和S47研究文件，它们不参与本次冻结推理。公开研究分支基于main，独立于PR #2/#3；不自动合并或改写v0.2.0发布tag和历史SOURCE_MANIFEST。
- 未新增INT4/NF4整网、QLoRA/低位训练、任意模型迁移、长期收敛、TTFT/吞吐或能效认证。所有推理仍受原SmolLM固定模型、shape及上下文限制。

## 复现

在已有环境激活后运行；需要已有SmolLM2-135M-Instruct及requirements-models.txt。prepare拒绝覆盖已有权重；原始数据清理后需重新准备。所有NPU命令串行；data阶段需要联网取固定revision测试文本。没有新pip包或原生构建要求。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_output
foreach ($stage in @("prepare","fit","select","independent")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_output_quality.py -ScriptArguments @($stage)
}
foreach ($variant in @("baseline","candidate","fp16")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_output_quality.py -ScriptArguments @("hardware","--variant",$variant)
}
foreach ($variant in @("baseline","candidate")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_output_quality.py -ScriptArguments @("hardware","--variant",$variant,"--route","kv")
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_output_language.py -ScriptArguments @("data")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_output_language.py -ScriptArguments @("cpu")
foreach ($variant in @("baseline","candidate")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_output_language.py -ScriptArguments @("hardware","--variant",$variant)
}
```

研究代码：[校准/整网](../../w8_output_quality.py)、[文本评测](../../w8_output_language.py)、[离线测试](../../../../test_torch_npu_w8_output.py)。

## 源码与运行身份

| 文件/产物 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_output_quality.py | `080faae3356189bce7530fc0f8e995b966b017c60da0c4df5ad202946ead2894` |
| npunlock-probe/research/w8_output_language.py | `33e5258361dcbcb5c6b625f1be5661999e6387cf2a1a90eb9e393b4ff0c88993` |
| test_torch_npu_w8_output.py | `a045b3d7799f6c194cf6328f32a9f12216a2d04242f72c58fbd6663d289f9ac3` |
| torch_npu_lowbit_correction.py | `585642159d23366afcff326b95db8189d4d960509e68e2ab7a6bb6f94d44aec5` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时 weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时 corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时 input-local.npz | `3b0bb375accc574145cac82b85bf502b0b127bd283435e010f9f4ba4bd2da449` |
| 临时 output-fit.json | `20de222eb7cd8d5a2dbd299a56941f0a58666a478d1d9028fa2d3da361891df6` |
| 临时 selection.json | `d6cd90aa146b7358b152333fc89f9a04e17674acc74f811ae523203ae18aa5d6` |
| 临时 independent.json | `bd1d63717be0e3cd38560dddc0c4c1f42ee9f3ca765b861df2ede8fa117acac1` |
| 临时 corpus.json | `4edda7621268f1274a1f6144146fbec13de0e330860c323c88e5b4fa84458f43` |
| 临时 language-data.json | `e82464f036933db7df375e88cf3a413d3a0e618735e3a6e219561da695bbc688` |
| 临时 language-cpu.json | `cbb6424c2eacb3969614257a108b897b1268c7de6daae7216a010a6df0e54716` |
| 临时 language-npu-baseline.json | `8dcb097e6200d42ce5d322d03fbeb5b0ea34d076ae7ba0234a39e08d41324ed6` |
| 临时 language-npu-candidate.json | `14f426eadaf75d5fc28ad085a44a8e046e0b992ca14e785f68ac4dc8e10b7935` |

## 交付与清理

保留两份复现源码、9项新增离线测试与本报告；研究分支保存历史。原始权重、因子、文本、token、JSON和日志在报告审计后清理，统计随后追加。

本轮删除47 份临时NPZ、JSON和日志，共147829860 字节（140.982MiB）。清理前后核对1931份源码、内核、文档、公开副本与运行组件SHA-256，全部一致，随后补充本清理记录及空间台账。研究目录仅保留REPORT.md；原模型、环境和全局缓存保留，所有本轮测试进程已退出。
