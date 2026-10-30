# S58：FP16 解压拟合代理与末层 down 成对试点

2026-10-30，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；沿用本地SmolLM2-135M-Instruct与W8＋rank32研究配置。

本轮按[S57](../w8-final-layer-audit/REPORT.md)的诊断增加显式FP16权重解压代理。在相同实际NPU输入上，新代理主投影relative L2比精确权重代理低**59.17%**，基线局部校准SSE预测偏差由**22.18%降至0.60%**；但逐项一致率下降、scaled max上升，仍不是DPU的逐位模拟。

只拟合末层down已有输出因子的两个固定候选，真实NPU局部校准SSE分别下降4.9790%／4.9321%。两者内部通过仍243/290，新代理最大KL变差；原规则保留baseline。内部排名更好的原代理候选在8条新提示上234→233/260，未产生独立质量收益。**生产默认、原参考与原精度门槛不变，低位整网质量仍未达标。**

## 固定范围与数据顺序

- 使用原基线W8编码、FP16 scale及rank32因子；两份基线NPZ与历史SHA逐字节相同。本轮没有使用S56已经更新全部210个输出因子的tail-e2权重，也没有宣称修复S57旧候选的zipper拒绝。
- 原32条加原16条新增校准文本，共48条固定轨迹。实际采集每条完整64-token的末层down输入，共**3,072行、1,769,472个投影输出元素**，包括提示前缀；不是S55整网目标中的1,657个后缀位置，更不是同等数量的独立文本。
- 校准只读取上述48条。原内部8条／290位置用于候选选择；另8条新提示／260位置在首次NPU观测和拟合前冻结，与原40条、历史校准／内部／独立提示做文本去重。
- 新提示为teapot、mirror、tiles、ticket、stapler、lighthouse、museum、seeds。首次内部选择完成后才查看这些新提示的NPU指标，之后没有改选、重训或更换提示。现在这些提示已被查看，后续需另留未查看文本。
- 两个候选在采集前固定：`exact-a50`与`decoded-a50`，相同原rank32输入基、相同ridge及0.5更新幅度。只允许第30层`mlp.down_proj.weight`的**576×32输出因子B**改变，共18,432个可调系数。其他209个因子及全部编码／scale逐字节相同，参数量、推理结构和已有30次预热提交不增加。

## 只改变拟合代理中的权重解压

令G为实际NPU gate/up/SiLU路径产生的down输入，Z为原输入因子A在NPU计算的reduced值；两者均由原图观测得到。两个主投影代理为：

```text
exact:   M = Half(G.double @ (INT8_codes * FP16_scale).double.T)
decoded: M = Half(G.double @ Half(INT8_codes * FP16_scale).double.T)
```

乘scale在精确FP64参照中重建，decoded随后显式转Half再转Double，模拟权重进入FP16 MatMul前的解压舍入；两种代理都保留FP64点积以隔离这一差别。**并未模拟DPU所有累加、调度或融合舍入**。`proxy_main`只用于研究拟合，不改`FrozenSmolLayer.reference_cpu`，不替换独立正确性参考。

对3072×576的实际原NPU主投影，同输入比较如下。relative L2的共同分母均为实际NPU输出；fixed-atol计数是原辅助诊断，不是更改后的模型通过门槛。

| 代理 | relative L2 | max abs | scaled max | 逐项一致比例 | 旧fixed-atol不一致数 |
|---|---:|---:|---:|---:|---:|
| exact-a50 | 0.0005642860 | 4 | 0.0009163014 | 86.660486% | 0 |
| decoded-a50 | 0.0002304186 | 1 | 0.0016559410 | 56.222081% | 855 |

新代理降低大数值误差，max abs由4降至1；但一致比例86.66%→56.22%，scaled max反而升高，辅助固定容差不一致数0→855。不能将平均L2改善称作所有位置更准确或硬件语义完全对齐。

## 同输入、同输入基的输出因子拟合

局部目标Y为原checkpoint FP16 down权重作用于**同一实际G**：FP32乘加后转Half。它不是原FP16模型在自己前缀上的完整logits目标，因此这是一处投影的重建试点，不能当成端到端蒸馏。

复用[S53](../w8-qkv-quality/REPORT.md)的设备输入基ridge拟合器。对每种M，使用相同Z、Y及原B0：

```text
H = Z.T @ Z / n
lambda = 0.01 * mean(diag(H))
B_fit = ((Y-M).T @ Z / n + lambda * B0) @ inv(H + lambda*I)
B_candidate = Half(B0 + 0.5*(B_fit-B0))
proxy SSE = || Half(M + Half(Z.float @ B_candidate.float.T)) - Y ||_F^2
```

求解为CPU FP64准备；只有Half候选的局部代理SSE严格改善才接受，否则恢复原B。两种固定候选都被接受。没有训练NPU反向、改变rank、扩大参数或按内部／新提示调学习率。

原NPU完整FFN观测与整层输出在三条校准轨迹逐项一致；采完48条后，保留的第一条原G／残差再进入原down尾图，与保存的原corrected输出逐项一致。随后两种候选复用相同NPU G／残差，运行各自真正的DPU down尾图，得到实际校准SSE。

| 候选 | 原因子代理SSE | 拟合后代理SSE | 拟合后实际NPU SSE | 实际SSE降低 | 改变系数数 |
|---|---:|---:|---:|---:|---:|
| exact-a50 | 2918.757640 | 2732.104338 | 3564.082254 | 4.9790% | 18408 |
| decoded-a50 | 3728.471527 | 3541.538786 | 3565.842224 | 4.9321% | 18410 |

实际NPU基线SSE为3750.838183766003。原代理对基线低估22.1839%，解压代理低估0.5963%；各自拟合后，代理对对应NPU实测SSE的预测偏差为23.3434%／0.6816%。因此解压代理对这一局部重建误差的预测更贴近设备。

不过，解压候选的实际SSE仍略高于原代理候选（3565.842224对3564.082254）。这两个SSE是局部训练／校准集上的投影误差，不能解释为整网loss降低，也没有用它们额外挑选新参数。模型选择继续使用原内部规则。

## 真实NPU内部筛选

局部守卫仍为同输入relative L2<0.003、scaled max<0.01；整网位置门槛仍为未中心化raw logits L2<0.03、KL<0.02。晋级还要求通过数严格增加，平均／最大L2、最大KL及token一致数均不退化。

| 配置 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| exact-a50 | 243/290 | 2.042862% | 13.334028% | 0.003978168 | 286/290 |
| decoded-a50 | 243/290 | 2.043470% | 13.357194% | 0.004341763 | 286/290 |

基线摘要与历史原样复现。两个新候选的通过数都没有增加，平均L2均变差，decoded-a50的最大KL也升高。冻结结果为`selected=baseline`、`candidate=exact-a50`；后者仅是内部排序更好的诊断候选，不启用为默认。

## 新提示成对诊断

冻结选择后，仅比较基线与内部候选**exact-a50**；decoded-a50未通过内部晋级，本轮没有它的新提示NPU质量结果，也不把exact候选的结果归为decoded收益。

| 路径 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 234/260 | 1.556003% | 8.383530% | 0.007061235 | 258/260 |
| candidate | 233/260 | 1.557557% | 8.255805% | 0.007275087 | 258/260 |

新提示候选少1个通过位置，平均L2和最大KL略升、最大L2降低，token一致数不变。没有根据这组结果重新选择或拟合。这里是固定B1/S64、teacher-forced数值检查，部分轨迹可能包含首个EOS后继续位置，不代表自由生成、PPL、KV或更长上下文质量。

## 资源、错误修复与回归

- **6个正式NPU进程、3,346次提交**均完成执行与资源检查并健康shutdown。180项实际执行的局部检查全部通过：校准首条30项、三个内部首条各30项、两个新提示组首条各30项。没有推广为所有轨迹的每层都经过独立局部验证；整网质量通过数仍按上表单列。
- 另一次早期运行在采完48条后遇到记录字段重复：复用拟合器已经返回`changed_coefficients`，外层再次以关键字传入导致TypeError。该进程1,600次提交、健康关闭；删除重复字段后以相同数据和数学设置重跑，基线实际SSE完全复现。它不计为候选精度拒绝，也不隐瞒为成功实验。七个实际进程合计**4,946次提交**。
- 默认1536图／512MiB预算未改，所有进程均无图淘汰、context_failed或隔离buffer；shutdown后活跃buffer、缓存session／block归零，无挂起、强制结束或驱动重置。正式拟合后端Tensor峰值21,057,024字节，内部／新提示4,787,712字节，不含全部模型／驱动内存。
- 五个完整整网验收进程的重复调用均30次提交，输出逐项一致，无新session或显式host拷贝。四项原图／保留输入控制逐项相同；无独立吞吐或功耗基准。
- 参数审计确认两候选各仅一处B矩阵变化，其余209因子、所有A及全部packed权重不变，三组完整因子哈希不同。新增10项离线测试覆盖解压取消误差、Half边界、非法格式／输入／溢出、单层更新边界和拟合身份防篡改。
- 完整工作区**374项检查通过**（336+32+6）；公开副本361项，360通过、1项原有嵌套source-release检查跳过，该项在完整工作区通过。修复记录字段后完成最终回归。
- 228份C／ELF／汇编／manifest fixture与父提交、root及公开副本逐字节相同，16份旧源码／运行时／模型身份不变。无生产后端、C/C++、SHAVE ELF、DLL/PYD、依赖、环境或模型权重变化，未下载模型。
- 公开分支`research/w8-fp16-decode-proxy`基于[S57 PR #13](https://github.com/LCS-Official/torch-intel-npu/pull/13)。root保留不参与冻结decoder的S46分类损失版本保护；公开副本沿用父分支生产后端。main、旧PR及v0.2.0 tag保持原值。

## 后续优先级

本轮验证了“显式模拟FP16解压能改善局部SSE预测”，但它没有成为更好的整网候选。下一步优先用**实际NPU主投影输出直接拟合**作为对照，继续冻结末层down输入基和其他参数，区分剩余代理误差与低秩表示／局部目标的限制。若局部实际SSE继续改善而整网仍不改善，再考虑同设备语义的整网logits目标；保持原teacher及独立新提示验收，不继续仅凭代理下降启用默认。

未新增INT4／NF4、长期训练、PPL、KV decode、自由生成或跨模型／平台认证。生产低秩选项仍默认关闭。

## 复现

使用已有项目环境和本地模型。prepare拒绝覆盖已有数据；硬件命令串行。本轮三种内部配置和两种新提示路径均正常完成；通用实现仍会保存原局部精度拒绝并中止相应整网评分，未知错误必须处理后重跑。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_decode_proxy
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_decode_proxy.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_decode_proxy.py -ScriptArguments @("fit")
foreach ($profile in @("baseline","exact-a50","decoded-a50")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_decode_proxy.py -ScriptArguments @("internal","--profile",$profile)
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_decode_proxy.py -ScriptArguments @("select")
foreach ($label in @("baseline","candidate")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_decode_proxy.py -ScriptArguments @("fresh","--label",$label)
}
```

数据在记录字段修复前已冻结，来源SHA对应当时源码；修复只删去重复记录字段，提示、token、代理算术及拟合设置未变。正式NPU记录绑定下表最终源码；两份基线NPZ与历史相同，不承诺所有JSON时间／来源元数据跨运行逐字节一致。清理后需重新prepare／fit，中间激活仅驻留内存。

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_decode_proxy.py | `df085c82ef3b901b66ee460c7bd2bb15c5387d359aa5fd5f2def0cafbeba9d89` |
| test_torch_npu_w8_decode_proxy.py | `9828e7719c556e424f9686b27bf60a166bdd3be3505ac9833b841ba10e05958a` |
| npunlock-probe/research/w8_final_layer_audit.py | `2d631c7264979af1c3c5a328aef35e32a8501df5fea6b42ed1b5bf94383fd95f` |
| npunlock-probe/research/w8_final_layer_down.py | `879942aa7e11e3fa14a1a4894afe11e544f346050c7cf0cd165996a2341b7012` |
| test_torch_npu_w8_final_layer.py | `ce923a0321432b960b2818bda0db10579cb2d925c0417a7f18701bea4f48c940` |
| npunlock-probe/research/w8_native_selection.py | `d83a9f8d2c2a21c3ee53d6375bed40be0a070bd78c5467a3e7f3ca6bfb7ddd6e` |
| npunlock-probe/research/w8_tail_quality.py | `0d4a61bea978c16b6e630e3504d8a5d9dc6eb78d936cf93f25f397922a487b61` |
| npunlock-probe/research/w8_logits_quality.py | `a865d2805c0aa87e1e29c6f534c1a51d76023f9b992e5c02fa48171e43bb7d25` |
| npunlock-probe/research/w8_qkv_quality.py | `add3cbc212e6045a6bdedb825450de2a24ae7f4655ae860690bfd2c91f9a79e0` |
| npunlock-probe/research/w8_gate_quality.py | `bc7e56ab7b6bb19291fb107b876f467e264f2cca409f02980494e1777d0e40b2` |
| npunlock-probe/research/w8_scale_quality.py | `89cbc5a302ae28ce51e3e29e91204fbbf4ab0c6fdb5f2309b9efb929b65dd0db` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_npu_lowbit.py | `1c579ad4f2de9528d3fbd407c30df52bcfd29ac11b63cc202ffacd9c9137b01d` |
| torch_npu_lowbit_correction.py | `585642159d23366afcff326b95db8189d4d960509e68e2ab7a6bb6f94d44aec5` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| models/SmolLM2-135M-Instruct/model.safetensors | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| 临时weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时decode-fit.npz | `b1349bd8c31d46e6ae16bce54eece75ac7250d92b820065eb50669ae4f2bd7f2` |
| 临时tail-data.json | `d159d2d020c28f02abfb79bc9c701ec366c40f132867c6a27a72ba90df9ecaa7` |
| 临时fresh.json | `331514c658f1a14d0ac4b2280358aa68646dc60c50aff04dfe5d5bf985d4b527` |
| 临时decode-fit.json | `1d036120048c5118c851944580c9bd4797466f2b39a638f109b3ef5e29ff0b81` |
| 临时selection.json | `c4741190c37ae2e476c4f9b0b17167eae2a1ac96e8fa9763e0120909bd93a8ea` |
| 临时factor-identities.json | `d9b68e91990723b3b6a7b7f3a7d3139b761581645a7e793151493803508f576a` |
| 临时audit-summary.json | `515f8c4a1c4cb0df2c8ac1484677174d180b87aa4564300ce7ac58124b65ba0d` |

| 临时NPU记录 | 执行状态 | 提交 | SHA-256 |
|---|---|---:|---|
| decode-fit | passed | 1696 | `1d036120048c5118c851944580c9bd4797466f2b39a638f109b3ef5e29ff0b81` |
| internal-baseline | passed | 330 | `2d6bc8ad95232af9893728f6dc1e69571cd791f36d917f0dcb1a8ba625b88bf3` |
| internal-exact-a50 | passed | 330 | `d8c80fdc2109f6ab24dd8894843fac77075493115c1c686b6584ef6956ce03bf` |
| internal-decoded-a50 | passed | 330 | `a2338ea017dcb319f9e44914f045ef931216b8f5291558e8a7ae0eb8131d4578` |
| fresh-baseline | passed | 330 | `fa6847db0a857f9e47b240cc23efbc7e9d2f3d1c004f6e35ca8af1b15c7800b3` |
| fresh-candidate | passed | 330 | `402de85eac339010e762d11b392a8b18ca40b69e66d01fdc1a9b9a11c1cf9b1a` |
| decode-fit-recording-error | failed | 1600 | `29a30289a97cc036d0f07e13a9ffecb0cbe2fe0bae79332d20dad4dacbb9f6ba` |

## 清理

2026-10-30末次删除本轮结果目录35份临时NPZ、JSON和日志（含首次记录错误、参数／token、拟合／验收／审计及清理核对清单），共116,200,125字节（110.817MiB）。清理前后1,997份保留源码、内核、文档、公开副本及运行组件SHA-256全部一致，随后补充本段及存储台账。结果目录仅保留REPORT.md；模型、环境和全局缓存保留，七个实际NPU进程均已退出。
