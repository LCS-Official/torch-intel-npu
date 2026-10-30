# S56：真实 NPU 校准选择与独立泛化验收

2026-10-29，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0。沿用本地 SmolLM2-135M-Instruct、现有 W8＋rank32 适配器及项目环境。

本轮将真实 NPU 整网输出纳入校准候选选择：在原48条校准轨迹上比较六个预先固定的更新幅度，按实际设备目标选择，再冻结候选并执行原内部晋级规则。真机校准目标降低 **8.5975%**，通过位置1604→1622/1657；选中的数值参数仍是S55第二轮全幅更新。内部通过仍243/290，回归491→487/528；新提示的基线和候选均触发末层局部守卫。因此**不晋级、不改变默认基线**，没有把校准改善写成独立质量改善。

这里的baseline指实验中的W8＋rank32基线；低秩校正选项仍默认关闭，没有将其自动启用于普通模型。

## 固定边界与选择顺序

- 不改INT8编码、FP16 scale、全部rank32输入基或推理结构；只在准备阶段改变全部210个已有输出矩阵，4,976,640个FP32 master参数，前向使用Half。NPU参数量、存储字节和图拓扑不增加。
- CPU拟合直接复用[S55](../w8-tail-quality/REPORT.md)的48条校准轨迹、原FP16 teacher、整批梯度和两次更新。两次主目标为0.001217754960／0.001123347702，初始0.001238042029。没有根据真机、内部或新提示指标重新训练。
- 六个候选在首次真机校准前固定：两次CPU更新各与原输出因子按25%／50%／100%混合，FP32计算插值再舍入Half。预算和目标中的位置按提示标识及绝对token位置对齐，重复、缺失、非有限或负预算均拒绝。
- 真机校准使用原32条加16条新增校准，共48条／1,657个位置。原内部8条／290位置不参与真机校准，仍只执行原晋级规则。选择后保留16条已经查看过的回归提示／528位置，另冻结8条新的独立提示／268位置。
- 新提示为zipper、magnet、books、thanks、seesaw、dawn、station、postcards；在拟合前冻结，与原校准、内部、历史回归及S45提示做文本去重。回归集是S54／S55已查看提示的合并，不能称为新的独立验收集。
- 先冻结三组文本／token，再拟合CPU种子，然后串行运行七份真机校准；按真机目标选择并保存全部记录哈希，之后才执行内部、回归及新提示。冻结选择与运行记录被更改或缺失时，下游验收拒绝继续。没有按内部／新提示结果改选。

真实NPU执行完整30层decoder；embedding、最后norm与lm_head仍按现有研究路径显式在CPU完成，原FP16 teacher也在CPU。不是完整模型所有算子都驻留NPU，更不能推广为普通`.to("intel_npu")`的通用低位支持。

## 真实 NPU 校准目标

沿用S55的每提示目标，`e`为未中心化raw logits相对L2平方，`k=KL(teacher||student)`；top10%分别按e和k取`ceil(0.1*N)`，至少一个位置。区别在于本轮候选及逐位置baseline预算都来自实际NPU输出，而非CPU量化参考：

```text
mean(e + 0.1*k + 2*relu(e-0.03^2))
+ 0.5*mean(top10%(e)) + 0.5*mean(top10%(k))
+ 2*mean(relu(e-e_native_baseline))
+ 2*mean(relu(k-k_native_baseline))
```

48条提示的损失等权平均；原位置门槛仍为L2<0.03、KL<0.02，局部同输入守卫仍为L2<0.003、scaled max<0.01。选择首先按真实设备校准目标最小值，完全相同时再用原质量排序；目标没有严格优于baseline时不晋级。未知异常、不健康关闭或不完整记录中止选择；仅已记录的局部精度拒绝可被排除。

| 真机校准路径 | 校准目标 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.001265837506 | 1604/1657 | 1.209343% | 6.649548% | 0.005075410 | 1640/1657 |
| tail-e1-a25 | 0.001341813770 | 1601/1657 | 1.206989% | 6.481218% | 0.005628136 | 1640/1657 |
| tail-e1-a50 | 0.001281052586 | 1613/1657 | 1.195729% | 5.833276% | 0.004417712 | 1642/1657 |
| tail-e1-a100 | 局部守卫拒绝 | — | — | — | — | — |
| tail-e2-a25 | 0.001290483204 | 1609/1657 | 1.200703% | 5.636230% | 0.004637922 | 1643/1657 |
| tail-e2-a50 | 0.001231028685 | 1616/1657 | 1.197311% | 5.682629% | 0.004138928 | 1642/1657 |
| tail-e2-a100 | 0.001157007307 | 1622/1657 | 1.169664% | 4.729521% | 0.004271543 | 1642/1657 |

`tail-e1-a100`在第30层scaled max=0.01012098416686058，relative L2=0.0006843818118795753，按原规则拒绝，未继续计算其整网校准分数。不是设备错误，也没有放宽精度门槛。

冻结的设备选择为`tail-e2-a100`，校准目标从0.001265837506降至0.001157007307；通过数多18，但不能据此通过内部晋级。六种混合没有找到比S55数值参数更好的新方案，本轮主要新增了真实设备参与选择及完整冻结审计流程。

## 内部与已查看回归集

| NPU路径／集合 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| internal-baseline | 243/290 | 2.041913% | 13.391750% | 0.004049684 | 286/290 |
| internal-tail-e2-a100 | 243/290 | 2.028494% | 13.225685% | 0.004538313 | 285/290 |
| regression-baseline | 491/528 | 1.551949% | 8.233340% | 0.034045011 | 520/528 |
| regression-candidate | 487/528 | 1.550132% | 7.966191% | 0.035698410 | 520/528 |

原晋级要求通过数严格增加，并且平均／最大L2、最大KL及token一致数均不退化。内部候选仍243/290，最大KL升高且token一致数少1，故最终`selection.selected=baseline`。随后回归候选通过数少4、最大KL升高；这与S55对应两组相加的方向一致。相同参数的重新选择没有解决泛化问题。

## 新独立提示：按原守卫拒绝，整组分数缺失

新集合第一条`zipper`（Describe how a zipper closes a jacket.）在第30层触发同输入CPU参考守卫：

| 路径 | 局部relative L2 | scaled max | max abs | 结果 |
|---|---:|---:|---:|---|
| baseline | 0.0006576719461008906 | 0.010628203861415386 | 4.0 | 原0.01守卫拒绝 |
| candidate | 0.0006463177851401269 | 0.010292915627360344 | 4.0 | 原0.01守卫拒绝 |

两条路径都完成30个局部检查后健康关闭，整网评分位置数为0。**没有有效的NPU 268位置通过率**，不能写成0/268，也不能用下面CPU结果补作NPU分数。这里的4.0是未经过最后norm/head的局部hidden-state绝对差异，不能等同于logits误差或模型质量。

基线也失败，说明这个局部边界并非新候选特有；但尚未定位是哪个计算阶段造成，不能直接归因为驱动、单个DPU算子或Half舍入。没有跳过首条提示、忽略坐标、继续未认证评分或调整门槛。下一轮优先在相同输入下对第30层attention／FFN边界做数值重放，并区分局部参照差异与整网量化泛化。

## CPU整网参考诊断

冻结候选后，用相同token、原FP16 teacher和相同W8＋rank32参数执行六组CPU对照；assert没有导入torch_intel_npu、没有NPU调用。它只诊断泛化，不参与选择、不替代NPU验收。

| CPU路径／集合 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline/internal | 246/290 | 2.002960% | 13.091786% | 0.003829703 | 286/290 |
| baseline/regression | 492/528 | 1.541576% | 7.363860% | 0.033708718 | 520/528 |
| baseline/fresh | 238/268 | 1.723973% | 6.100598% | 0.004465702 | 266/268 |
| candidate/internal | 245/290 | 2.029757% | 12.961422% | 0.003840984 | 286/290 |
| candidate/regression | 491/528 | 1.559234% | 7.714130% | 0.032097068 | 520/528 |
| candidate/fresh | 232/268 | 1.766411% | 6.396745% | 0.004531424 | 266/268 |

CPU候选在内部、回归及新提示的通过数均下降，新提示238→232/268、平均及最大L2／KL均升高。至少存在校准泛化不足，不能仅靠进一步压低设备校准目标认定低位质量已解决；NPU新集合又有独立局部守卫问题，两者都需要处理。

这些仍是固定B1/S64、teacher-forced数值检查。部分提示的64-token轨迹包含首个EOS后的继续位置，不是自由生成质量或独立样本量。新提示这轮已被查看，后续需另留未查看文本；没有新增PPL、KV decode、自由生成、长期训练、INT4／NF4或其他模型认证。

## 参数身份与有符号零

两份基线NPZ及CPU拟合种子NPZ与S55逐字节相同；拟合种子SHA为`73e5fbde2b5afe651ffa84e55b3336a23438d94c325cbd384acc724f619e6863`。本轮最终100%混合使用`prior + (fitted-prior)`，全部210个输出因子的数值与S55第二轮种子完全相同，但该公式归一化了**33个有符号零**，最终因子字节并非完全相同。没有非零系数变化，也没有把最终混合因子宣称为逐字节复刻。基线及六个候选的七个运行因子哈希全部不同，避免缓存身份混用。

## 资源、回归与范围

- **13个正式NPU进程、11,250次提交**：10个完整完成执行／资源检查，3个按原局部精度规则拒绝（一次校准、两次新提示）。所有进程都健康shutdown，context_failed=false，活跃buffer、session、隔离字节及缓存block归零；没有挂起、强制结束或驱动重置。
- 390项局部检查中387通过、3项拒绝，不能写成390项全部通过。十个完整进程的重复整网调用仍30次提交，没有新session或显式host拷贝，输出逐项一致；未运行额外吞吐或功耗基准。
- 默认1536图／512MiB预算未改，13份记录淘汰均为0。完整进程后端Tensor峰值4,787,712字节，拒绝路径4,713,984字节；不包含全部模型／驱动内存。
- 新增14项离线检查，覆盖有限混合、输入基保护、有符号零、位置预算对齐、真实设备目标选择、未知失败保护、只允许冻结候选进入内部验收，以及最终验收必须验证设备选择身份。
- 完整工作区**354项检查全部通过**（316+32+6）；公开副本341项检查，340通过、1项原有嵌套source-release检查跳过，该项在完整工作区通过。
- 228份C／ELF／汇编／manifest fixture与父提交、完整工作区、公开副本逐字节相同，9份旧源码／运行时／模型身份不变。未修改生产后端、C/C++、SHAVE ELF、DLL/PYD、依赖或环境，没有模型下载。
- 完整工作区保留S46分类损失版本保护，它不参与冻结decoder；公开副本沿用父分支生产后端。本轮独立分支`research/w8-native-selection`基于[S55 PR #11](https://github.com/LCS-Official/torch-intel-npu/pull/11)，不修改main、旧PR或v0.2.0 tag。

## 复现

在已安装项目环境和已有本地模型下执行。首次prepare同时冻结数据；已有文件会拒绝覆盖。硬件命令串行，已知局部精度失败必须保留记录；未知失败或不健康设备不能自动忽略。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_native_selection
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("prepare")
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("fit")
foreach ($profile in @("baseline","tail-e1-a25","tail-e1-a50","tail-e1-a100","tail-e2-a25","tail-e2-a50","tail-e2-a100")) {
    try {
        .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("calibrate","--profile",$profile)
    } catch {
        $r = Get-Content -Raw "npunlock-probe/research/results/w8-native-selection/calibration-$profile.json" | ConvertFrom-Json
        if ($profile -ne "tail-e1-a100" -or -not $r.healthy -or -not $r.local_guard_failure -or -not ($r.local_guard_failure.relative_l2 -ge 0.003 -or $r.local_guard_failure.max_scaled_error -ge 0.01)) { throw }
    }
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("select-device")
# Read device-selection.json for the frozen diagnostic candidate; this run chose tail-e2-a100.
foreach ($profile in @("baseline","tail-e2-a100")) {
    .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("internal","--profile",$profile)
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("select")
foreach ($group in @("regression","fresh")) {
    foreach ($label in @("baseline","candidate")) {
        try {
            .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("evaluate","--group",$group,"--label",$label)
        } catch {
            $r = Get-Content -Raw "npunlock-probe/research/results/w8-native-selection/$group-$label.json" | ConvertFrom-Json
            if ($group -ne "fresh" -or -not $r.healthy -or -not $r.local_guard_failure -or -not ($r.local_guard_failure.relative_l2 -ge 0.003 -or $r.local_guard_failure.max_scaled_error -ge 0.01)) { throw }
        }
    }
}
.\run_torch_npu.ps1 -Script npunlock-probe\research\w8_native_selection.py -ScriptArguments @("cpu-audit")
```

数据最初冻结时的来源SHA记录了加入最终选择身份检查之前的源码；提示、token、拟合方法及数值目标未因此改变。所有正式真机记录和拟合记录均绑定下表最终源码。清理后需重新prepare／fit；不承诺所有JSON元数据和驱动输出跨运行逐字节相同。

## 源码与运行身份

| 文件 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_native_selection.py | `d83a9f8d2c2a21c3ee53d6375bed40be0a070bd78c5467a3e7f3ca6bfb7ddd6e` |
| test_torch_npu_w8_native_selection.py | `eab967381104df92903fd9f09d4ecb9297f9abd2f7285a5ceba90e4d36d2c94a` |
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
| 临时tail-data.json | `c6439632e04e8fc7e5e5739549039b16be2f5cc90e661a8b8076e9bb26be3241` |
| 临时regression.json | `32ccccbe1c74bbae4578185096669d4af46d289acc48636ef929474e04983f3b` |
| 临时fresh.json | `87df8d60096c408bb8c1577f2880386c0f13a9a4915631c3a0910dff37e600cd` |
| 临时logits-fit.json | `12f91028ee6c843974ca4042e302ff92b43c69ccd6cb7e6b6b13ffcbb57986f7` |
| 临时device-selection.json | `1530bc74ecc5f3b699d19f844d246dfd672998f55746db21abe3aae7a60904fb` |
| 临时selection.json | `f4a1db307e4dffc8218098e5ebec0e353dc9ef9bff2109c20c6eceb83bc6729a` |
| 临时factor-identities.json | `23a60bd6d82f35f329e02f07d919c5ff3c30e4ffbd8227af4d14412030ac78a8` |
| 临时cpu-generalization.json | `aec17c976aecc28e5665edda2d548e6107597bf90739d66767e61111268b65d0` |
| 临时audit-summary.json | `3317678f88e764ca8d2812ad1d94a52b881a17cdfa6c61cd59e96f261555339f` |

| 临时NPU记录 | 执行状态 | 提交 | SHA-256 |
|---|---|---:|---|
| calibration-baseline | passed | 1530 | `62a1e78e67bc5dff38eb1c434d4a22f2a5f5c7aaca2bc8652dcc515e85eae648` |
| calibration-tail-e1-a100 | failed | 90 | `775b1c1f142f83f5b93d90b6c7294487ac7342234871ed95dffc8921c5915287` |
| calibration-tail-e1-a25 | passed | 1530 | `272e6f0da0d646d4963e9add253ee615de587732d44b6d6eee1c8066a993572f` |
| calibration-tail-e1-a50 | passed | 1530 | `af7bc5bfc2aa04384fb456421cad9cc7852c9bec31771483f130a7141fa84a57` |
| calibration-tail-e2-a100 | passed | 1530 | `d5163ab88409f328d9e1e06dbf30faf4613b1a72b3bb1733807d2c2da7c0c179` |
| calibration-tail-e2-a25 | passed | 1530 | `f18ff0beef48064caba32c751468b05155bb7c1d7c09f9993f1e171adbbdd05f` |
| calibration-tail-e2-a50 | passed | 1530 | `bbe98000c91e68c6d5a68fe7c5435cd275d0c52fce7e1cc9c55006ade1a58554` |
| fresh-baseline | failed | 90 | `988e55f2b9ee98af1024c84fff1dc121a26e7337f365e345d7b093c921f660ff` |
| fresh-candidate | failed | 90 | `bc1ddad7c830c1b559d284949a188a2fcc805203a8ebad77bc64011f42fa32b2` |
| internal-baseline | passed | 330 | `9550594a83c225897ea704114436d92ebfddde4cf7b40400dfd135026dc47dba` |
| internal-tail-e2-a100 | passed | 330 | `cec04d58aa8d2c89475a12040802c21476db80af706519b9f98cc702294bbe92` |
| regression-baseline | passed | 570 | `992f944eadfa93cbb0d0d2dcdeae7e688cd924314be733f8b2891de97d182531` |
| regression-candidate | passed | 570 | `d8f57be2a0d66288de2c54e0cdd8fbaf8c057ba44b1d5ebdeeb9bbeb61db6a13` |

## 清理

2026-10-29末次删除本轮结果目录54份临时NPZ、JSON和日志（含CPU对照、冻结选择／运行审计及清理核对清单），共140,222,557字节（133.727MiB）。清理前后1,983份保留源码、内核、文档、公开副本及运行组件SHA-256全部一致，随后补充本段及存储台账。结果目录仅保留REPORT.md；模型、环境和全局缓存保留，13个正式NPU进程均已退出。
