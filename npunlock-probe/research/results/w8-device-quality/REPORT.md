# S49：NPU同输入数值诊断与设备内筛

2026-10-28，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；沿用本地SmolLM2-135M-Instruct。

本轮新增同输入逐阶段诊断和真实NPU内部筛选。五个取样层中，两次低秩GEMM误差很小，补偿与残差相加逐项相等；更大的相对误差位于门控输入和主W8投影。设备内筛仍选中input-local，内部通过243→246/290，但独立前向/KV复现S48的有限收益与退化，**不替换原rank32实验配置，不改变默认实现或验收门槛**。

筛选入口现已实际在NPU执行，而非仅用CPU参考挑选。诊断在本次取样下没有发现应直接修改补偿相加或两次低秩矩乘的错误；剩余数值对齐应优先拆解门控/主矩乘及其设备校准目标。

## 固定协议与源码关系

- 原S44的40条轨迹、W8权重与rank32因子重新生成，NPZ哈希与S44/S48一致。复用S48的8种候选及拟合实现：32条/2048行只用于拟合，8条/290位置只用于内部选择；没有新增候选或重调系数。
- 拟合输入仍为CPU精确W8参考的量化前缀，尚未使用设备激活重新校准。本轮新增加的是设备内筛，以及诊断时读取真实NPU中间结果；激活只在内存中使用，不落盘。
- NPU内筛全部B1/S64，编号4/9/14/19/24/29/34/39，九种既有配置各独立进程；另设全FP16控制。以失败位置数、最大raw L2、平均raw L2排序，同分优先基线。FP16只作控制。
- 每个内部进程先核对首例全部30层的staged CPU误差，再完成8条轨迹。选择文件绑定十份设备记录、拟合文件和两份源码的SHA；未完成或不健康的进程不能参与选择。
- 选择冻结后才调用独立评测入口。本轮程序没有使用独立/文本行选择候选，但这些集合是此前已评过的历史数据，不是新盲测；候选家族亦来自S48。
- 复用S48的独立评测器，仅在本CLI进程临时注入冻结选择加载函数并在finally恢复；未更改Torch全局算子、生产后端、指标、生产模型图或默认配置。相关路由/异常恢复有CPU测试。
- 当前研究依赖[PR #4](https://github.com/LCS-Official/torch-intel-npu/pull/4)的S48工具，因此新分支以该分支为父分支；不自动合并旧PR或更新v0.2.0 tag。

## 同输入诊断设计

只使用内部轨迹4，基线30层前向推进，固定检查层索引0/6/14/24/29。每个取样层从真实NPU attention残差和SHAVE RMSNorm输出继续执行FFN；由原FFN图导出门控输入、主W8输出、低秩中间值、补偿输出、校正后输出与最终残差。保留原decompression注解、Swish版本及全部输入。

每一项比较使用相同的实际前驱输入：主W8矩乘与精确code×scale以及FP16舍入权重两种CPU参考比较；第二次补偿GEMM使用实际NPU第一次GEMM输出作为输入；两次相加使用实际NPU的两个操作数。这样把局部运算差异与前缀传播区分开。

额外导出可能改变编译器优化，所以另核对多输出诊断图与普通FFN、分段执行与完整融合层的输出。本次五层均逐元素数值相等；只承诺这个取样，不推广为所有形状/编译路径相等。所有诊断图均核验DPU矩阵及INT8任务证据。

下表为相对向量L2，分母是各阶段自身输出；不同阶段的百分比不能直接当作对最终logits的贡献率。

| 层 | RMSNorm2 | 门控输入 | W8精确权重参考 | W8舍入权重参考 | 第一次补偿GEMM | 第二次补偿GEMM（同输入） |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.000314% | 0.054273% | 0.024081% | 0.027827% | 0.000534% | 0.000090% |
| 6 | 0.000000% | 0.054272% | 0.024935% | 0.033628% | 0.000311% | 0.000304% |
| 14 | 0.000465% | 0.054331% | 0.026228% | 0.031991% | 0.000542% | 0.000449% |
| 24 | 0.000000% | 0.057373% | 0.023675% | 0.033136% | 0.000481% | 0.000261% |
| 29 | 0.000155% | 0.038865% | 0.057425% | 0.022327% | 0.000744% | 0.000588% |

五层的补偿相加、残差相加、导出图与普通FFN、分段与融合层四项比较均relative L2=0、逐项相等比例=100%。两次低秩GEMM约0.00009%–0.00074%，门控输入约0.0389%–0.0574%，主W8精确权重参考约0.0237%–0.0574%。

门控输入的比较包含gate/up投影、SiLU与乘法的组合，本轮未单独分离这些来源，不能直接断定是SiLU误差。FP16舍入权重参考在最后一层更近，却在其余四层更远；没有据此统一改写CPU参考或放宽原门槛。没有证明驱动故障或FTZ问题。

## 真实NPU内部筛选（290位置）

| 配置 | 通过 | 平均raw logits L2 | 最大L2 | 最大KL | 相同token |
|---|---:|---:|---:|---:|---:|
| baseline | 243/290 | 2.0419% | 13.3917% | 0.0040497 | 286 |
| input-local | 246/290 | 2.0279% | 12.3620% | 0.0040646 | 286 |
| block-0.25 | 180/290 | 3.2044% | 18.9547% | 0.0340266 | 283 |
| block-0.50 | 180/290 | 3.2349% | 23.5575% | 0.0328698 | 283 |
| block-1.00 | 218/290 | 2.4190% | 16.1922% | 0.0094283 | 285 |
| basis-local | 241/290 | 2.0727% | 11.7007% | 0.0037479 | 286 |
| basis-0.25 | 241/290 | 2.1451% | 13.5450% | 0.0046200 | 286 |
| basis-0.50 | 224/290 | 2.5089% | 16.2428% | 0.0067066 | 286 |
| basis-1.00 | 183/290 | 3.0111% | 16.8388% | 0.0080390 | 286 |
| fp16 | 290/290 | 0.3417% | 1.5620% | 0.0001814 | 289 |

设备内筛仍选中input-local。CPU内部基线S48为246/290，真实NPU基线为243/290，说明CPU结果不能直接代替设备统计。设备内筛确认其内部小幅优势，也不保证独立模型质量达标。全FP16控制290/290通过。

## 独立CPU与真实NPU复核

保留原17例/142位置与S45的16例/128位置。CPU为原长度；NPU forward为64-token右侧补零因果计算，只计原位置；KV为bird/cat/fox/repeat32的98位置，one-token prefill和分组decode。全部30层decoder及KV在NPU，embedding、末端norm/lm_head为显式CPU边界。

| 平台/路线 | 配置 | 集合 | 通过 | 平均raw L2 | 最大L2 | 最大KL | 相同token |
|---|---|---|---:|---:|---:|---:|---:|
| CPU | baseline | regression | 126/142 | 1.7505% | 10.1581% | 0.0212464 | 141 |
| CPU | baseline | fresh | 111/128 | 1.6415% | 6.0134% | 0.0072343 | 128 |
| CPU | candidate | regression | 126/142 | 1.7196% | 9.9508% | 0.0213116 | 141 |
| CPU | candidate | fresh | 112/128 | 1.6077% | 5.9964% | 0.0071497 | 128 |
| CPU | fp16 | regression | 142/142 | 0.0000% | 0.0000% | 0.0000000 | 142 |
| CPU | fp16 | fresh | 128/128 | 0.0000% | 0.0000% | 0.0000000 | 128 |
| NPU forward | baseline | regression | 126/142 | 1.6581% | 9.7826% | 0.0198278 | 141 |
| NPU forward | baseline | fresh | 113/128 | 1.6319% | 6.3928% | 0.0067631 | 127 |
| NPU forward | candidate | regression | 127/142 | 1.7098% | 10.6301% | 0.0212879 | 141 |
| NPU forward | candidate | fresh | 113/128 | 1.6338% | 6.6327% | 0.0062594 | 127 |
| NPU forward | fp16 | regression | 142/142 | 0.3165% | 1.1532% | 0.0001050 | 142 |
| NPU forward | fp16 | fresh | 128/128 | 0.3084% | 1.5434% | 0.0002107 | 128 |
| NPU KV | baseline | 4条轨迹 | 82/98 | 1.9752% | 10.1119% | 0.0198410 | 97 |
| NPU KV | candidate | 4条轨迹 | 82/98 | 1.8832% | 9.7444% | 0.0198467 | 97 |

NPU历史前向126→127/142、新集合仍113/128；平均/最大误差及最大KL退化。KV仍82/98：修好cat/48、repeat32/52，新增cat/46、repeat32/45失败，与S48一致。每位置仍要求raw L2<0.03且KL<0.02；去均值和token一致性不替代原门槛。

三个forward进程的90层同输入staged检查通过，全FP16独立控制270/270通过；两组KV的8次末步重放均输出数值相等且不新增session或host upload/download，每次30次调用。

## 文本证据复用范围

选出的input-local与S48完全相同。核对W8 NPZ、基线因子NPZ、新down因子NPZ、模型、适配器、Python后端、session DLL和PYD共8项SHA，均与S48报告一致。由此本轮引用[S48报告](../w8-output-quality/REPORT.md)的历史文本证据，**没有重跑或新增PPL/生成认证，也没有下载文本**。这些历史调用不计入本轮NPU提交数。

S48固定2048目标的CPU PPL基线28.036675007、候选28.035407361；NPU基线28.037196037、候选28.041923332，八条生成与原基线相同。因此即使用硬件选出这一候选，也没有新增证据支持将它设为默认。

## 验证与边界

- 16个正式NPU进程：1个诊断、10个内部配置、3个独立前向、2个KV；共18,500次提交，全部正常shutdown，context_failed=false，live buffers、cached blocks/sessions和隔离字节归零。
- 默认1536图/512MiB预算保留。390个逐层staged检查通过；审计复算4716份质量位置的门槛、通过数、均值/极值与token计数，以及50份诊断比较。执行/资源验收通过不表示所有低位质量位置通过。
- 本地272项离线检查通过：根目录234（含7项新图导出/设备选择/路由恢复检查）、Python API32、发布工具6。独立公开分支259项检查：258通过、1项因没有嵌套source-release副本按原规则跳过；该对照在完整工作区已通过。
- 228份fixture C/ELF/汇编/manifest与父分支一致；未修改生产后端、内核、原生库、默认参数、安装依赖或模型。没有新环境、模型下载、图表或持久图缓存。
- 不提供任意模型/shape的数值归因；没有分离gate/up/SiLU各自影响，没有实际NPU激活拟合，没有新增INT4/NF4整网、低位训练、长上下文、性能或能效认证。
- 下一步可先分离门控子链与主W8投影的设备计算，再研究使用实际设备激活/输出的校准；本轮结果不支持优先改写已经高度一致的补偿相加。

## 复现

需要父分支S48工具、已有运行环境及SmolLM2-135M-Instruct。prepare拒绝覆盖已有权重；临时数据清理后需重新准备。各NPU命令必须串行。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest test_torch_npu_w8_device
foreach ($stage in @("prepare","fit","probe")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_device_quality.py -ScriptArguments @($stage)
}
foreach ($profile in @("baseline","input-local","block-0.25","block-0.50","block-1.00","basis-local","basis-0.25","basis-0.50","basis-1.00","fp16")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_device_quality.py -ScriptArguments @("internal","--profile",$profile)
}
foreach ($stage in @("select","independent")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_device_quality.py -ScriptArguments @($stage)
}
foreach ($variant in @("baseline","candidate","fp16")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_device_quality.py -ScriptArguments @("hardware","--variant",$variant)
}
foreach ($variant in @("baseline","candidate")) {
  .\run_torch_npu.ps1 -Script npunlock-probe\research\w8_device_quality.py -ScriptArguments @("hardware","--variant",$variant,"--route","kv")
}
```

CLI还提供复用S48的data/cpu-language/hardware-language入口；本轮未执行这些入口。需要重新跑文本时必须联网获取固定切片，并将新结果另行记录，不能把这里引用的S48数值称作新测。

## 源码与产物身份

| 文件/产物 | SHA-256 |
|---|---|
| npunlock-probe/research/w8_device_quality.py | `852501d5f89aaba05eaf033f6f4088cfa3dc4d0ddc95313f283bee864196b366` |
| test_torch_npu_w8_device.py | `a5d378d20badd378ea61394214876561aaf82fcd87088485e79427052575551c` |
| npunlock-probe/research/w8_output_quality.py | `080faae3356189bce7530fc0f8e995b966b017c60da0c4df5ad202946ead2894` |
| torch_npu_smollm.py | `e031aa61b603c5f5f910c56d4dff6057942e8d3f57e202164c55bc21a06c6515` |
| torch_intel_npu.py | `fbf33413ac5497d5b6f4764360ad658850139a69e04dfc586cc83336d9ec4e30` |
| npunlock-probe/session-runtime/npunlock_session.dll | `1bd05e741f3d9eb8ada857d044783c0a5d90f6802d09161cdba27c2f3d8c2d90` |
| npunlock-probe/torch-device/_torch_intel_npu.pyd | `3ace20a4dca300ee54cc21d8d68de359848c5e7770476c38f34770a2c08bb5fd` |
| 临时 weights.npz | `f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62` |
| 临时 corrections.npz | `659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195` |
| 临时 input-local.npz | `3b0bb375accc574145cac82b85bf502b0b127bd283435e010f9f4ba4bd2da449` |
| 临时 probe.json | `a8f1b6fcb69cf92952f869854cc97526e7cfeb853b2d890b13b7641c83885076` |
| 临时 selection.json | `212e1ca5c5ff3d2261d26b681cc1b0cbf4cffe1d65538c9c0684860ffd4a5e30` |
| 临时 independent.json | `8b56814f0ade6f7c7f18a60202c4becb0b306f24ae795ccc230830ee628927bf` |
| 临时 language-evidence-reuse.json | `4d9b5f012f181514f0849bdfe52135ec26528fdf14996af5ee3dd524fbe60b60` |

## 交付与清理

保留一份研究驱动、7项新离线测试及本报告。研究分支基于S48父分支保存历史；不合并旧PR，不改发布tag或历史SOURCE_MANIFEST。原始NPZ、JSON与日志将在报告核验后清理，统计随后追加。

本轮删除59 份临时NPZ、JSON和日志，共147664656 字节（140.824MiB）。清理前后核对1937份源码、内核、文档、公开副本与运行组件SHA-256，全部一致，随后追加本清理记录及空间台账。研究目录仅保留REPORT.md；原模型、环境和全局缓存保留，本轮测试进程均已退出。
