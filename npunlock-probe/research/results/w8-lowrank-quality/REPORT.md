# S44：W8 整网低秩误差修正

2026-10-27，Windows / Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），驱动32.0.100.5540、编译器8.3、Torch2.14.1+cpu、Transformers5.18.0；使用已有本地SmolLM2-135M-Instruct。

本轮加入默认关闭的 **INT8主权重＋FP16低秩残差**，准备、模型图和真机整网均已跑通。真实NPU的142个独立回归位置，平均logits相对L2从2.2377%降至1.7145%（下降23.38%），质量通过从118增至126；decode图常量比全FP16少38.01%。

**这仍不是低位整网质量全部达标。** 新配置仍有16个位置超过原3% L2门槛，最大KL比全W8更差；2,048个真实文本目标的PPL仅改善约0.0099%，8条自由生成与全W8完全相同。因此保留为实验选项，不替换原默认，不宣称语义质量明显提高。INT4/NF4整网本轮未扩展。

## 方法与实现

概念参考[LQER，ICML2024](https://proceedings.mlr.press/v235/zhang24j.html)的低秩量化误差重建，本项目自行实现带1% ridge的完整激活协方差目标，没有复制其代码，也不是该论文方案的逐项复现。

对原FP16权重W和未提前舍入的W8解码权重Wq，令E=W−Wq，H=XᵀX/m+0.01·mean(diag(XᵀX/m))·I；零能量时保留数值下限。Cholesky分解H=LLᵀ，对EL做FP64 SVD，截取rank r，得到近似E的两个因子B、A；还原协方差变换并存为FP16。推理计算为：

```text
y8 = FP16(X @ Wq.T)
z  = FP16(X @ A.T)
dy = FP16(z @ B.T)
y  = FP16(y8 + dy)
```

主投影仍是逐行INT8权重、FP16激活。两次补偿矩乘和加法写入同一个NPU编译图；CPU只负责明确的离线校准与数据准备。FP16因子和原W8图均是冻结推理参数，不提供梯度或低位训练。

改动包括：

- [torch_npu_lowbit_correction.py](../../../../torch_npu_lowbit_correction.py)：不可变bytes因子、形状/有限值/范围检查、FP64校准和rank截取；核心依赖沿用torch/numpy。
- [torch_npu_smollm.py](../../../../torch_npu_smollm.py)：Layer/Stack新增可选`corrections=None`；forward、prefill、decode和显式CPU参考均覆盖。Stack按绝对checkpoint层编号传递完整map。只接受匹配shape、rank≤32、已有显式W8权重的投影；同一投影不能同时启用原FP16保护和残差。
- [研究与复现脚本](../../w8_lowrank_quality.py)、[审计](../../w8_lowrank_audit.py)、4项核心和5项可选模型离线测试，以及CI登记。

没有修改`quantize_weight/calibrate_int8`、SHAVE ELF/C、C++、DLL/PYD或默认图缓存预算。默认FP16/W8的普通图(1,3,576)、fused prefill1/decode1共6组XML/常量与已发布副本逐字节相同，默认CPU参考逐位一致；默认`corrections=None`保持原行为。输入dict在准备时复制，因子本体不可变，防止后续修改使图缓存身份失效。

## 固定校准与选择

沿用旧40条提示，每条使用官方CPU FP16 greedy轨迹到64个消费token。编号`i%5!=4`的32条、2,048行激活拟合全部30层/210个投影；另8条、290个预测位置只用于rank选择。独立回归和WikiText没有参与拟合或选rank。

W8基底仍是原GPTQ适配：FP64、1% damping、128列块、固定FP16逐行scale与逐行RTN回退。210个矩阵校准SSE均不超过RTN。准备W8约125.64秒，另拟合最大rank32因子约26.46秒，均为CPU离线过程，不是NPU推理速度。

候选rank **8/16/32在评测前固定**，小rank使用同一组最大rank因子的前缀。按失败位置数、最大原始L2、平均原始L2排序，最后才比较因子字节。全FP16和原50%保护策略仅作控制，不参与本轮rank排名。

| 内部CPU候选 | 通过/290 | 平均原始logits L2 | 最大L2 | 最大KL |
|---|---:|---:|---:|---:|
| 全W8 | 189 | 2.8033% | 11.6684% | 0.008204 |
| ＋rank8 | 214 | 2.3447% | 10.9067% | 0.005039 |
| ＋rank16 | 224 | 2.2414% | 10.2003% | 0.004409 |
| **＋rank32，选中** | **246** | **2.0030%** | **13.0918%** | **0.003830** |
| 旧50% FP16保护控制 | 259 | 1.5953% | 8.1314% | 0.003772 |
| 全FP16控制 | 290 | 0 | 0 | 0 |

rank32按预设的“先减少失败位置”规则胜出，其最差L2反而高于rank16，不能描述为每个指标都更好。旧50%保护在内部质量上更强，但保留FP16的权重和图常量更多。

## 独立模型回归

沿用17个历史回归案例、142个预测位置；它们与40条拟合/内部选择提示不重合，但不是新采集的大型盲测。bird/garden/cat各32位置，重复前缀33位置，其余13条各1位置。全部配置消费同一串CPU greedy token进行teacher forcing，容量64。

参照为官方CPU FP16完整模型；保持原门槛 **原始logits相对L2 < 3%、KL(reference||actual) < 0.02**。不以去均值指标替换原门槛，不把argmax一致视为全部质量通过。

| 路径/配置 | 通过/142 | 平均logits L2 | 最大L2 | 最大KL | argmax一致/142 |
|---|---:|---:|---:|---:|---:|
| CPU W8 | 117 | 2.243806% | 10.133561% | 0.011365214 | 140 |
| CPU W8＋rank32 | 126 | 1.750537% | 10.158082% | **0.021246415** | 141 |
| CPU旧50%保护控制 | 129 | 1.470607% | 8.115854% | 0.002540685 | 140 |
| **NPU W8** | **118** | **2.237747%** | **10.905755%** | **0.011397425** | **140** |
| **NPU W8＋rank32** | **126** | **1.714470%** | **10.111856%** | **0.019841025** | **141** |
| NPU FP16 | 142 | 0.348106% | 1.219016% | 0.000181764 | 142 |

新配置相对本轮NPU W8：平均L2下降23.384%，最差L2下降7.280%，通过数增加8。但最大KL从0.01140上升到0.01984；CPU同配置更超过0.02。不能据平均误差下降升级为通用默认，也不能只用NPU恰未超过KL门槛掩盖CPU退化。

NPU新配置剩余16个失败位置，均违反原L2门槛：bird的37/38/41/54，cat的34/35/38/39/48/51/53，fox的23，repeat32的32/41/52/54（0起始）。最高KL位于bird/38，约0.01984；最高L2位于cat/35，约10.11%。均值改善没有消除局部尖峰。

全部30层decoder与KV位于NPU；embedding、final norm、lm_head及指标计算显式在CPU。不是将整个HF模型所有模块迁移到NPU，也不是任意模型仅改设备名即可运行。

## 真实文本NLL/PPL与自由生成

重新下载原评测的固定[WikiText-2 test版本](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)，核对HTTP revision及文本SHA。起始行0/500/1000/1500/2000/2500/3000/3500，各32行，每段取前257个token。数据在rank冻结后读取，没有据其结果调整权重或精度。

64输入token、步长32，每段7窗；第1窗计64目标，后6窗各计最后32目标，保证每段256目标恰好计一次。合计56窗、**2,048个真实目标**，FP64 log-softmax累加总损失后计算token加权NLL与PPL。不是完整WikiText测试集或长上下文成绩。CPU控制首窗与安装校准Linear之前的官方模型逐位一致。

| 配置 | CPU NLL/token | CPU PPL | NPU NLL/token | NPU PPL |
|---|---:|---:|---:|---:|
| FP16 | 3.329926874 | 27.936298767 | 3.329969144 | 27.937479644 |
| 全W8 | 3.333585489 | 28.038694120 | 3.333631410 | 28.039981705 |
| W8＋rank32 | 3.333513475 | 28.036675007 | 3.333532058 | 28.037196037 |

NPU新配置比W8的PPL低约 **0.00993%**，仍比NPU FP16高约0.36%。幅度很小，固定片段之间可能互相抵消；没有统计显著性或模型能力提高结论。

自由生成复用旧8条、未参与拟合/选择的提示，每个配置沿自己的token轨迹greedy解码，EOS停止或最多32新token；最多消费63位置，不强制喂CPU续写。

| NPU配置 | 与CPU FP16完整生成相同/8 | 与同配置CPU相同/8 | 与NPU FP16相同/8 |
|---|---:|---:|---:|
| FP16 | 7 | 7 | 8 |
| 全W8 | 6 | 8 | 5 |
| W8＋rank32 | 6 | 8 | 5 |

新配置与本轮W8 **8/8完整生成完全相同**，没有生成质量提升证据。原小模型答错、格式遵循和截断问题也不能仅归因于低位执行。生成一致率是数值/路径比较，不是任务准确率。

## 实际压缩与DPU证据

原decoder七类投影共106,168,320个元素。准备阶段字节数：FP16为212,336,640；W8 codes+scales为106,479,360；rank32因子额外19,537,920。合计126,017,280，为原FP16投影存储的约59.35%。不包括embedding、lm_head、norm、KV、Python对象或CPU参考。

30份单token decode图常量区汇总：

| 路径 | 常量/字节 | 常量/MiB | INT8矩阵任务 |
|---|---:|---:|---:|
| 全W8 | 113,441,280 | 108.1860 | 840 |
| W8＋rank32 | 135,912,960 | 129.6167 | 840 |
| 全FP16 | 219,240,960 | 209.0845 | 0 |

新配置比全FP16常量少38.01%，比W8增加21.43MiB；旧50%保护的历史decode常量为166,525,440字节，新配置更小，但质量门槛通过数不及旧保护方案。此处没有重跑旧混合策略的NPU评测，不把历史数字当成新控制组。

所有低位配置的每层都具有非elementwise INT8 DPU任务，`amode=0,wmode=2`；补偿的IR包含FP16常量及两次MatMul，既有INT8解压标记保留。840是编译器分块任务数，不是矩阵数或原生4-bit计算证据。

三种完整模型的后端Tensor分配峰值均约19,496,380字节；这不含编译图blob/常量、驱动、分配缓存、CPU原始权重和参考模型。不能据此声称整个模型只占19MB，也没有推理加速或能耗优势benchmark。

## 验证、异常与剩余风险

- 四份正式低位专项真机记录（rank8/16/32探针＋三种完整模型）合计 **97,929次图提交**。每种完整模型32,640次：592个回归输入token、17次回归热重复、56个语言窗口、411个自由生成阶段消费token、8次语言热重复，另120次gamma初始化。
- 三种模式各30个完整64-token窗口的局部CPU/NPU检查。新配置最大局部L2为0.068058%，最大scaled error为0.662145%，均通过原0.3%/1%执行门槛。探针3种rank亦通过；这验证局部计算，不代替整网质量门槛。
- 每种模式17次回归热重复、8次自由生成热重复均为30次图调用，无新增会话或显式payload上传下载；语言重复还核对输出/KV/位置91份哈希和旧seed90份哈希。复制计数不覆盖驱动内部DMA。
- 正式记录退出后context健康，live/cached/quarantined/session归零。默认FP16/W8字节级图/CPU参考控制、不可变因子、错shape/rank/策略/map拒绝均覆盖。
- **203项离线检查通过**：166项根目录（含4项新核心、5项可选模型）、32项npunlock API、5项源码发布工具。仅装核心依赖的CI允许跳过需safetensors的模型项，不下载checkpoint。
- `w8_lowrank_audit.py`复核2,154条标量位置、固定排名/原门槛、210份packed权重与因子、全部语言窗口目标计数及编译模式。没有保存原始logits，因此清理后需要重新执行模型，不能仅凭报告重算词表级KL。

开发期先由离线图测试抓到局部变量`b`遮蔽batch维，已改为明确的因子名称，随后默认/新图检查通过。一次W8控制因20秒编译超时中止，shutdown检查通过，完整新进程重跑成功；不计入正式成功提交数。

**FP16控制另出现一次设备提交错误**：默认512MiB图缓存预算下，142位置回归已完成，在切换至完整64-token语言窗口后遇到`zeFenceHostSynchronize`/`0x70000001`，没有将其记为通过。用小型SHAVE前反向检查确认新进程恢复后，FP16评测进程将图blob预算改为768MiB，完整重跑通过，且无图淘汰；W8和新配置仍用默认512MiB，亦无图淘汰。三类FP16图常量更大与缓存切换是待调查方向，尚未建立因果，没有因此修改生产默认或声称修复设备错误。失败运行没有留下完整最终计数，不混入97,929次正式记录。额外恢复检查为221次提交。

完整重跑之间只调整研究脚本的失败记录和缓存预算参数；15个准备/选择/CPU参考/语言计分/设备计算定义的AST与先前记录一致，摘要为`7ed067ff2de58826f267a784161552b62b91241d166626dc43004cf155f9e46c`。审计保留旧runner指纹作为历史来源，避免把这些流程调整伪装成新的数值优化。

## 用法与复现

已有核心环境和可选模型依赖即可；无需新DLL/PYD或内核重建。因子必须与同次准备的W8权重配套，研究加载器核对权重/因子/checkpoint/源码SHA；公开构造器只验证表示及shape，调用者仍负责配对，不能混用其他checkpoint的同shape因子。

```python
from torch_npu_lowbit_correction import fit_lowrank_correction
from torch_npu_smollm import FrozenSmolStack

# 对每层每个投影，以代表性CPU激活准备；这里示意单个矩阵。
correction = fit_lowrank_correction(original_fp16_weight, packed_weight, cpu_inputs, rank=32)

# 两个map均按checkpoint绝对层编号组织；每层使用投影全名。
stack = FrozenSmolStack.from_checkpoint(
    model_dir, num_layers=30, mode="w8a16",
    packed_weights=packed_by_layer, corrections=corrections_by_layer,
)
```

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPUNLOCK_GRAPH_CACHE=''
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','prepare')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','select')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','independent')
# 此步访问固定公开测试数据，后续不据其结果重新选择rank。
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','data')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','cpu-language')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','npu-probe')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','npu-corrected')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','npu-w8')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_quality.py -ScriptArguments @('--case','npu-fp16','--graph-cache-mib','768')
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_lowrank_audit.py
```

后续应在新的冻结数据上比较更合适的残差目标、量化配置与成本，而不是根据本轮16个失败位置反复调参。扩大INT4/NF4前仍要补齐其独立执行/压缩证据和整网质量；本轮只认证已测的SmolLM2配置、B1/容量64、冻结W8A16与rank32组合，不认证任意模型、低位训练或通用CUDA兼容。


## 构建与来源指纹

沿用66枚resident ELF/汇编、27份C源码及3个原生组件，共162项内容指纹核对通过。未重建内核、Torch PYD或session DLL。模型safetensors SHA256为`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。

本轮主要源码SHA256：

```text
torch_npu_lowbit_correction.py
585642159d23366afcff326b95db8189d4d960509e68e2ab7a6bb6f94d44aec5
torch_npu_smollm.py
25f422fabaa65cac475cc020c613f93d48e9e69e3c23b0a8dc23ad1798135e17
test_torch_npu_lowbit_correction.py
b8102d6d3d9a5331680213bc1af32cfdedcc0095d54cae21250643a306a4e255
test_torch_npu_smollm_correction.py
f43a2407668ac05fe594e058f243fc6fd4c027aa6f276694c84cdcfb8e32ac13
npunlock-probe/research/w8_lowrank_quality.py
5fd23230a5898b1b5f9d50e956424d26b560cea4df6941a5c7dd79305df0bb11
npunlock-probe/research/w8_lowrank_audit.py
77dfd37ad6dd41392dbaad4c00ed5acfaa091c4ee44961f0188417a3d58955d3
.github/workflows/offline.yml
381de02a48fe7b592d86ec0aef86d42d1c3a921ee7972401e3796dd236269646
```

准备权重NPZ SHA256：`f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62`；因子NPZ：`659fcba486d8328ab2f1bac35c04b7695be945fdd35613c2dc2377c17c48c195`。它们是本轮产物指纹，清理后需先重新prepare，不能把旧数值直接套用到其他权重或校准数据。

## 清理与交付

按用户空间策略，删除本轮16份临时文件，共125,634,330字节（约119.8MiB），包括W8权重/FP16因子NPZ、公开测试文本/token清单、标量/设备证据JSON、恢复检查JSON和临时AST对照副本。清理前后1,955份保留源码、内核、报告及原生组件内容指纹一致，随后完善本报告及空间台账。

保留新增CPU准备模块、模型图接入、测试、研究/审计复现脚本、双语首页与本报告；没有新增依赖、环境、模型权重、图表、磁盘checkpoint或持久图缓存。全部测试进程已退出，source-release仍干净且未更新，未推送GitHub。原始JSON/NPZ已删除，复现先运行prepare，再按上面的顺序选择/冻结数据/验证。

