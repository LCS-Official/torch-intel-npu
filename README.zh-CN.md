# PyTorch backend for Intel NPU4000

<p align="center">
  <a href="./README.md"><kbd>English</kbd></a>&nbsp;&nbsp;
  <a href="./README.zh-CN.md"><kbd><strong>简体中文</strong></kbd></a>
</p>

让 Lunar Lake 的 Intel NPU 执行 PyTorch Tensor 运算、小网络训练与显式适配的模型推理。
项目基于 [npunlock](https://github.com/hsfzxjy/npunlock)，实现 `intel_npu` 设备后端、自定义 SHAVE 内核、
冻结模型的 DPU 图执行，以及缓冲区、会话和编译图复用。

**版本：v0.2.0 Experimental Preview，更新于 2026-10-27。**
本版纳入截至S45的兼容性、容量、AMP、Attention及模型改进；安装与真机验证范围见[发布验收](RELEASE_READINESS.md)。
发布状态见 [GitHub Releases](https://github.com/LCS-Official/torch-intel-npu/releases)。
离线检查、真实NPU验证与GitHub托管CI分别记录，不互相替代；本次结果见[发布验收](RELEASE_READINESS.md)。
目前适合 NPU 算子与模型适配研究，尚未达到通用 CUDA/XPU 后端的兼容性。

[安装说明](INSTALL.md) · [兼容性矩阵](COMPATIBILITY.md) · [性能实测](#性能实测) · [未来工作](#未来工作) · [来源与引用](REFERENCES.md)

本版源码验收见 [发布验收](RELEASE_READINESS.md)，版本正文见 [RELEASE_NOTES.md](RELEASE_NOTES.md)。
报告问题与贡献请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

**v0.2.0之后的S48研究：**[量化前缀上的down／输出校准](npunlock-probe/research/results/w8-output-quality/REPORT.md)实现并评测8种候选，包括固定输入基的约束回归。内部最佳候选未带来稳定设备收益：KV通过数不变，NPU困惑度略升，故不替换默认方案或放宽质量门槛；报告保留CPU/NPU、KV、文本和生成对照。

**v0.2.0之后的S49研究：**[设备数值诊断与NPU内筛](npunlock-probe/research/results/w8-device-quality/REPORT.md)使用真实中间输入核对down补偿链，八条内部轨迹实际在NPU评测。五层取样中补偿相加逐项相等，较大差异位于门控输入及主W8投影。硬件仍选中S48同一候选，独立质量不足以替换原方案；默认配置和质量门槛保持原值。

**v0.2.0之后的S50研究：**[gate/up/SiLU拆分](npunlock-probe/research/results/w8-gate-quality/REPORT.md)在真实forward/decode输入上区分投影、激活和乘法差异。SHAVE激活原型使20处局部FFN更接近CPU参考，但整网内部最大KL略升，预热提交30→660，未通过预设晋级规则；生产默认保持原值。

**v0.2.0之后的S51研究：**[受限W8 scale／低秩联合校准](npunlock-probe/research/results/w8-scale-quality/REPORT.md)使校准投影SSE下降0.889%，但内部NPU通过243→242/290；全FFN候选触发原局部精度门槛。共享前缀重放将一个可复现峰值定位到attention边界的传播，不能推广为整网统一原因。候选未晋级，默认保持原值。

**v0.2.0之后的S52研究：**[Attention分段诊断](npunlock-probe/research/results/w8-attention-quality/REPORT.md)复现已知峰值，并在NPU上逐项验证QK、softmax和PV替换。FP32 softmax降低取样的局部L2，但整网内部通过243→242/290，预热提交30→240；原型未晋级。下一步针对真实NPU Q/K/V输出校准，以原FP16模型整网质量为目标。

S31优化[损失函数与算子容量](npunlock-probe/research/results/operator-capacity/REPORT.md)：硬标签CE/NLL按完整类别行分块，多packet训练不再保存整批稠密mask；百万行NLL仅用43张缓存图，完整梯度与CPU一致。普通Conv2d输出、原生Transformer稠密工作区、Half掩码softmax和纯DMA拼接解除旧容量检查，实际覆盖范围及剩余边界见报告。复用原内核，无新增依赖或宿主构建；已纳入v0.2.0。

S30改善[大Tensor与图缓存复用](npunlock-probe/research/results/memory-scale/REPORT.md)：单allocation扩至512MiB，页对齐子范围复用图、逐元素运算按元素数复用图，分块结果直接写回最终缓冲。通过超过旧256MiB的完整梯度对照、2050个偏移和1820种形状的缓存复用检查；1.34亿参数网络完成两步FP32 AdamW，状态2.00GiB、后端峰值2.25GiB，CPU参照范围见报告。仍保留1536图上限、各算子工作区与单buffer存储；约14GiB训练尚未实测认证。更新需重建session DLL和Torch PYD，无新增依赖或SHAVE ELF，已纳入v0.2.0。

S32继续改善[容量与图生命周期](npunlock-probe/research/results/graph-capacity/REPORT.md)：单allocation上限1GiB（FP16/BF16逻辑Tensor最多512MiB），普通算子图缓存满时采用LRU回收，执行调用与已准备的chain持有的图延后释放。自定义冻结图仍保留，全部槽位被这类图占用时仍拒绝。需重建session DLL和Torch PYD，无新依赖或SHAVE ELF；当前验收及未覆盖边界见报告，已纳入v0.2.0。

S33继续完善[冻结模型容量与图缓存](npunlock-probe/research/results/frozen-capacity/REPORT.md)：INT8/INT4/NF4 Linear、Transformer、SmolLM及矩形GEMM图均可在调用中重建并参与LRU；缓存同时限制1536项/512MiB图blob，超预算可重建图临时执行后释放。弱引用工厂不持有模型，链式decode保留正在使用的图。另通过2.69亿参数的两步FP32 AdamW，状态4.00GiB/后端峰值4.25GiB，CPU核对范围为抽样更新，耗时约35分钟；不代表训练加速或14GiB整网认证。已有S32组件无需重建或新增依赖；实测范围及剩余边界见报告，已纳入v0.2.0。

## 两条执行路径

S45修正[校正因子的缓存身份与Smol图族回收](npunlock-probe/research/results/smollm-cache-lifetime/REPORT.md)：默认512MiB下连续两轮FP16前向/KV流程通过；承压时同族失效重建，分组计划预算不足则提前保护，可用torch.intel_npu.set_graph_cache_budget显式调整。旧16个W8校正失败位置已在真实NPU复现，其中14处包含显著整体logit偏移；[两种新校准候选未胜出，新增整网对照仍有退化项](npunlock-probe/research/results/w8-guarded-quality/REPORT.md)，未放宽门槛或新增INT4/NF4整网认证。无新依赖，已纳入v0.2.0；原生释放的具体根因仍待定位。

S44新增[W8整网低秩残差修正](npunlock-probe/research/results/w8-lowrank-quality/REPORT.md)：INT8主投影加少量FP16因子，在SmolLM2-135M的30层decoder真机回归中，平均logits相对误差从2.24%降至1.71%，通过位置从118/142增至126/142；decode常量比全FP16少38%。PPL收益不足0.01%，8条生成与全W8相同，仍有16处未过原门槛，保持默认关闭。无需新依赖或内核重建；FP16控制组缓存预算与设备错误单列说明，INT4/NF4整网未新增认证，已纳入v0.2.0。

S43优化[分块Attention的SHAVE热路径与自动分块](npunlock-probe/research/results/flash-optimization/REPORT.md)：128×256、head dimension 16的预热前后向，FP32从3.59降至1.72秒，FP16从3.71降至1.85秒，约快2倍；每轮图调用减少约70%，Tensor峰值仍比math低86%–87%。优化已写入C内核和后端分块选择，保留IEEE边界及packet预算。此测例仍比math约慢一倍，不是整网或通用训练加速结论；无需新依赖或DLL/PYD重建，已纳入v0.2.0。

S42新增[SHAVE分块Attention与strided NestedTensor实验支持](npunlock-probe/research/results/flash-nested/REPORT.md)：标准SDPA可显式选择FlashAttention式在线softmax/反向重计算，支持有限FP16/FP32、GQA和Q/K/V梯度；默认Encoder的Nested推理、复制、MHA/LayerNorm及Nested SDPA已接通。128×256测例的后端Tensor峰值降低86%–88%，含编译冷测试约慢3倍，尚无速度优势。新增两枚ELF，原生DLL/PYD和依赖不变；dropout、可微mask、jagged/Nested训练及稳态性能仍有限，已纳入v0.2.0。

S41扩展[Attention与原版HF模型迁移](npunlock-probe/research/results/attention-migration/REPORT.md)：补位置arange、整数cumsum/isin/标签padding、空KV拼接及梯度、RoPE部分视图计算与native MHA/Encoder空输入。原版Llama/GPT-2两层随机小配置通过FP16/FP32推理、cache、贪心generate、FP32 top-k采样重放及两步训练/恢复对照；共享参数需按报告使用PyTorch交换转换选项。无需新依赖或原生重建，不代表任意预训练模型、FlashAttention、top-p或长期训练已认证，已纳入v0.2.0。

S40接通[标准GradScaler入口](npunlock-probe/research/results/amp-standard/REPORT.md)：导入后可直接使用`torch.amp.GradScaler("intel_npu")`，包括提前导入别名；通过后端构造适配复用CPU缩放控制与NPU梯度检查/反缩放。支持iterable、Tensor新scale、溢出跳步、累积及状态恢复；仍需FP32主参数和显式`foreach=False,fused=False`。无新依赖、原生构建或ELF，不代表任意模型AMP或长期收敛认证，已纳入v0.2.0。

S39补齐[Attention空维度与随机接口](npunlock-probe/research/results/attention-rng-boundaries/REPORT.md)：SDPA支持空batch/序列/头及零QK/V特征，并对齐mask梯度与Dropout随机数消耗；新增可广播Tensor概率Bernoulli和NPU随机键排序的randperm，支持Generator/out与状态重放。仍受dtype精确表示、工作区和现有排序容量约束，不代表大规模采样加速或通用HF生成。无需新依赖、原生构建或ELF，已纳入v0.2.0。

S36继续扩展[MaxPool与极值容量](npunlock-probe/research/results/pool-capacity/REPORT.md)：普通MaxPool1d/2d按包执行，不再以旧总采样量拒绝；反向按通道平面分组，长轴极值用设备INT64恢复全局索引，并省去不可微索引输出的无用梯度缓冲。该轮单包、单平面及存储边界保留；实测范围见报告。无新依赖或原生构建，已纳入v0.2.0。

S37解除[选择反向的2²⁴单平面元数据限制](npunlock-probe/research/results/selection-capacity/REPORT.md)：大平面使用新增UINT32元数据内核，实测16,777,219元素MaxPool完整前后向和长轴极值，以及16,798,727条梯度累加，FP16/FP32均通过。小路径与原62枚ELF不变，新增5,740字节ELF；无新依赖或DLL/PYD构建。各dtype存储、其他算子前向与吞吐限制仍在，已纳入v0.2.0。

S38进一步完善[字节存储与密集大平面](npunlock-probe/research/results/storage-shape/REPORT.md)：九dtype按实际字节数使用1GiB单buffer预算，FP16/BF16/INT16容量翻倍，BOOL/INT8增至四倍；密集MaxPool在SHAVE生成窗口/索引，并用几何缩小反向扫描。已通过4097×4097完整CPU前后向和1GiB Half完整取负检查，最大形状范围见报告。需重建PYD并同步新内核，无新运行依赖，已纳入v0.2.0。

S35优化[普通Conv2d工作区](npunlock-probe/research/results/conv-capacity/REPORT.md)：按窗口/fold工作量自动选择有界路径，结果逐块写回，减少完整im2col、CPU索引缓存和结果块堆积。实测3×3卷积后端分配峰值FP32为7.00→1.84MiB、FP16为7.35→1.63MiB；完整前后向和两步训练通过CPU对照。小窗口保留原路径，单buffer及其他卷积边界不变，无新依赖/原生构建，已纳入v0.2.0。

S34继续改善[算子专用容量](npunlock-probe/research/results/index-capacity/REPORT.md)：Embedding可按特征列分块，超出旧完整索引包时复用已有NPU读取/梯度内核；sort/topk输入输出总量保护扩到16,777,216元素，计算包大小不变。具体大词表、长索引、排序与训练验收见报告；保留通用索引反向、卷积/池化等独立边界。沿用现有原生组件和内核，无新依赖，已纳入v0.2.0。

| 路径 | 使用方式 | 当前适用范围 |
|---|---|---|
| PyTorch 设备后端 | `tensor.to("intel_npu")`、`model.to("intel_npu")` | 已支持的 FP16/FP32 算子、一阶梯度和基本 SGD；主要使用自定义 SHAVE 计算 |
| 冻结模型适配器 | `QuantizedLinear`、`FrozenSmolLayer`、`FrozenSmolStack` | 显式生成融合图，用 DPU 执行矩乘；用于 FP16 和部分低位推理 |

**普通 FP16 `torch.mm` 目前仍通过 FP32 SHAVE 路径计算并转回 FP16，不会自动使用适配器的 DPU FP16 矩乘。**
因此，下文模型/DPU 性能不能套用到任意 `.to("intel_npu")` 模型。冻结适配器也不提供 LLM 训练。

S29补充[损失函数兼容性](npunlock-probe/research/results/loss-extended/REPORT.md)：加权NLL/硬标签CE、可微软标签及权重CE、广播和混合FP16/FP32 MSE，以及原生forward/backward/out接口。包含空维度、特殊权重、双方梯度与三步组合训练的CPU/NPU对照；复用现有内核，无新增依赖。硬标签packet、精确预分配out和一阶梯度等边界保留。这些本地改动已纳入v0.2.0。

S28继续扩展[SDPA与GQA支持](npunlock-probe/research/results/sdpa-extended/REPORT.md)：支持非空rank2–64、不同rank广播，以及GQA/MQA的前向和共享K/V梯度；attention矩阵检查扩大到当前存储上限，实际验证超过旧百万元素边界。复用既有SHAVE/DMA，无新依赖；仍保留每行65536、因果query长度1048576、单缓冲区及图缓存限制。这些本地改动已纳入v0.2.0。

S27继续扩展[矩乘与模型支持](npunlock-probe/research/results/matmul-workspace/REPORT.md)：长K矩乘改用滚动补偿累加，避免堆叠全部分块结果；`matmul`支持rank1–64及有界非连续只读视图，补齐空批次/零K的`bmm`。65539维Linear与5维matmul组合网络通过FP16/FP32独立CPU/NPU两步训练对照。普通FP16矩乘仍走FP32 SHAVE；容量、图缓存和算子专用边界保留，已纳入v0.2.0。

S26继续扩大[大Tensor与模型支持](npunlock-probe/research/results/capacity-model/REPORT.md)：单缓冲区上限256MiB、存储rank0–64；扩大普通数学/归约/Linear与softmax行分块范围，并验证约1704万参数Linear的独立CPU/NPU训练。各dtype逻辑元素上限、算子工作区与1,536图缓存边界仍保留；已纳入v0.2.0。

S25新增[容量、索引和dtype兼容性](npunlock-probe/research/results/compat-foundation/REPORT.md)：单缓冲区最多64MiB、存储rank0–8；动态布尔/高级索引、可写切片、gather/scatter、整数常用运算、BF16部分运算、混合dtype拼接及sin/cos/pow等已接通。210万参数网络已做FP16/FP32各两步独立CPU/NPU训练对照。算子专用工作区、图缓存与模型限制仍保留；这些本地改动已纳入v0.2.0。

默认运行不需要 OpenVINO Python/runtime 包，仍依赖 Intel NPU 系统驱动及其图编译器。
设备通过 PyTorch PrivateUse1 注册为 `intel_npu:0`，没有合并进 `torch.xpu`。

## 快速开始

已验证平台为 **Windows x64 / Core Ultra 7 258V / NPU `8086:643e`**，
驱动 `32.0.100.5540`、图编译器 `8.3`、Python 3.12（本轮3.12.14）、Torch 2.14.1+cpu、NumPy 2.5.3。
目前只对这一设备/驱动组合完成验证；准备 MSVC x64 与 Windows SDK 后，在源码根目录运行：

```powershell
conda create -p .\.conda\npu --no-default-packages --override-channels -c conda-forge python=3.12 pip -y
.\.conda\npu\python.exe -m pip install --no-cache-dir -r requirements-build.txt
# 在 x64 Visual Studio Developer PowerShell 中构建
.\build_npu.ps1 -Python .\.conda\npu\python.exe
.\run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\tools\smoke_install.py
```

独立 MSVC 工具链可通过 `-Toolchain` 指定，完整步骤见 [INSTALL.md](INSTALL.md)。
源码构建生成宿主 DLL、worker、session 接口及 Torch PYD；默认使用随源码提供、带哈希校验的 SHAVE ELF。
修改 SHAVE C 另需外部 MoviTools。当前没有可直接 `pip install` 的项目 wheel。

| 依赖清单 | 直接依赖 | 用途 |
|---|---|---|
| [requirements.txt](requirements.txt) | Torch CPU wheel、NumPy | 核心运算与已支持的训练 |
| [requirements-models.txt](requirements-models.txt) | 核心 + safetensors、Transformers | 检查点、tokenizer 与模型示例 |
| [requirements-build.txt](requirements-build.txt) | 核心 + CMake、Ninja | 从源码构建；构建后可卸载这两个工具 |

以下保存为 `my_test.py`，用同一 `run_torch_npu.ps1 -Python .\.conda\npu\python.exe -Script .\my_test.py` 入口运行：

```python
import torch
import torch_intel_npu  # 在首次 backward 前导入，注册 intel_npu:0

x = torch.arange(17, dtype=torch.float32).to("intel_npu")
print((x + 70000).cpu())

model = torch.nn.Linear(4, 3).to("intel_npu")
inputs = torch.randn(2, 4).to("intel_npu")  # 随机数在 CPU 生成
target = torch.zeros(2, 3).to("intel_npu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(inputs), target)
loss.backward()
optimizer.step()
print(loss.detach().cpu())
```

2026-10-16本地开发版新增普通PyTorch迁移检查：MLP、卷积网络与TinyResNet的FP16/FP32训练、
SGD动量/衰减及显式CPU checkpoint恢复通过。优化器使用 `foreach=False, fused=False`；
本地后续已补齐默认BatchNorm的INT64计数、运行统计与恢复，以及下述有界分类损失；AdamW限定路径见下，Transformer常用层仍有明确缺口，不能承诺任意模型只改设备名即可运行。
复现命令为 `.\run_torch_npu.ps1 -Script .\torch_npu_migration_demo.py`，范围见[迁移验收报告](npunlock-probe/research/results/torch-migration/REPORT.md)。
这些修复已纳入v0.2.0。

新增[INT64与默认BatchNorm验收](npunlock-probe/research/results/int64-batchnorm/REPORT.md)：INT64仅提供有界存储与标量计数子集，
现由S24扩展为rank 0–5（含零元素）、最多524,288元素，NPU计数由SHAVE整数加法/进位执行；没有整数到FP32的数值转换。
`BatchNorm2d`默认running buffers、累计平均、FP16/FP32训练、eval/reset和显式CPU checkpoint恢复已通过有限检查。

新增[分类训练验收](npunlock-probe/research/results/classification/REPORT.md)：`log_softmax`、`NLLLoss`和`CrossEntropyLoss`的FP16/FP32前向与一阶反向。
分类输入当前支持rank1–5；NLL/硬标签CE使用同设备INT64标签与可选类别权重，CE另支持同shape浮点软标签及标签/权重梯度。CE支持`label_smoothing`在[0,1]、三种reduction，硬/软标签的`ignore_index`和mean归一化差异见顶部S29报告。
`log_softmax`当前支持rank1–64任意合法轴，最多67,108,864元素、归一化轴宽<=65,536。硬标签NLL仍要求`目标数*(C+4)<=65,536`。
损失计算与标签检查在NPU，非空硬标签前向同步读取4字节错误标志；不使用CPU张量算术回退。
同时修复纯FP16 BatchNorm保存统计的舍入边界；六种小模型/精度组合完成四步独立CPU对照与CPU checkpoint恢复，仍不认证任意深层FP16训练。
标准调用如下（`logits`为模型的NPU输出）：

```python
labels = torch.tensor([0, 1], dtype=torch.int64, device="intel_npu")
loss = torch.nn.CrossEntropyLoss()(logits, labels)
loss.backward()
```

2026-10-18本地新增[AdamW验收](npunlock-probe/research/results/adamw/REPORT.md)：普通单Tensor更新、动量/二阶矩、AMSGrad、maximize和显式CPU checkpoint恢复。
显式使用`foreach=False, fused=False, capturable=False, differentiable=False`及Python数值超参数；参数和动量计算在NPU，step计数与偏差修正沿用PyTorch的CPU/Python控制流。
当前推荐FP32训练；卷积紧邻BatchNorm时显式设`bias=False`，后端不会自动改写模型。
纯FP16的独立更新检查通过，但整网独立CPU轨迹仍有失败；默认`eps=1e-8`还会舍入为零，零梯度时CPU/NPU都可能出现NaN。
FP16测试显式使用`eps=1e-4`，这仍不能保证整网轨迹一致；没有自动提高epsilon或改用FP32动量，该旧测试未使用后续S17的AMP。

2026-10-18本地新增[LayerNorm与GELU验收](npunlock-probe/research/results/norm-activation/REPORT.md)：
`nn.LayerNorm`/`F.layer_norm`和`nn.GELU`/`F.gelu`的FP16/FP32前向与一阶反向，支持GELU的`none`与`tanh`两种公式。
LayerNorm复用NPU的FP32统计与逐元素运算；GELU使用四个自定义SHAVE内核，FP16输入在设备内转换后计算。
含这两层的残差前馈小网络已完成SGD的FP16/FP32及AdamW的FP32四步CPU轨迹对照。
随后完成[混合LayerNorm统计修复](npunlock-probe/research/results/layernorm-mixed/REPORT.md)：
FP16输入配FP32 weight/可选bias的独立前后向通过，归一化宽度最多65,536；原失败重放及22组扩大检查通过。
后续[混合训练更新修复](npunlock-probe/research/results/mixed-training-rounding/REPORT.md)纠正了Tensor `add_`的FP16 alpha转换，
默认与tanh两种GELU配显式FP32分类损失的混合网络均完成四步独立CPU对照，保留原FP32参数梯度阈值。
FP16损失的两组严格整网对照仍失败，不承诺通用混合训练、长期收敛或AMP。
后续[FP64分类对照与32步边界](npunlock-probe/research/results/classification-precision/REPORT.md)中，10组精度检查通过，
两个固定四样本网络均正常学习；但严格CPU轨迹分别从第6/第5步出现偏差，32步检查明确未通过，四步结论不推广为长期一致。
随后[第5/6步归因与独立验证集](npunlock-probe/research/results/mixed-step-diagnosis/REPORT.md)确认，SGD融合/分步乘加及half尾部舍入足以解释原两个失败点；
同输入局部算子与动量重放一致。64训练/64独立验证样本的合成任务中，CPU/NPU验证准确率同为75%、预测全部相同；严格轨迹失败仍保留。
这不表示完整Transformer或AMP已经支持，已纳入v0.2.0。

随后新增[softmax、bmm与小型attention训练验收](npunlock-probe/research/results/attention-primitives/REPORT.md)：
标准`torch.softmax`/`nn.Softmax`和三维`torch.bmm`的FP16/FP32前后向、无梯度out与分块检查通过。
softmax在S19支持rank 1–4的任意合法正/负轴，最多1,048,576元素、归一化轴宽<=65,536。bmm要求相同批次与dtype、不广播，遵守全局存储上限。
`bmm → softmax → bmm`的Q/K/V梯度及含LayerNorm、Linear、残差的小网络四步SGD对照通过，分类损失显式FP32。
计算复用SHAVE FP32及设备内转换；这是兼容性扩展，尚未加速batched GEMM。高维matmul与SDPA的后续范围见下。

本地后续新增[高维matmul与SDPA验收](npunlock-probe/research/results/sdpa/REPORT.md)：`matmul`/`@`支持rank 2–4的批次广播及一阶梯度，`mm`仍限二维。
标准`F.scaled_dot_product_attention`支持同rank 3D/4D的FP16/FP32 Q/K/V、`dropout_p=0`、因果或同设备浮点加性mask，含mask梯度与全屏蔽行零输出。
新增可微clone/contiguous布局整理，保持密集stride或转成连续存储；含双头因果attention、LayerNorm、GELU和残差的网络通过四步CPU训练对照，损失显式FP32。
S13当时将SDPA分数总量由65,536扩至1,048,576元素；S28进一步扩展到当前存储检查并接通GQA，具体rank、容量和训练边界以顶部S28报告为准。后续布尔mask、Dropout和标准模块的有界支持见下；未认证任意完整Transformer。
这是基于SHAVE和现有矩乘的数学实现，尚未进行性能优化，不等同于FlashAttention或DPU融合attention。

本地后续补齐[向量与标量迁移接口](npunlock-probe/research/results/vector-migration/REPORT.md)：`matmul`/`@`扩至rank 1–4，新增有界`dot`/`mv`的一阶梯度和out；FP16/FP32 `loss.item()`同步读取一个标量，保持后续反向可用。
进一步新增[有界索引与Embedding](npunlock-probe/research/results/embedding/REPORT.md)：设备端INT64索引、`index_select`/普通`nn.Embedding`的FP16/FP32前后向、重复索引累加及padding_idx，使用新的SHAVE内核；含Embedding的FP32分类网络通过四步独立CPU训练对照。
初版packet最多65,536个FP32元素，并有反向工作量限制，尚不适合大词表。
随后新增[拼接与切片整理](npunlock-probe/research/results/layout-migration/REPORT.md)：同dtype FP16/FP32/INT64的有界`cat/stack`、部分view和零stride广播的只读设备DMA整理，以及FP16/FP32切片/select、split/unbind的一阶梯度。
切片保留共享storage/version；需要完整独立数据时使用`clone()`。不支持部分目的写入、空Tensor或混合dtype拼接。
进一步新增[BOOL与mask验收](npunlock-probe/research/results/bool-mask/REPORT.md)：真正1字节BOOL存储及转换、六种比较、布尔逻辑、三参数`where`、`masked_fill`及浮点一阶梯度，SDPA布尔mask中True表示保留。
带INT64 padding判断、布尔因果mask、SDPA和masked_fill的普通FP32 attention小网络完成四步独立CPU训练对照。比较保留subnormal/NaN与INT64双字语义，条件选择不通过乘法模拟。
随后新增[RNG与Dropout](npunlock-probe/research/results/rng-dropout/REPORT.md)：SHAVE上的Philox随机生成、FP16/FP32 `rand/randn`、种子/状态恢复、`fork_rng`与普通`nn.Dropout`一阶梯度。带Dropout的FP32小网络通过相同mask的CPU梯度/更新对照及checkpoint下一步逐位重放。
后续[SDPA内部Dropout](npunlock-probe/research/results/sdpa-dropout/REPORT.md)将设备随机mask用于注意力概率，支持FP16/FP32的Q/K/V及浮点mask一阶梯度，保留原有广播、因果与全屏蔽行语义。SDPA按传入概率应用Dropout，eval时调用者需明确传`dropout_p=0`。
显式Generator和离散采样的后续有界支持见S16，有限AMP接入见S17；动态布尔索引、通用AMP及异步执行仍有缺口，也不提供CUDA专属接口兼容；BOOL all/any后续有界支持见下，详见[迁移优先级](COMPATIBILITY.md#cuda式模型迁移仍缺什么)。

本地新增[标准MHA/EncoderLayer迁移](npunlock-probe/research/results/standard-modules/REPORT.md)：复用设备DMA整理打包Q/K/V的只读切片，补齐baddbmm及一阶梯度，并以现有NPU算子承接默认推理入口；不要求用户修改标准模块或关闭全局MHA开关。10组真机检查通过，含FP16/FP32前后向、默认推理及两层小网络的FP32训练/checkpoint重放。这是有界小模型兼容性扩展，不等于新增了硬件融合内核或完整Transformer支持。

后续[标准TransformerEncoder稠密容器](npunlock-probe/research/results/transformer-container/REPORT.md)通过两层FP16/FP32前后向与稠密推理检查，补齐triu/tril及BOOL all/any，补充FP16 native masked-softmax舍入兼容内核；接通Embedding、可学习位置、两层Encoder、padding感知池化、分类头与CrossEntropy的FP32训练/SGD动量更新，并检查新模型/新优化器中的CPU checkpoint重放。padding推理显式使用`enable_nested_tensor=False`；自动NestedTensor路径、encoder-decoder Transformer及通用AMP仍不支持。本轮未做性能或长期收敛评测。

随后[扩大标准模型验收](npunlock-probe/research/results/transformer-scale/REPORT.md)把同一token分类架构扩大至42,211和232,387参数；两种规模的FP32及23万参数的FP16模型均通过全部参数梯度、SGD动量更新及新模型/优化器中的逐位断点重放，分类损失显式使用FP32。修复实际遇到的Embedding容量阻塞：自动按列拆分索引packet，复用既有SHAVE内核与设备DMA，保持重复token梯度的累加顺序；2048×35的非整除分块通过FP16/FP32逐位前后向对照，2048×64词嵌入用于扩大模型。通用张量容量及单列索引限制仍存在，没有新增依赖或放宽数值容差。

新增[完整行attention分块](npunlock-probe/research/results/attention-rows/REPORT.md)：softmax/log_softmax及SDPA分数总量扩至1,048,576元素，单行/key仍<=65,536；因果内核携带全局query行偏移，保留跨head和尾块位置及原RNG语义。标准两层模型扩大到sequence96、batch4、147,456分数，完整训练/恢复各精度结果及内存实测见报告。编译图会话上限调整为1,536项并保留至shutdown，磁盘缓存仍默认关闭。仍存完整分数矩阵，不是FlashAttention或任意长上下文支持。

本地[转换图复用](npunlock-probe/research/results/cast-reuse/REPORT.md)让元素数相同的FP16↔FP32转换共享图，保留原始形状、IEEE转换及设备内DMA压紧。该优化减少首次编译和缓存压力，不减少每次转换的提交数，也没有增加CUDA运行时或通用DPU训练支持。

随后优先补齐[常用数学与模型迁移](npunlock-probe/research/results/unary-math/REPORT.md)：新增公开exp/log/tanh/sigmoid/SiLU/rsqrt及sqrt的一阶梯度，接通标准RMSNorm；FP16/FP32门控网络与有限标准TransformerDecoder的训练/恢复/推理检查见报告。Decoder复用已有自/交叉attention算子，未新增DPU训练或CUDA运行时兼容。

S16继续补齐[离散随机数与采样](npunlock-probe/research/results/discrete-rng/REPORT.md)：INT64 `randint/random_`、FP16/FP32 `multinomial`及原生`torch.Generator(device="intel_npu")`，包含独立随机流、state/clone和生成token的小模型训练/checkpoint验收。随机值与采样在SHAVE执行，multinomial每包只回读4字节校验标志；out要求已有精确shape，不放回要求足够的正权重类别，详见兼容性矩阵。没有新增依赖；从旧本地版本更新须重编Torch PYD。

S17新增[有限AMP](npunlock-probe/research/results/amp/REPORT.md)：标准`torch.autocast("intel_npu",dtype=torch.float16)`与NPU专用`torch.intel_npu.amp.GradScaler()`。保留FP32模型参数，矩乘/卷积使用FP16，损失和归一化等保留FP32；梯度检查和反缩放在NPU，每个优化器回读一个BOOL标志，主机仅维护缩放/增长控制状态。支持溢出跳步与checkpoint；MLP严格训练对照通过，卷积严格FP32梯度仍有限制，详见报告。S40随后接通标准`torch.amp.GradScaler("intel_npu")`；BF16 AMP、任意模型迁移和低位整网质量仍未补齐；有界动态索引与randperm后由S25/S39补齐。无新增依赖/SHAVE ELF，更新需重编PYD；这轮不声明训练加速。

S18完成[AMP卷积差异归因与扩大验收](npunlock-probe/research/results/amp-conv/REPORT.md)：原S17网络同初始化/输入/标签的CPU Half权重梯度差异可由逐batch舍入解释，保留现有NPU FP32累加和原严格失败记录。六种局部几何、原SiLU网络和三卷积残差网络的四步训练通过独立FP32累加控制，checkpoint逐位重放；修复空GradScaler checkpoint的异常类型。CPU控制只用于验收，不进入NPU执行路径；这一轮无需重编内核/宿主扩展，没有新增依赖或性能结论。

S19进一步补齐[常用模型迁移接口](npunlock-probe/research/results/portability/REPORT.md)：FP16/FP32分组与深度Conv2d、Conv1d、任意合法轴softmax/log_softmax、多维硬标签NLL/CE、label smoothing及带alpha的rsub。复用现有NPU内核和DMA，分组卷积串行执行；没有新增OpenVINO依赖、SHAVE ELF或宿主重建。普通分割小网络与标准Transformer的有限训练/AMP/checkpoint验收及数值容差见报告；这些本地能力已纳入v0.2.0，也不提供CUDA运行时兼容。

S24继续补齐[视觉和排序兼容性](npunlock-probe/research/results/vision-sort-extended/REPORT.md)：新增五维/空 Tensor 基础、INT64/BOOL 稳定排序及 k=0、kthvalue/median/nanmedian/mode/FP32 quantile；视觉新增 Conv3d/ConvTranspose3d、三维插值和归一化、抗锯齿、3D/自适应池化与 Unpool、padding、shuffle、Unfold/Fold、affine_grid/grid_sample 及一阶梯度。当前验收范围和容量限制见报告；这不表示所有视觉/排序配置或 CUDA 兼容性已经完成。旧本地版本需重编设备 PYD；没有新增 pip 依赖，计算仍走 NPU。

S23新增[视觉与浮点排序](npunlock-probe/research/results/vision-sort/REPORT.md)：1D/2D nearest/nearest-exact/linear/bilinear/bicubic插值、GroupNorm/InstanceNorm、分组ConvTranspose1d/2d，以及稳定sort/argsort、max/min值与索引、更大k的topk。包含有界一阶反向、原地激活组合和CPU对照；新排序SHAVE ELF随源码提供，无新依赖或C++重建。近似平局激活经排序后可能放大整网梯度差异，报告保留严格轨迹失败及固定路由诊断；并非通用CUDA完成声明。这些原缺口现由S24扩展；部分显式scale的旧CPU前后向不一致配置在训练前拒绝，可改用size或recompute_scale_factor=True。

S22新增[MaxPool、原地ReLU与topk](npunlock-probe/research/results/pool-topk/REPORT.md)：标准MaxPool1d/2d、空间INT64索引及重叠反向；非view非叶结果可用`nn.ReLU(inplace=True)`训练；`torch.topk`支持有界FP16/FP32值梯度与设备INT64索引、长行分块、largest/双out。三个源代码内核随源码提供，无新依赖或C++重建。S22的k<=128/工作量限制已有上述S23扩展，k=0/空结果现由S24扩展，容量限制仍在；相等值topk不承诺CPU相同索引顺序，部分/可微view写入仍有限。

S21新增[极值归约、向量范数与梯度裁剪](npunlock-probe/research/results/reduction-norm/REPORT.md)：argmax/argmin返回设备INT64索引，可微amax/amin，0/1/2/正负无穷向量范数，以及标准`torch.nn.utils.clip_grad_norm_(..., foreach=False)`。新增isnan/isinf/isfinite和同dtype零维设备Tensor乘法系数；极值按行/长行分块并在NPU合并原始索引，保留首个tie/NaN位置。沿用现有容量、FP16/FP32和非foreach边界；后续topk见S22，sort、任意p/矩阵范数或通用CUDA支持仍缺，无新增依赖。

S20新增[索引、平均池化与限幅](npunlock-probe/research/results/index-pool/REPORT.md)：有界gather/scatter_add/take_along_dim、AvgPool1d/2d、64-bin以内AdaptiveAvgPool1d/2d、可微minimum/maximum/clamp、Hardtanh/ReLU6及clip_grad_value_。索引保持INT64在SHAVE校验/规范化，scatter保留self起始累加；每个独立索引行同步读取4字节标志。池化复用现有FP32运算，变化的分母网格只属于几何元数据。新ELF/manifest随源码提供，无需新增Python依赖或C++重建；MaxPool、更大索引和通用CUDA生态仍未覆盖，不声明性能加速。

后续[持续训练与功耗实测](npunlock-probe/research/results/sustained-training/REPORT.md)将网络扩大到4层、793,604参数：FP32连续训练182.2秒、6步，NPU平均忙碌91.7%。
独立一分钟复测中Intel PMT的NPU域平均0.525W、整包平均9.82W；Windows共享内存约43MiB。
这是固定合成分类任务的SHAVE训练结果，不代表DPU峰值算力、完整语言模型训练或CPU/GPU能效优势。

另做[同工作量DPU/CPU/GPU能效基准](npunlock-probe/research/results/gemm-energy/REPORT.md)：六tile、四输出DPU图的FP16稳态约10.39TFLOPS，INT8约14.38TOPS，仍未达到258V的47TOPS规格。
1024³矩乘的两轮整包能耗为CPU最省能已测路径121.64mJ、GPU FP16 2.58mJ、DPU FP16 5.44mJ；本轮GPU更快且每次耗能更低。
DPU FP16的NPU域平均约15.16W，这与前述SHAVE训练的0.525W属于不同负载；后台未锁定，不外推完整模型/训练或硬件峰值。

本地开发版新增 `FrozenSmolIO`，用于SmolLM NPU推理的CPU两端，只加载embedding、最终RMSNorm和共享输出矩阵，
无需构造完整CPU decoder。调用示例（`ids`为调用者准备的CPU int64 token IDs）：

```python
from torch_npu_smollm import FrozenSmolStack
from torch_npu_smollm_io import FrozenSmolIO

io = FrozenSmolIO.from_checkpoint(model_directory)
stack = FrozenSmolStack.from_checkpoint(model_directory, num_layers=30, mode='fp16')
with torch.no_grad():
    hidden, kv = stack.prefill(io.embed_tokens(ids).to('intel_npu'))
    logits = io.logits(hidden.cpu())[:, -1].float()
```

仍需先导入 `torch_intel_npu` 注册设备。此入口仅用于已验证的tied SmolLM FP16推理，不替代CPU参考模型或支持训练；
模块本身只依赖Torch与safetensors，tokenizer由调用者提供，已纳入v0.2.0。

## 当前能力

| 功能 | 已实现与验证范围 |
|---|---|
| NPU Tensor | FP16/FP32 存储、CPU/NPU 复制、部分密集转置与布局处理；Level Zero shared allocation |
| 算子与训练 | 加减乘除、部分一元运算/归约、矩乘、Linear、Conv2d、BatchNorm、ReLU、池化、MSE；本地新增有界LayerNorm/GELU、softmax、bmm、高维matmul、SDPA、clone布局梯度、log_softmax/NLL/CrossEntropy及AdamW；有限参数组合的一阶 autograd |
| 真正 FP32 | SHAVE 上的 FP32 算术、补偿归约与矩乘；未用 FP16/TF32 替代这条计算路径，GEMM 性能仍有限 |
| FP16 模型推理 | SmolLM2-135M-Instruct 的完整 30 层 decoder、prefill、KV cache、decode；decoder/KV 在 NPU，embedding、最终 norm、lm_head、采样在 CPU |
| 低位推理 | 显式 `QuantizedLinear` 的 INT8、INT4、NF4 权重与分组实验；W8A16 / W8A8 图路径。INT4/NF4 支持不等于原生 4-bit 乘加，也不等于通用低位 Tensor |
| 运行时优化 | 图融合、原生会话复用、链式提交、缓冲区复用、可选磁盘图缓存；FP16/W8/混合prefill提前命中；本地轻量CPU模型入口 |
| 安装基线 | 宿主组件源码构建、离线测试及真实NPU运算/训练/模型检查；本版具体覆盖见[发布验收](RELEASE_READINESS.md) |

完整能力以[算子/参数/梯度兼容性矩阵](COMPATIBILITY.md)及对应测试为准；表中的模块名称不代表所有参数组合都受支持。
最近的安装验证见 [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)，历史算子/训练细节见 [TORCH_NPU_README.md](TORCH_NPU_README.md)。

下列Sxx报告保留原实验阶段的条件；当前能力与限制以[兼容性矩阵](COMPATIBILITY.md)为准。

### 已知边界

- 当前S38存储支持rank0–64、单缓冲区最多1GiB，按dtype实际字节计量：FP16/BF16最多536,870,912元素，FP32/INT32最多268,435,456，INT64最多134,217,728，单字节dtype最多1,073,741,824。存储上限不等于全部算子或14GiB整网训练已认证；算子工作区仍有独立限制，Smol图承压按S45同族失效策略处理。
- SmolLM 适配器固定 64-position KV 容量，batch 上限 8，仍受形状与工作区限制；不能直接运行任意 Hugging Face 模型或长上下文。
- 纯FP16深层训练仍有数值限制；AMP已接通标准torch.amp.GradScaler("intel_npu")入口，限FP32主参数和已测配置；二阶梯度、跨CPU/NPU可微复制、广泛优化器、任意可微原地操作及重叠目的写入尚未覆盖。
- INT8/UINT8/INT16/INT32/INT64常用算术及转换已接通；BF16目前限存储、转换及部分FP32中间计算的算子。完整dtype/算子覆盖、完整RNG/采样、异步多流和GPU+NPU协同仍缺；不支持的算子报错。
- W8 完整模型校准尚未全部通过既有质量门槛；INT4/NF4 仍以 Linear/小网络实验为主，不能宣称完整低位 LLM 已适配。当前模型评估优先显式选择 FP16。
- CPU/GPU/NPU 共用物理内存不代表零搬运、零同步；统一内存不能消除驱动内部 DMA 和精度/布局转换。

2026-10-15 的[完整模型低位质量研究](npunlock-probe/research/results/w8-sequential-calibration/REPORT.md)
比较了顺序校准和分组 INT8，尚无可启用候选；分组图被当前编译器展开为 FP16，生产默认保持不变。

本地开发后续的[混合精度质量实验](npunlock-probe/research/results/w8-projection-mixed/REPORT.md)
新增默认关闭的 `fp16_projections`：保留 MLP Up/Down 为 FP16，其余五类投影用校准 W8。
真实 NPU 通过位置由 118/142 提高到 129/142，平均 logits 相对 L2 由 2.238% 降至 1.471%；
30 层 decode 图常量比全 FP16 少约 24%。仍未全面达标；后续速度实测见下，此参数已纳入v0.2.0。

随后完成的[误差诊断与逐层敏感度研究](npunlock-probe/research/results/w8-layer-sensitivity/REPORT.md)
确认这13个失败位置的误差能量平均约92.3%来自logits整体偏移，仍保留原质量门槛。
逐层37.5% FP16候选在真实NPU上为126/142，低于原策略的129/142，未升级正式接口或默认。

新的[短上下文PPL与自由生成评测](npunlock-probe/research/results/w8-ppl-generation/REPORT.md)已完成：
WikiText-2固定2,048个目标token、64输入位置下，真实NPU的FP16/全W8/原混合50%/逐层37.5% PPL分别为27.937/28.040/27.938/27.928。
两个混合配置在8条新提示上生成结果相同，但样本不足以认证通用质量；原logits门槛和生产默认保持不变。

## 性能实测

以下是各轮报告中的实测，**不是芯片峰值或通用模型排名**。
各表保留独立的日期、执行路径和计时口径。

### CPU模型与权重准备：进一步减少启动耗时

2026-10-16，新增轻量CPU入口并减少混合50%的重复层构造。相同缓存图、完整30层、batch1/24-token、每种3对新进程：

| 配置 | 原脚本入口到首token s | 新入口到首token s |
|---|---:|---:|
| FP16 | 5.707 | 5.326 |
| 校准W8 | 6.169 | 5.702 |
| 混合50% FP16 | 5.917 | 5.280 |
| 逐层37.5% FP16 | 6.515 | 6.127 |

启动阶段峰值私有提交量约减少169–202MiB。12次完整32-token生成的新旧输出和KV逐位一致；
这是CPU准备优化，未认证热decode吞吐提升。需显式使用轻量入口，完整CPU参考脚本仍保留参考模型。
详细分段、内存口径与复现见[启动准备报告](npunlock-probe/research/results/smollm-startup-prepare/REPORT.md)。

### W8与混合精度：已有缓存的启动

2026-10-16，已将FP16的prefill提前缓存命中扩展到校准W8及显式混合精度。
完整30层、batch1、24-token输入；相同编译blob、每种3对全新进程交替测量：

| 配置 | 原完整IR首token s | 提前命中首token s | 耗时降低 |
|---|---:|---:|---:|
| 校准W8 | 1.525 | 0.723 | 52.60% |
| 混合50% FP16 | 1.701 | 0.824 | 51.55% |
| 逐层37.5% FP16 | 1.660 | 0.797 | 51.97% |

此表从模型准备后计时；脚本入口到首token分别由6.215/6.641/6.581秒降至5.410/5.780/5.732秒。
所有输出和30层KV逐位一致。收益依赖已有磁盘缓存，默认仍关闭；源码指纹变化后缓存需重建一次。
没有据此宣称热decode吞吐提升或低位质量全面达标。实现、失效/损坏检查和边界验证见
[W8提前缓存报告](npunlock-probe/research/results/w8-prefill-cache/REPORT.md)。该改动尚未推送公开仓库。

### 校准W8与混合精度：单请求

2026-10-16，完整30层SmolLM2-135M，batch 1、固定24-token输入与32个预测位置，
四种配置消费同一条参考token轨迹，2次预热、7轮交替取中位数。CPU两端与同步均计入；这不是自由生成质量测试。

| 配置 | 热首token ms | 解码 token/s | 进程生命周期峰值私有提交 GiB |
|---|---:|---:|---:|
| FP16 | 22.794 | 72.717 | 3.078 |
| 校准W8 | 22.651 | 74.987 | 2.188 |
| 混合50% FP16 | 22.977 | 72.975 | 2.685 |
| 逐层37.5% FP16 | 22.944 | 73.859 | 2.561 |

本轮W8解码吞吐约提高3.12%，混合策略的速度差较小；主要收益是减少内存。
峰值包含加载、编译、运行与验收，现有测试路径仍保留完整CPU模型及原FP16权重，不是NPU独占显存。
已有图缓存的新进程，模型准备后首token分别约0.942/1.516/1.621/1.663秒（各2次中位数）；
该轮测量早于上述缓存优化，当时低位路径仍重建prefill IR。缓存默认关闭，临时缓存已清理。
分段耗时、波动、质量限制与复现见[端到端效率报告](npunlock-probe/research/results/w8-efficiency/REPORT.md)。生产默认不变。

### 模型生成：热请求

2026-10-10，Core Ultra 7 258V / Arc 140V / NPU4000，SmolLM2-135M-Instruct 完整 30 层，
固定 24-token 提示词、greedy 生成 32 token；额外预热 2 次，正式 7 次取中位数。
下表单位为 **整个 batch 合计 token/s**，包含 CPU 两端、每步 token 返回主机及同步，排除加载和首次编译。

| 路线 | batch 1 | batch 4 | batch 8 |
|---|---:|---:|---:|
| CPU FP16，8 线程 | 30.668 | 69.064 | 108.509 |
| Arc 140V FP16，eager | 25.446 | 105.455 | 207.850 |
| Arc 140V FP16，SDPA | 32.325 | 122.545 | 196.090 |
| NPU FP16 decoder + CPU 两端 | 72.925 | 167.394 | 288.046 |

batch 8 的 NPU 288.046 token/s 对应每请求约 36.01 token/s，不能当作单请求速度。
这些 NPU 配置与 CPU FP16 参考的最大 logits 相对 L2 误差约 0.74%–1.12%，固定测试中的生成 token 一致；
这不代替完整语言模型质量评测。

CPU/GPU 使用 Transformers 模型，NPU 使用专门融合的适配器，设备放置和 Torch 环境也有差异；
桌面负载与频率未锁定。GPU batch 8 配对复核存在明显波动，未据此承诺稳定排名或速度比。
完整条件、原始数值摘要和质量门槛见 [模型对照报告](npunlock-probe/research/results/model-cross-device-followup/REPORT.md)。

### 首 token：已有缓存的新进程

2026-10-11，同一模型、FP16、batch 1、24-token 提示词，使用相同 30 份 prefill 编译图，
各 3 个新进程交替对照。磁盘缓存均已建立，区别是是否提前命中并跳过 IR 重建。

| 计时口径 | 完整 IR 路径 | 提前缓存命中 | 中位耗时降低 |
|---|---:|---:|---:|
| 模型准备后到首 token | 2.009 秒 | 0.967 秒 | 51.8% |
| Python 脚本入口到首 token | 6.795 秒 | 5.491 秒 | 19.2% |

第二行包含库导入、初始化及模型准备，不含 PowerShell/Python 解释器启动。
**0.967 秒依赖已有磁盘缓存**：同轮无缓存首次 prefill 编译样本约 28.56 秒，不能把缓存命中收益用于首次安装体验。
本次改动没有证明热吞吐提升。详见 [首 token 优化报告](npunlock-probe/research/results/startup-ir-cache/REPORT.md)。

图磁盘缓存默认关闭，需显式设置 `NPUNLOCK_GRAPH_CACHE`；每目录容量上限 512 MiB。
删除缓存后下次需要重新编译，运行命令与空间约定见 [安装说明](INSTALL.md) 和 [空间记录](WORKSPACE_STORAGE.md)。

### 矩乘：指定形状的有效吞吐

2026-10-05，逻辑操作数为 `2*M*K*N`，形状均为 `1024×1024 @ 1024×1024`，冻结权重、FP16 输出。
输入/输出预分配，排除编译和显式上传/下载，包含驱动调用、图内 DMA 与 fence；31 次取中位数。

| 执行路径 | 中位耗时 | 有效吞吐 |
|---|---:|---:|
| DPU FP16 | 0.6733 ms | 3.18949 TFLOPS |
| DPU W8A16 | 0.6308 ms | 3.40438 TOp/s |
| DPU W8A8 | 0.5462 ms | 3.93168 TOPS |

W8A16 只有权重量化，不能把该项称为全 INT8 TOPS；W8A8 已检查实际 8-bit DPU 操作数。
同轮普通 FP32 `torch.mm` 的 256 方阵为 614.8185 ms、约 0.05458 GFLOPS，走 SHAVE 分块与合并路径，
与上述 DPU 路径的精度、形状和计时范围不同。它提示当前通用 FP32 GEMM 仍很慢，不能用于推断整个芯片的 FP32 峰值。
详见 [矩乘吞吐报告](npunlock-probe/research/results/throughput-20261005/REPORT.md)。

## 未来工作

以下是按当前证据排序的方向，不是已实现功能或交付日期承诺。

| 优先级 | 方向 | 下一步与验收依据 |
|---|---|---|
| 近期 | 扩大发布验证 | 本轮源码构建与真机回归见 [发布验收](RELEASE_READINESS.md)；补充长时间压力回归与另一台干净 Windows 安装验证 |
| 近期 | 首 token 与运行时开销 | 剖析缓存命中的 ELF/DPU 重复解析、库导入与模型重复准备；保持同等校验，做交替 A/B 与输出一致性检查 |
| 近期 | 模型覆盖与质量 | 扩展提示词、batch、KV 边界和数值测试；再扩展上下文与模型，W8/INT4/NF4 必须分别通过质量门槛 |
| 中期 | 实际模型热点 | 重新测 RMSNorm/SHAVE 的指令与寄存器开销，处理大 batch 的缓冲区复用；逐项验证端到端收益 |
| 中期 | 通用算子与训练 | 按实际模型缺失算子补齐派发与梯度，改善通用 FP32 GEMM；扩展S17的AMP模型/优化器验收并继续检查数值边界 |
| 长期 | GPU+NPU 协同与分发 | 研究分层/流水线调度、内存和同步成本，再探索自动图分区；改善安装打包，扩大设备/驱动验证范围 |

稳定 v1.0 需要明确且可复现的兼容性、正确性、安装与性能基线，以及更广的模型/环境验证；
当前的单机实验结果尚不足以承诺通用训练或 GPU+NPU 加速。
优化候选只有通过数值检查并取得实测收益才进入默认路径，例如
[四行 SIMD 归约候选](npunlock-probe/research/results/shave-fp32-pipeline/REPORT.md)虽数值通过但更慢，未启用。

## 源码、文档与复现

| 路径 / 文档 | 内容 |
|---|---|
| `torch_intel_npu.py`、`torch_npu_*.py` | 设备后端、运行时、量化、模型适配器与示例 |
| `npunlock/` | 修改后的基础库；上游基准与差异见 [UPSTREAM.md](UPSTREAM.md) |
| `npunlock/tests/fixtures/npu4000/` | 自有 SHAVE C、汇编、ELF 及 manifest |
| `npunlock-probe/torch-device/backend.cpp` | PrivateUse1 原生扩展 |
| `npunlock-probe/research/` | 诊断、构建、优化实验源码与 Markdown 报告 |
| [INSTALL.md](INSTALL.md) / [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md) | 安装命令、依赖与新环境验证范围 |
| [COMPATIBILITY.md](COMPATIBILITY.md) | 算子、精度、梯度、布局、模型支持与发布回归基线 |
| [TORCH_NPU_README.md](TORCH_NPU_README.md) | 历史详细进展；旧路径/实验链接可能不再有效 |
| [SOURCE_RELEASE_REPORT.md](SOURCE_RELEASE_REPORT.md) | 精简源码、来源整理、快照验证与导出边界 |

源码快照约15.5 MiB，逐文件大小与SHA-256记录在 `SOURCE_MANIFEST.json`。
模型、环境、SDK/工具链、宿主二进制及驱动生成的 `.blob` 不打包；完整夹具回归需要另备图数据。
历史原始数组、图缓存和计时 JSON 已按存储策略清理，报告保留测量摘要；复测需要重新生成实验输入/输出。

<details>
<summary>离线测试与源码快照校验</summary>

安装核心依赖后可运行无需硬件的离线回归；可选模型依赖缺失时会明确跳过相应测试。本版通过与跳过数量见[发布验收](RELEASE_READINESS.md)：

```powershell
$env:PYTHONPATH = "$PWD;$PWD\npunlock\python"
python -B -m unittest discover -s . -p 'test_torch_npu*.py'
python -B -m unittest discover -s npunlock/tests/python -p test_python_api.py
python -B tools/test_source_release.py
```

首次构建前，在导出目录运行 `python -B tools/export_source_release.py --verify .` 校验文件集与哈希。
构建产生本地文件后，严格快照校验会报告额外文件。开发工作区含上游 Git 元数据时可重新导出：

```powershell
python -B tools/export_source_release.py --output source-release-next
python -B tools/export_source_release.py --verify source-release-next
```

导出器拒绝覆盖已有目录，快照不包含嵌套 `.git`。历史研究audit可能依赖已清理的数据，不属于上述离线回归。

</details>

## 许可证与引用

新增代码采用 Apache-2.0，与 npunlock 上游一致；第三方改编部分保留对应许可和版权声明。
**npunlock 是直接基础**；OpenVINO 的部分公开实现与 Intel NPU compiler/ELF/Level Zero 源码也有明确引用关系，
详见 [REFERENCES.md](REFERENCES.md)、[references.bib](references.bib)、[NOTICE](NOTICE)、
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE](LICENSE)。
引用本项目时请记录实际版本/提交并同时标注被采用的上游工作；文献列表不能替代许可证声明。
