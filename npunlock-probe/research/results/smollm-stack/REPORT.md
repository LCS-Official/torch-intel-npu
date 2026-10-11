# SmolLM2 连续层常驻 NPU：FP16 / W8A16

日期：2026-10-05。实机 Intel `8086:643e`，驱动版本 `1005540`，
PyTorch `2.14.1+cpu`；运行环境为本工作区 `.conda/npunlock`。

已增加 `torch_npu_smollm.FrozenSmolStack`，验证相邻 2／4 层的连续执行。
层间隐藏状态始终为 FP16 `intel_npu` Tensor。它复用已有单层实现，
矩阵走 DPU、RMSNorm 走真正 FP32 SHAVE，未新增依赖、修改内核或设备后端。
每层有 4 张同步执行的图；2／4 层分别有 8／16 次提交。
同一序列的层具有依赖关系，依次执行；没有跨层图融合或异步流水线。

## API 和范围

```python
import torch
import torch_intel_npu
from torch_npu_smollm import FrozenSmolStack

stack = FrozenSmolStack.from_checkpoint(
    r"E:\Pytorch_xfire_NGPU\models\SmolLM2-135M-Instruct",
    start_layer=0, num_layers=4, mode="w8a16")
with torch.no_grad():
    output = stack(hidden_states.to("intel_npu"))
```

`hidden_states` 是起始层的隐藏状态，不是 token ID；返回最后一层的隐藏状态，
不是 logits。权重读取、INT8 编码和 RoPE 常量在 CPU 准备，forward 算术在 NPU。
`reference_cpu` 是显式诊断接口，不由 forward 调用。

已验证 2／4 层、`fp16`／`w8a16`、起始层 0／10／26；构造器接受检查点内
非空连续层范围，其他层数尚未单独验证。输入保持单层契约：FP16 `(B,S,576)`，
B<=8、S<=64、从零开始的完整 causal 前缀。所有子层必须 eval；
训练子层会在任何层执行之前被拒绝。没有 KV cache、padding、自定义 mask、
任意位置或冻结层训练支持，也未验证 30 层全 NPU 生成。

冻结适配器和普通 ATen 后端是不同入口；这里未改变普通 `torch.mm` 的精度或路由。

TF32 追问的资料核查：Intel 的 [NPU4 MAC 架构资料（第 31 页）](https://cdrdv2-public.intel.com/824436/2024_Intel_Tech%20Tour%20TW_Lunar%20Lake%20AI%20Hardware%20Accelerators.pdf#page=31)
列出 INT8 和 FP16；公开资料及当前接入路径没有可用的原生 TF32 矩乘模式。
[OpenVINO 2026 NPU 文档](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
接受 F32／F16 图类型，但说明硬件浮点计算精度为 FP16；图 IO 为 F32 不能证明
DPU 是 FP32 或 TF32 乘法。该说明针对官方图路径，不否定本项目验证的 SHAVE FP32。
TF32 的 8 位指数／10 位尾数计算语义可用于近似加速，但不满足严格 FP32 约定；
参考 [NVIDIA 格式说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)。
本轮没有增加 TF32 模式，也没有用 FP16 冒充 TF32。

## 数值、拒绝条件和生命周期

7 个隔离进程全部通过，合计 69 条记录、12,850 次 NPU 提交。
其中 36 条整段数值记录包括 benchmark 前的 12 条对照；108 条单层对照
使用该层在 NPU 上实际接收到的前驱输出。

| case | 记录数 | NPU 提交数 |
|---|---:|---:|
| probe | 4 | 168 |
| numeric | 16 | 648 |
| boundary | 4 | 168 |
| guards | 21 | 82 |
| causality | 4 | 120 |
| hybrid | 8 | 1,248 |
| performance | 12 | 10,416 |

数值样本包括 B1/S1、B1/S16、B1/S32、B2/S17、B4/S64 与最大 B8/S64。
保持原门槛，没有放宽容差：

| 比较 | 原门槛 | 本轮最大值 |
|---|---:|---:|
| NPU vs 同权重／同量化的分阶段 CPU：relative L2 | <0.3% | 0.1670% |
| 同上：每元素 abs(error)/(abs(reference)+token RMS) | <1% | 0.6472% |
| 分阶段未量化 CPU vs 官方 CPU：relative L2 | <0.1% | 0.0590% |
| NPU vs 官方未量化 CPU：relative L2 | <3% | 2.2775% |

最大量化质量误差来自第 26–29 层、B4/S64 的 W8A16 样本。
执行误差和量化误差分别记录；它们不是同一个指标。
原来的固定容差 `rtol=.01, atol=.005` 在部分输出仍失败，JSON 保留逐项计数。
这不是逐位等价或任意输入准确性保证。

19 个拒绝检查包含非法／bool／浮点层范围、未知精度、CPU／FP32 输入、
形状越界、局部存储视图、父模块训练、最后一个子层训练和活动 autograd。
它们均在执行前拒绝，未产生新提交、编译或主机复制。
另验证完整 dense permutation 的 NPU materialization 和独立模块的热缓存复用。
4 个因果性检查在改变后 8 个 token 时，前 9 个 token 输出保持逐位一致。

热整段调用的显式上传／下载／新会话均为零。诊断中的逐层下载在该测量区间之外。
所有进程在 shutdown 前显式 Tensor 分配归零；shutdown 后 live buffers/bytes 为零，
没有 context failed 或隔离缓冲区。这是有限检查，不是长期运行稳定性认证。

## 真实短文本生成

demo 用一个 Bridge 替换前 2／4 层，整段只上传一次、下载一次；
剩余 28／26 层、embedding、final norm、logits 在 CPU，`use_cache=False`。
每次生成重算完整前缀，因此本实验不测 KV cache decode。

两条提示 × 两种精度 × 两种层数，8 组结果都复现 CPU 的贪心生成 token。
每组最多生成 8 个新 token；其中第二条遇到 EOS 提前结束。
最大 logits relative L2 为 0.5888%，最大 next-token KL 为 0.002058，
分别通过已有 3%／0.02 门槛。这是 smoke test，不是广泛的语言模型质量评估。

## 同进程性能

CPU 8 线程；21 轮交错计时，每轮轮换顺序。权重准备、编译、初始预热和
CPU mask/RoPE 准备不计入。NPU 时间包含 Python 调用、主机提交和同步 fence，
不是纯内核时间。CPU 对照是官方浮点层，不是 CPU W8A16 实现。

表中单位 ms，为暖运行中位数：

| B/S | 层数 | NPU FP16 常驻 | NPU W8A16 常驻 | W8A16 整段 1 对传输 | W8A16 每层 1 对传输 | CPU FP16 | CPU FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/1 | 2 | 5.491 | 5.189 | 5.444 | 5.752 | 1.787 | 1.732 |
| 1/1 | 4 | 10.599 | 10.184 | 10.427 | 11.435 | 3.622 | 3.482 |
| 1/32 | 2 | 7.659 | 7.236 | 7.109 | 7.704 | 4.632 | 3.632 |
| 1/32 | 4 | 14.440 | 13.509 | 13.473 | 14.490 | 9.273 | 7.166 |
| 4/64 | 2 | 17.039 | 16.083 | 16.841 | 17.427 | 18.626 | 14.203 |
| 4/64 | 4 | 29.817 | 29.972 | 30.661 | 32.299 | 26.262 | 25.358 |

“常驻”输入／输出留在设备，计时中没有显式主机复制；“整段 1 对传输”包含
一次 CPU->NPU 和一次 NPU->CPU；“每层 1 对传输”包含相同段前／段后传输，
另外在各层之间往返 CPU。后两列是比较层间复制成本的对应工作流。
三条 NPU 路径的算术和图提交数相同，计数器分别确认 0／1／层数 次上传和下载。
复制计数只覆盖显式 Tensor 复制，排除驱动 DMA 和编译器内部行为。

W8A16 整段传输对比逐层传输，本轮中位延迟减少约 3.4–8.8%；
B4/S64、4 层约减少 5.1%。个别包含传输的中位数略低于常驻中位数，
说明毫秒级差异存在系统／调度波动，不能把所有差值解释成精确复制成本。
各路径的 21 个原始样本保存在 `performance.json`。

这轮所有尺寸的 NPU 常驻 W8A16 都慢于 CPU FP32 参考；不能宣称已有速度优势。
减少层间复制是有效的接入方式，但还不是这些小层段延迟的主要突破。
2→4 层时间大致随层数增长，当前也没有跨层融合的收益。
后续优先验证 KV cache、prefill/decode 两条路径，再研究 W4A16；
若进一步优化延迟，应依据分段测量处理 norm 和图提交开销。

## 内存统计边界

B4/S64、4 层编译图的常量 section 合计：FP16 30,235,648 bytes，
W8A16 16,079,872 bytes；最大 B8/S64 为 31,710,208 与 17,542,144 bytes，
约 30.24／16.73 MiB。这里只计算该层段的 attention/FFN 图常量，不是进程
所有缓存的总常量，更不是 NPU 总物理内存占用。

boundary 进程显式共享 Tensor 分配的累计峰值为 8,856,576 bytes（约 8.45 MiB）。
它包含输入、输出、中间值及缓存 gamma，排除编译器常量、CMX、驱动内部内存，
也不包含 CPU 模型和 Python 权重快照。不能用这个数字推断整体峰值或 17.9 GB
设备预算。普通每 Tensor 1,048,576 元素和最多 512 张缓存图的限制保持不变。

## 复现与来源

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_stack_demo.py -ScriptArguments '--case','probe'
# 依次运行 numeric / boundary / guards / causality / hybrid / performance。
# 不要同时启动多个 NPU 测试进程。
```

`summary.json` 汇总所有结果，独立 case JSON/log 保留详细误差、DPU 描述、
图哈希、复制计数、依赖版本与原始计时；NPZ 保留实际输入和三种输出。
`before/` 保存修改前适配器、两份文档，以及 probe/numeric 使用的 runner。
前两轮后只在 runner 中补充了整段传输的 benchmark 对照及峰值范围说明；
生产适配器的哈希在所有 case 中一致。原来的单层研究结果没有被覆盖。

检查点为 `HuggingFaceTB/SmolLM2-135M-Instruct`，固定 revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`。本轮重新校验 safetensors SHA256：
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。
离线加载、不执行远程模型代码、不导入 OpenVINO runtime。
默认 RMSNorm ELF、基础后端、session DLL 和设备扩展保持上轮版本，
所有 source/build/kernel 哈希与模型来源记录于 JSON。
