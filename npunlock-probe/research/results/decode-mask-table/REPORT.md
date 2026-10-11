# Decode 位置编码与因果掩码共用 Gather

2026-10-09，Core Ultra 7 258V / Lunar Lake NPU 8086:643e，驱动 32.0.100.5540，编译器 8.3。

已接入 `torch_npu_smollm.py` 的默认固定 64-token decode 图。
将 RoPE cos / sin 与因果掩码放入一张 FP16 常量表，按同一位置索引 Gather 一行，再取三个片段。
原来的两次 RoPE Gather 和 `Greater + Select` 掩码构建合为一次 Gather。
KV 更新仍使用原 Equal 和精确半精度位复制 Select，未采用 KV one-hot 表候选。
位置在现有 1..63 有效 decode 范围内，常量掩码保持原来的 +0 / -Inf 位模式。

本项自动作用于该模型已有 decode 路径，无需新的参数或依赖。
没有修改 SHAVE C / ELF、DPU 矩乘、精度转换、native runtime、Torch 扩展或量化策略。
prefill 图不变；通用 ATen 后端不受这项模型图优化影响。

## 消融选择

全部普通图采用交错正反顺序，31 轮、每版 62 个样本，固定输入和同一份只读 KV seed。
只计驻留 decode，不含 CPU embedding / 输出头或边界传输；暖路径驱动分配与显式复制均为零。
两个早期查表候选均不够稳定，未接入默认路径：

| 方案 | 四层 B1 decode，ms | 胜出轮数 | 30 层 B1 decode，ms | 胜出轮数 |
|---|---:|---:|---:|---:|
| 该轮原版 | 1.743 | — | 18.318 | — |
| 只将因果掩码改为独立表 | 1.699 | 17/31 | 17.640 | 19/31 |
| 掩码表 + KV one-hot 表 | 1.715 | 18/31 | 18.105 | 16/31 |

随后选择共用 RoPE / 掩码 Gather，独立配对实验结果如下。
各行有自己的同时段基线，不能用上一表的绝对时间与本表计算增益。

| decoder 层数 / batch / 前缀 | 原→共用 Gather，ms | 延迟降低 | 胜出轮数 |
|---|---:|---:|---:|
| 4 / 1 / 24 | 1.800→1.737 | 3.5% | 20/31 |
| 4 / 8 / 24 | 2.216→2.143 | 3.3% | 24/31 |
| 4 / 4 / 63 | 1.763→1.762 | 约 0.1% | 17/31 |
| 30 / 1 / 24 | 13.081→12.790 | 2.2% | 24/31 |

B1、B4、B8 的四层实验均检查全部 63 个有效 decode 位置，包含第 64 个 token。
B4 使用实际 63-token 前缀。输出 / 所有层 KV / position 逐位一致，原 seed 不变。
30 层对照的 91 个输出 / KV / position 数组逐位一致，90 个原 seed 数组不变。
原始样本：[四层 B1](joint/l4-b1-s24-fp16.json)、[B8](joint/l4-b8-s24-fp16.json)、
[B4 长前缀](joint/l4-b4-s63-fp16.json)、[完整 30 层](joint/l30-b1-s24-fp16.json)。

## 实际设备任务

独立使用 PERF_COUNT 图，四层各自 prefill / decode，重建 query 后每图采样 3 次。
两版共 48 份独立任务记录，普通 / profiling 图以及两版本的输出均逐位一致。
任务 profiling 会引入额外开销，以下不能与普通图墙钟相加来计算瓶颈占比。

| 每层 B1 decode 的任务记录数 | 原版 | 共用 Gather |
|---|---:|---:|
| SHAVE 总数 | 106 | 86 |
| Greater | 12 | 0 |
| Select | 20 | 16 |
| Convert | 4 | 1 |
| Gather | 2 | 1 |
| DPU 总数 | 96 | 96 |

剩余 SHAVE 类别数量均不变。实际 SHAVE 任务减少约 18.9%，没有改变 DPU 工作量。
四层 SHAVE 时间区间并集之和的三样本中位数为 1.312→1.268 ms；这是独立 profiling 观测，
未锁定频率和后台负载。驱动没有提供 DMA 时长，不能据此排除数据搬运开销。
每层 decode IR 常量从 7,095,896 增至 7,104,220 字节，增加 8,324 字节；
该组编译 blob 从 7,781,056 降至 7,764,304 字节。
原始记录：[原版](profile/baseline/b1-s24-fp16.json)、[共用 Gather](profile/joined/b1-s24-fp16.json)。

## 真实生成

30 层 decoder 全部在 NPU；CPU 执行 embedding、最终 norm、lm_head 和 argmax。
B1、24-token 提示，固定生成 32 token，不提前停止 EOS。
预热后 7 轮 ABBA 对照，每版本 14 次完整生成；计时包含 CPU 阶段和边界传输。
切换图在整次生成计时之前完成；两版共享 prefill 图，均使用默认四层分组。
原始样本标签 `1` / `4` 表示原 / 新图，均不是 group_size=1。

| 模式 | 整体 tokens/s，原→新 | 吞吐增加 | 更快轮数 | 首 token，原→新 ms |
|---|---:|---:|---:|---:|
| FP16 | 61.00→63.65 | 4.3% | 5/7 | 24.243→24.242 |
| W8A16 | 69.38→71.78 | 3.5% | 7/7 | 22.290→22.467 |

全部 32 步 logits / token 逐位一致，每次仍为 960 次图调用，decode chain_graphs=930、chain_groups=248。
decoder 阶段平均每步耗时的样本中位数：FP16 13.444→13.121 ms，W8A16 11.851→11.344 ms。
CPU 阶段也有波动，整体吞吐增益不能全部归因于设备优化。
FP16 只赢 5/7 轮，B4 长前缀几乎持平，W8A16 首 token 略慢；不能保证所有负载和阶段都提速。
原始数据：[FP16](generation/fp16.json)、[W8A16](generation/w8a16.json)。

两版每次完整生成均有 86 次驱动分配、无新编译，最终活动 / 缓存 / 隔离缓冲区与会话归零。
此前 RMSNorm 生成记录也都是 86 次，不能将驻留固定输入 decode 的零分配结论套到整次生成上。
本轮初次 wrapper 错误地断言零分配，保留了
[失败记录](generation/fp16-allocation-assertion.json) 和原输出；修正为新旧分配计数一致后完整重跑。
没有借此放宽模型数值 / 生命周期检查，也没有调整生产缓冲池。

## 正式接入与复现

默认实现只有 `torch_npu_smollm.py` 的 decode 图构建发生语义变化。
接入后的 8 项既有回归任务共 74 项通过，涵盖 norm、数值、边界、保护、因果性、混合与堆叠模型。
[promotion.json](promotion.json) 保存前后源码指纹与接入前审计；
[audit.json](audit.json) 核验源码范围、输出、原始任务记录、生成与正式回归。
只保存压缩模型输出、任务记录与小型源码快照，未复制完整模型编译 blobs。

离线核验最新实现：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_mask_audit.py
```

研究入口：`decode_mask_table.py` 对照早期候选，`decode_mask_joint.py` 对照共用 Gather，
`decode_mask_profile.py --name baseline / joined`，
`decode_mask_generation.py --mode fp16 / w8a16`。
已保存的 baseline graph 与完整优化前源码用于复现；不要覆盖本目录的冻结证据，复测应使用新的输出目录。

本轮未重跑通用训练全集，未增加 64-token 以外的模型能力。
W8A16 既有首步 logits 相对 CPU FP16 的 L2 误差仍为 5.69%，超过 3% 门槛；FP16 同项约 0.484%。
图之间逐位一致不等于量化质量问题解决。
下一项值得测量整次生成的 86 次分配具体发生在哪些步骤，验证是否与 128 块空闲缓存上限有关，
再决定是否优化缓冲池；也可继续针对 KV Select 的加载 / 存储调度优化。
