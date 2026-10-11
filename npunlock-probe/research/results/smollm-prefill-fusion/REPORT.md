# SmolLM2：多 token prefill 的整层单图

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

冻结 SmolLM2-135M FP16/W8A16 的多 token forward/prefill 已默认使用整层单图。B≤8、S≤64 的连续 FP16 NPU 输入沿用原 attention、两次 RMSNorm、FFN 与 KV 更新，每层暖 prefill 提交 4→1 次、输出分配 4→3 次。decode 继续沿用上一阶段的一图／四输出分配路径，调用 API 不变。

## 实现与正确性

生产改动只扩展已有图构造器的多 token 入口和融合选择条件，并保留一个用于同实例对照的私有 prefill 开关。沿用相同的 FP32 SHAVE RMSNorm、Neumaier 补偿求和、epsilon=1e-5 和两次 FP16 舍入；矩乘继续走 FP16/W8A16 DPU。没有改 precision helper、布局白名单、CMX／别名／重定位保护，也没有重建内核或 native DLL/PYD。

256 个真实多 token 图（FP16/W8A16 × B1..B8 × S=2,3,7,8,17,32,63,64 × forward/prefill）全部与原四图基线逐位一致，包括 y 与 K/V。每个 blob 仅修改共享 Power 代码窗口中的 5136 字节，窗口总长 21216 字节；长度、其他代码、数据、DMA、barrier 与任务字节不变。45 种观测到的物理行排列／行分片组合均符合上一阶段完整行校验，不需要新的载体布局。

48 个单 token forward/prefill/decode 图的 IR XML 与权重字节完全未变。另有 8 项非法形状／入口检查，包括 S>64、B>8、多 token decode、诊断 stage 和 CPU reference 的内部 layer 参数拒绝。单 token 的数学和设备图保持原样。

## 同实例暖 prefill 对照

同一模型、输入与驻留参数，仅切换私有 prefill 开关。14 组，各 41 轮 ABBA/BAAB，每策略 82 个样本。输出/K/V/位置计数位模式一致，暖显式主机上传、下载与编译为零。

| 层数 / B / token | 模式 | 四图 ms | 整层单图 ms | 中位延迟降幅 | 更快轮数 |
|---|---|---:|---:|---:|---:|
| 1 / 1 / 7 | fp16 | 3.685 | 1.275 | 65.40% | 41/41 |
| 1 / 1 / 7 | w8a16 | 3.434 | 1.230 | 64.17% | 41/41 |
| 1 / 1 / 32 | fp16 | 3.683 | 2.000 | 45.69% | 41/41 |
| 1 / 1 / 32 | w8a16 | 3.036 | 1.622 | 46.59% | 41/41 |
| 1 / 1 / 64 | fp16 | 3.393 | 2.171 | 36.02% | 41/41 |
| 1 / 1 / 64 | w8a16 | 3.531 | 2.115 | 40.09% | 41/41 |
| 4 / 1 / 32 | fp16 | 8.469 | 4.668 | 44.88% | 41/41 |
| 4 / 1 / 32 | w8a16 | 8.533 | 4.708 | 44.83% | 41/41 |
| 1 / 4 / 32 | fp16 | 3.951 | 2.723 | 31.09% | 41/41 |
| 1 / 4 / 32 | w8a16 | 4.159 | 2.804 | 32.57% | 41/41 |
| 4 / 4 / 64 | fp16 | 23.874 | 18.941 | 20.66% | 40/41 |
| 4 / 4 / 64 | w8a16 | 22.407 | 18.437 | 17.72% | 41/41 |
| 1 / 8 / 64 | fp16 | 9.897 | 8.517 | 13.95% | 41/41 |
| 1 / 8 / 64 | w8a16 | 9.861 | 8.845 | 10.30% | 41/41 |

本轮中位延迟下降 10.3%–65.4%。四层 B1、32 token 的 FP16 为 8.47→4.67 ms，W8A16 为 8.53→4.71 ms；B8×64 的收益约 10%–14%。输入越大，减少提交在总时间中的占比越小。本轮没有设备时间戳或带宽计数，不能据此区分其余时间是计算、DMA 还是等待。

计时包含验证、绑定、提交、设备执行和同步 fence，排除初始加载、编译和显式传输。复制计数不含编译器／驱动内部 DMA。桌面负载、频率和内存状态未锁定；以上是本机暖层片段时间，不能换算为完整 LLM 生成速度或功耗。

## 与 CPU 的完整前缀 forward 对照

21 轮轮转各路径，CPU 为 FP16、8 线程。此表比较仅返回 hidden states 的 forward，NPU 每层一图；它与上表返回 K/V 的 prefill 有不同输出及分配范围，不能混用两表的绝对时间。

| 层数 / B / token | NPU FP16 ms | NPU W8A16 ms | CPU FP16 ms |
|---|---:|---:|---:|
| 1 / 1 / 33 | 1.881 | 1.882 | 1.974 |
| 1 / 4 / 64 | 5.463 | 5.625 | 5.815 |
| 2 / 1 / 33 | 3.720 | 3.791 | 4.559 |
| 2 / 4 / 64 | 9.633 | 9.456 | 12.112 |
| 4 / 1 / 33 | 7.074 | 6.584 | 7.280 |
| 4 / 4 / 64 | 20.730 | 20.617 | 25.093 |

## 原 decode 路径的独立实例对照

decode 的图／权重不变，两版本均每层一图。不同实例具有不同的准备、gamma 和分配状态，下表仍保留更慢的组，不把这些差值归因于图或内核优化。不能从该轮独立实例数据推断未改动的 decode 获得稳定加速。

| 层数 / B | FP16 旧→新 ms | W8A16 旧→新 ms | CPU FP16 ms |
|---|---:|---:|---:|
| 1 / 1 | 1.559 → 1.884 | 1.541 → 1.736 | 1.340 |
| 1 / 4 | 1.414 → 1.754 | 1.347 → 1.576 | 2.111 |
| 2 / 1 | 3.222 → 3.615 | 3.181 → 3.237 | 2.478 |
| 2 / 4 | 2.755 → 3.222 | 2.893 → 2.825 | 3.387 |
| 4 / 1 | 6.039 → 6.547 | 5.989 → 5.865 | 3.868 |
| 4 / 4 | 5.822 → 6.449 | 5.826 → 5.860 | 6.561 |

## 回归与快照

102 条真实权重数值记录、336 个固定输入 K/V 对照、252 次连续位置解码通过。最大执行相对 L2 0.2003%，scaled error 0.6472%；原 0.3%／1% 执行门槛及 3% 量化质量门槛未放宽。24 条新旧连续 decode、8 条普通 forward 与 8 组 CPU+NPU 混合生成保持位模式／生成 token 一致。

14 个多 token 快照检查与两组保留检查验证因果性、batch 隔离、独立 y/K/V、未写入位置的零 padding 和正确位置计数。同一实例 B1×2→B3×7→B2×17→B8×63→B1×64→B4×32→B1×2 切换后，所有旧输出／KV／位置快照不变。除首次参数准备外，新 gamma 广播来自 NPU 驻留参数。

首次快照测试错误地将新长度的 FP32 位置元数据上传计入 gamma 上传，被测试断言拒绝；context_failed=False。修正测试为先单独初始化并记录该不可变标量，再检查冷形状的 gamma 广播；未改生产代码或 warm 零复制门槛。失败源码／报告在 failures/cold-position-counter/。新前缀长度首次初始化仍可能上传一个 FP32 标量，warm prefill 没有该开销。

| 成功设备进程 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
| probe | 8 | 0 | 50 |
| numeric | 40 | 0 | 708 |
| boundary | 14 | 0 | 254 |
| guards | 23 | 0 | 55 |
| hybrid | 8 | 0 | 288 |
| paired | 24 | 0 | 252 |
| regression | 8 | 0 | 66 |
| positions | 44 | 0 | 670 |
| profile | 0 | 8 | 1020 |
| performance | 0 | 6 | 2352 |
| model-probe-fp16 | 128 | 0 | 740 |
| model-probe-w8a16 | 128 | 0 | 740 |
| ownership | 16 | 0 | 66 |
| control | 14 | 14 | 10972 |

成功设备进程合计 18,233 次记录内 NPU 调用，全部健康关闭、存活分配归零。通用 torch 后端、precision helper、SHAVE ELF 与 native SHA 均与上一阶段相同；既有 105 项通用精度、16 项 ELF／载体及 17 项载体保护证据在 smollm-whole-layer/，本轮另验证实际多 token 图与新增入口。

范围仍为冻结 SmolLM2-135M FP16/W8A16、B≤8、总位置≤64、相同前缀长度和无 padding mask。没有增加训练、任意模型、跨层并行或完整 LLM 全 NPU 后端。运行仍不依赖 OpenVINO Python/runtime，仍需要 Intel 驱动图编译器。

## 复现

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','smollm-prefill-fusion','--case','numeric'
.\run_torch_npu.ps1 -Script .\torch_npu_prefill_benchmark.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_prefill_probe.py -ScriptArguments '--mode','fp16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_prefill_probe.py -ScriptArguments '--mode','w8a16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_prefill_ownership.py
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe .\npunlock-probe\research\smollm_prefill_integrity.py
.\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_smollm_prefill.py
```

完整 sources、kernel manifest、before/after、changes.diff、原始／修改 blob 和证据 SHA 在本目录。初步 32 图报告及旧探测源码保存在 preliminary/；最终审计使用分模式的 256 图证据。模型 safetensors SHA：`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。
