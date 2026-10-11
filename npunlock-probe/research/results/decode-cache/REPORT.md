# Decode 缓存修复、连续生成和 KV ScatterUpdate 试验

2026-10-06，Lunar Lake NPU4000 / Core Ultra 7 258V，8086:643e，
驱动 32.0.100.5540，编译器 8.3，Torch 2.14.1+cpu。

已修复共享 Tensor 分配缓存无法适应形状切换的问题，接入默认 C++ 后端。
缓存上限仍为 16 MiB、64 块，最后一个 Storage/view 引用释放后才可缓存。
形状切换压力下，每 31 次四层 decode 的驱动分配由 403 次降为 0 次。
普通路径和混合生成的速度有较大波动，本轮没有稳定加速证据。
标准 ScatterUpdate 替换 KV Select 的实验跑通，但模式相关收益不一致，没有接入默认模型。

## 根因和实现

原分配器在缓存满时保留已有闲置块，直接释放新归还的块。
即使这些旧块只有约 8 KB，只要占满 64 个槽位，后续不同大小的 KV/输出仍会反复申请。
缓存命中时还以末尾元素覆盖被取走的位置，因此缓存排列不能表示归还时间。

现在需要腾出块数或字节空间时，释放最早归还的闲置块，再保留刚归还的块。
命中时通过 vector erase 保持剩余闲置块的顺序。索引容量仍预留 64 项；
Storage deleter 不分配新的索引内存。只匹配完整分配字节大小，不拆块、合并或复用活跃输出。
执行仍逐图同步，记录的绑定完成后清除；失败上下文仍禁止分配/复用并隔离资源。

生产修改仅为 `npunlock-probe/torch-device/backend.cpp` 和对应重建 PYD/清单。
`torch_intel_npu.py`、模型、RMSNorm C/ELF、原生 session runtime 均保持本轮开始时的指纹。
没有新增 Python 包、OpenVINO runtime 依赖或用户调用参数。
源/二进制快照在 `before/`、`after/`，源码 diff 在 `production.patch`。

旧 PYD SHA256：`25d48cf772022ca7908934483c123e38cb8923ddd86223c00c8d994f1a39ef60`。
新 PYD SHA256：`a46616720e3e204e28f88b2fb1dd709d5efb79b9dbc9efecd34e399543ce592f`。

## 小型复现：缓存满后改变分配大小

先归还 64 个不同大小的小块，再循环申请/释放 B4 的输出、两个 KV 和一个位置张量。
21 轮，共 84 次逻辑分配；该复现不提交 NPU 计算图。

| 指标 | 原版 | 新版 |
|---|---:|---:|
| 驱动分配 | 63 | 3 |
| 缓存复用 | 21 | 81 |
| 保留块数 | 64 | 64 |
| 保留载荷字节 | 8320 | 208780 |
| 每轮申请和释放四个张量，中位数 ms | 0.3509 | 0.0077 |

新版只在首次遇到这些大小时申请，后续复用。载荷字节统计不包括页对齐、驱动和图内存。
原始数据为 `allocation-before/after.json/log`。

## 四层模型：真实输入、原快照语义

原、新、新、原四个隔离进程加载各自实际 PYD、C++ 源和匹配构建清单。
每进程测试相同四层 SmolLM2-135M、真实 tokenizer/embedding、FP16 / W8A16、B1 / B4。
kernel、模型和编译器一致；四个进程记录的模型图 SHA256 完全一致。

每个单步路径预热 5 次，计时 31 次；包含创建返回 Tensor 的逻辑分配。
返回值删除在调用计时外，单独保存 cleanup 样本。
`natural` 路径沿用真实前一组模型/连续解码留下的缓存。
`shape-switch` 路径显式填满旧形状的小块，测试缓存饱和后的切换压力。
该压力条件是构造的，不能代表所有应用。

表格用两个进程各自中位数的算术平均；百分比为耗时下降，负数表示测得更慢。

| 模式 | 形状切换旧→新 ms | 耗时下降 | 每 31 次 decode 的驱动分配，旧→新 |
|---|---:|---:|---:|
| B1 S32 FP16 | 4.0568→2.8896 | 28.77% | 403→0，两轮一致 |
| B1 S32 W8A16 | 4.0569→3.3247 | 18.05% | 403→0，两轮一致 |
| B4 S16 FP16 | 4.0473→3.8111 | 5.84% | 403→0，两轮一致 |
| B4 S16 W8A16 | 5.7079→3.0009 | 47.43% | 403→0，两轮一致 |

| 模式 | 普通单步旧→新 ms | 耗时下降 | 驱动分配，旧→新 |
|---|---:|---:|---:|
| B1 S32 FP16 | 3.4971→3.9035 | -11.62% | 0→0 |
| B1 S32 W8A16 | 3.6481→3.6888 | -1.12% | 0→0 |
| B4 S16 FP16 | 4.2189→4.4097 | -4.52% | 31→0，两轮一致 |
| B4 S16 W8A16 | 4.3845→3.8662 | 11.82% | 0→0 |

另做真正推进位置的 32 次 decode：B1 从 32 到 64，B4 从 16 到 48。
输入预先常驻 NPU；第 1、16、32 步保存输出/KV/位置数组。
保存数组的诊断下载在逐步计时之外。

| 模式 | 连续 32 步的逐步中位数，旧→新 ms | 耗时下降 |
|---|---:|---:|
| B1 FP16 | 2.6194→2.6992 | -3.04% |
| B1 W8A16 | 2.8782→3.0843 | -7.16% |
| B4 FP16 | 2.8291→3.8479 | -36.01% |
| B4 W8A16 | 3.8380→2.7689 | 27.86% |

这项测试的驱动分配均为 27 次，两个版本、两轮一致；并非所有分配都已消除。
未锁定桌面负载、频率或热状态，多个普通路径新版本测得更慢，且进程之间波动明显。
本轮能确认形状切换时重复申请被移出暖循环；不能声称所有 decode 都加速，
也不能把普通路径的差异全部归因于分配器或设备计算。

每个模型进程完成 1760 次 NPU 调用，共 7040 次。
每进程 312 个输出/KV/位置数组与首进程原始字节完全一致，旧 seed 和输出保持不变。
单步计时无新增显式上传、下载或图编译。
原始数据为 `model-0-before`、`model-1-after`、`model-2-after`、`model-3-before` 的 JSON/log/NPZ。

## 固定 32 token 的完整混合生成

CPU embedding → 前四层 NPU → 后 26 层 CPU → CPU norm/lm_head、贪心选词。
这是明确的混合流程，不是全模型 NPU 生成吞吐。
FP16，新故事提示 24 个输入 token；固定生成 32 个 token，不按 EOS 提前结束。
首次完整运行预热和冷编译，随后每进程测三次；计时包括实际传输、CPU 层及选词。
缓存最终长度为 55，即前缀加 31 个已送入模型的生成 token。

| 进程 | 32 token 全流程中位数 ms | token/s |
|---|---:|---:|
| 原 0 | 967.6625 | 33.07 |
| 新 1 | 1253.6275 | 25.53 |
| 新 2 | 1535.2986 | 20.84 |
| 原 3 | 1720.7441 | 18.60 |

用两个进程中位数的平均耗时计算，原版约 23.81、新版约 22.95 token/s；
新版平均耗时上升 3.74%，不能报告完整生成加速。
排除首步 prefill 后，平均 decode 吞吐分别约 24.45 / 23.74 token/s。
原版两进程之间的吞吐变化已远大于这次平均差异；本测试没有隔离 CPU 层和背景负载的影响。

首步 logits 相对 CPU 原模型误差约 1.41%，保持原有 3% 阈值。
三次运行的生成 token 与第 1、16、32 步 logits 按字节一致；
四个进程的 12 个 token/logits 数组也逐位一致。
每进程完成 528 次 NPU 调用，包括冷初始化；每次计时运行固定 128 次图提交、无新增编译。

新故事提示的 W8A16 在旧运行时测得首步 logits 误差约 4.88%，超过原有 3% 阈值。
没有放宽阈值或删除失败，记录在 `initial-w8-quality/`。
本轮完整生成性能仅验证 FP16，W8A16 四层/连续 KV 对照继续保留。
这一限制在修改前出现，不能认作新缓存策略引入的问题，也不能声称 W8A16 对所有提示均达标。

## KV Select 的标准 ScatterUpdate 候选

依据 [ScatterUpdate-3 官方操作规范](https://docs.openvino.ai/2026/documentation/openvino-ir-format/operation-sets/operation-specs/movement/scatter-update-3.html)，
用数据、位置索引、单 token 更新和 axis=2 表示缓存更新。
它保留完整输出副本语义，因此还需要复制旧 KV，不能宣称变成原地更新或消除全部数据搬运。
候选以独立 Python 模块生成 IR，直接交给原驱动编译器，没有导入 OpenVINO runtime。
生产模型源码保持不变，已有 RMSNorm 载体检查与编译器/设备限制照常执行。

同进程 ABBA / BAAB，21 轮，每种实现 42 个样本，五次预热；普通执行 API，无 task profiling：

| 四层模式 | Select ms | ScatterUpdate ms | 耗时下降 |
|---|---:|---:|---:|
| B1 FP16 | 3.13870 | 2.93415 | 6.52% |
| B1 W8A16 | 3.38900 | 3.73865 | -10.32% |
| B4 FP16 | 4.19555 | 3.96300 | 5.54% |
| B4 W8A16 | 4.34670 | 4.19105 | 3.58% |

52 个输出/KV/位置数组逐位一致，1696 次调用完成、资源归零。
收益不一致，尚未覆盖全部位置、batch 和边界，故未接入默认 KV 图。
候选源、XML、blob、样本和日志保存在 `scatter/`、`scatter-run.log`。
后续应继续检查编译器对 KV 更新的布局/搬运调度，再决定是否采用。

## 回归、证据和复现

- 分配器原有五项检查通过：精确大小复用、特殊位模式及活跃 view、16 MiB/64 块上限、
  四线程 Storage 释放、持有 view 时 shutdown。
- 新增三项归还顺序检查通过：最旧块淘汰、中间命中不破坏顺序、最近大小继续复用。
- 模拟失败上下文检查通过；没有真实硬件故障提交，失败后驱动 free/reset 为零。
- 设备 33 项 / 4160 次调用、训练 11 项 / 2732 次调用、KV guards 23 项 / 55 次调用、
  KV boundary 14 项 / 254 次调用通过。训练精度规则保持原样，没有新放宽。
- 正常进程最终 live/cache/quarantine 均归零；模拟故障按原规则隔离。
- `audit.json` 核对实际源/二进制、未改源码、原始数组、图指纹、计数、计时中位数、
  进程顺序、完整生成和 ScatterUpdate 结果。

最初测量把构造压力缓存带入了后续连续链，现已在链前显式 empty_cache 隔离，
旧采样保存在 `initial-scope-order/`，不用于表格。
生成测试最初把冷初始化也要求为 128 次提交；修正为冷预热单独允许准备 gamma 的调用，
计时运行仍严格 128 次。原失败在 `initial-cold-count/`。
这些都是测量脚本修正，没有修改模型数值容差。

`measurement-script.py` / `generation-script.py` 为完整对照时的源码快照。
此前 RMSNorm/warm-allocation 的 audit 是对应阶段的历史构建证据；
新 PYD 应使用本轮 audit，不能把旧指纹检查当作新版本验证。

在 `E:/Pytorch_xfire_NGPU` 的 PowerShell 中运行：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_cache_audit.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_cache_checks.py
```

重新采样会覆盖本目录结果，应先保留原批次。
用保存的实际旧 PYD 创建第一组，再完成原、新、新、原顺序：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_cache_probe.py -ScriptArguments '--case','model','--runtime','before','--label','model-0-before'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_cache_generate.py -ScriptArguments '--runtime','before','--label','generate-0-before'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decode_cache_control.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\kv_scatter_probe.py
```

SHAVE RMSNorm 的上一轮优化继续生效；本轮没有更改通用 FP32 GEMM 或新增精度支持。
