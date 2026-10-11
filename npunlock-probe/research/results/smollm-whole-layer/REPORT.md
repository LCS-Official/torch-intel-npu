# SmolLM2：两图合为整层单图

2026-10-05，Lunar Lake 8086:643e，驱动 1005540（32.0.100.5540），编译器 8.3。

冻结 SmolLM2-135M FP16/W8A16 的 B1..B8 单 token forward/prefill/decode 已默认把两次 RMSNorm、attention、FFN 和 KV 更新放入同一图。每层暖 decode 从 2 次提交／5 次输出分配降到 1 次／4 次，四层从 8 图降到 4 图。移除了 attention→FFN 间 DDR 中间张量的独立分配与绑定；图内具体 DMA 由编译器调度，没有宣称消除所有数据移动。

## 实现和精度

复用已有 SmolLM 图构造器、resident gamma、多输出运行接口与 RMSNorm SHAVE 内核。两个具名 Power 载体分别接在 attention 与 FFN 前；每个图只替换一次共享的 Power 代码窗口，两处调用执行同一 RMSNorm 实现。FP32 Neumaier 补偿求和、epsilon=1e-5、先 normalize→FP16 再乘 gamma→FP16 的舍入顺序未变；DPU 的 FP16/W8A16 矩乘路径未变。没有新增或替换依赖，没有重建 ELF、native DLL/PYD，没有改通用严格 FP32。

B1 的两次 norm 都沿用内部复制完整行再切回第一行的方案，attention/FFN/公开 KV 仍为 B1。第二次 norm 在 B2..B8 可编译为带前导单例维度的布局。whole_layer 模式精确接受 0x4321 `(576,rows,1,1)` 与 0x2431 `(1,576,rows,1)`，并校验相应的完整行密集步长、行数、精度和三 memref 一致性。内核本来就支持跳过前导单例轴；未放宽 single-carrier 模式的布局白名单。

仍要求固定 8086:643e／编译器 8.3、核代码指纹、静态固定 CMX、边界、重定位及无输入输出重叠。48 个真实图（B1..B8 × FP16/W8A16 × forward/prefill/decode）全部与原两图实现逐位一致，包括 y、K/V 与 FP32 位置计数。原始／修改 blob 等长，5136 字节代码区以外逐位不变；代码窗口为 21216 字节，其他 SW/DPU 代码和任务不修改。

## 同实例性能对照

同一模型、输入、gamma 与原始 KV，仅切换私有整层开关。12 组，各 41 轮 ABBA/BAAB、每策略 82 个样本；所有 y/K/V/位置位模式一致。暖显式上传、下载、编译为零。四层两路径分别分配 20／16 次。

| 层数 / B / 前缀 | 模式 | 两图 ms | 整层单图 ms | 中位延迟降幅 | 更快轮数 |
|---|---|---:|---:|---:|---:|
| 1 / 1 / 1 | fp16 | 2.020 | 1.470 | 27.24% | 41/41 |
| 1 / 1 / 1 | w8a16 | 1.984 | 1.438 | 27.49% | 41/41 |
| 1 / 1 / 32 | fp16 | 2.162 | 1.655 | 23.43% | 39/41 |
| 1 / 1 / 32 | w8a16 | 2.005 | 1.400 | 30.17% | 41/41 |
| 4 / 1 / 63 | fp16 | 6.348 | 4.840 | 23.76% | 41/41 |
| 4 / 1 / 63 | w8a16 | 9.141 | 6.651 | 27.24% | 41/41 |
| 1 / 2 / 32 | fp16 | 1.874 | 1.491 | 20.43% | 40/41 |
| 1 / 2 / 32 | w8a16 | 1.930 | 1.405 | 27.21% | 41/41 |
| 4 / 4 / 63 | fp16 | 9.455 | 7.257 | 23.25% | 40/41 |
| 4 / 4 / 63 | w8a16 | 8.182 | 6.144 | 24.91% | 37/41 |
| 1 / 8 / 63 | fp16 | 2.231 | 1.639 | 26.53% | 40/41 |
| 1 / 8 / 63 | w8a16 | 2.099 | 1.570 | 25.21% | 39/41 |

本轮中位延迟下降 20.4%–30.2%，各组 37–41／41 轮更快。计时包括 Python 验证、绑定、提交、设备执行和同步 fence，排除初始加载、编译和显式传输。复制计数不含编译器／驱动内部 DMA。桌面负载、频率和内存状态未锁定；这些是本机暖层片段时间，不代表完整 LLM 速度或功耗。

control.json 在候选版本接入默认路径前生成，其 sources 中的旧 root 适配器 SHA 对应本目录 before 快照；实际计时使用 candidate.py。设备测试之后仅收紧了 CPU reference 的 segment 白名单，拒绝误接受的内部 layer 参数。审计逐字节验证生产文件相对 candidate.py 只有这一处校验差异，所有设备方法、图构造和运行代码完全相同；另有 3 项原 CPU 参考调用结果／trace 一致性和 1 项 layer 拒绝检查。sources 保存最终源码 SHA，tested_sources 保存设备测试时的 SHA，未改写原始数值／计时证据。

## 独立实例与 CPU

每组 21 轮轮转各路径；CPU 为 FP16、8 线程，重复 decode 后恢复原前缀。两图和整层为独立实例。下表保留所有中位时间，不能将独立实例／不同桌面负载的差值与同实例对照混用。

| 层数 / B | FP16 两图→整层 ms | W8A16 两图→整层 ms | CPU FP16 ms |
|---|---:|---:|---:|
| 1 / 1 | 2.119 → 1.926 | 2.032 → 1.680 | 1.341 |
| 1 / 4 | 1.983 → 1.999 | 1.906 → 1.705 | 1.999 |
| 2 / 1 | 4.217 → 3.681 | 3.983 → 3.167 | 2.390 |
| 2 / 4 | 3.842 → 3.441 | 3.907 → 3.166 | 3.973 |
| 4 / 1 | 8.294 → 6.965 | 8.049 → 6.235 | 4.446 |
| 4 / 4 | 7.794 → 6.874 | 7.579 → 6.003 | 7.112 |

## 执行分解

21 轮 traced 平均比例。归一化、attention、FFN 包含在 norm_layer 阶段，单独旧 bucket 为零不代表其中算子免费。native 包络包含绑定、提交、执行和等待，不能据此称为纯内核时间或断言已被计算／带宽完全限制。

| 层数 / B | 模式 | 整层阶段占比 | native 包络占比 | 原生调用数 |
|---|---|---:|---:|---:|
| 1 / 1 | fp16 | 77.4% | 57.7% | 1 |
| 1 / 1 | w8a16 | 76.2% | 52.6% | 1 |
| 4 / 1 | fp16 | 77.3% | 58.0% | 4 |
| 4 / 1 | w8a16 | 78.4% | 60.5% | 4 |
| 4 / 4 | fp16 | 79.4% | 55.9% | 4 |
| 4 / 4 | w8a16 | 78.8% | 58.7% | 4 |
| 1 / 8 | fp16 | 83.1% | 64.3% | 1 |
| 1 / 8 | w8a16 | 83.0% | 63.6% | 1 |

## 回归和边界

102 条真实权重数值记录、336 个固定输入 K/V 对照、252 次连续位置解码通过。最大执行相对 L2 0.2003%，scaled error 0.6472%；保留原 0.3%／1% 执行门槛与 3% 量化质量门槛。24 条新旧 cached decode、8 条普通 forward 与 8 组 CPU+NPU 混合生成保持位模式／生成 token 一致。

14 组 batch 隔离／快照检查，以及一个实例 B1→B2→B1→B8→B3→B1 切换通过；B1/B2 复用两行 gamma。两个调用者对同一缓存执行 6 次 warm decode，共 6 图，y/K/V/position 存储独立，旧输出与旧 KV 不变。执行仍由既有 NPU 锁串行化，此检查没有验证并行加速。

17 项载体保护检查（2 项接受、15 项拒绝）、16 项旧 ELF／载体离线测试与 105 项通用 FP16/严格 FP32 快速回归通过。首次只接受原布局的探测、随后只接受单例一行的探测分别拒绝 42／24 组；都是执行前布局保护拒绝，context_failed=False。诊断发现两种完整行排列以及 1／2 行分片后补齐精确校验，失败报告／源码在 failures/，没有修改内核或放宽密集步长／CMX 保护。

| 成功设备进程 | 检查记录 | 计时组 | NPU 调用 |
|---|---:|---:|---:|
| probe | 8 | 0 | 86 |
| numeric | 40 | 0 | 1176 |
| boundary | 14 | 0 | 506 |
| guards | 23 | 0 | 91 |
| positions | 44 | 0 | 670 |
| hybrid | 8 | 0 | 360 |
| paired | 24 | 0 | 332 |
| regression | 8 | 0 | 86 |
| profile | 0 | 8 | 1080 |
| performance | 0 | 6 | 5012 |
| model-probe | 48 | 0 | 260 |
| ownership | 16 | 0 | 375 |
| control | 12 | 12 | 6230 |

以上成功设备进程合计 16,264 次 NPU 调用，全部健康关闭、存活分配归零。通用快速回归另有 435 次调用。sources、kernel manifest、before/after、changes.diff 与证据 SHA 在 summary.json。

## 当前范围与复现

API 不变，模型／输入仍使用 intel_npu；prefill/decode 返回 NPU KV 快照。仍为冻结 SmolLM2-135M FP16/W8A16、B≤8、相同前缀长度、总位置≤64；多 token 前缀仍用原四图路径。不增加训练、任意模型、padding mask、跨层并行或完整 LLM 全 NPU 后端。运行不依赖 OpenVINO Python/runtime，仍依赖 Intel 驱动图编译器。

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_smollm_kv_demo.py -ScriptArguments '--phase','smollm-whole-layer','--case','numeric'
.\run_torch_npu.ps1 -Script .\torch_npu_whole_layer_benchmark.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_whole_layer_probe.py
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\smollm_whole_layer_ownership.py
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
.\.conda\npunlock\python.exe .\npunlock-probe\research\smollm_whole_layer_guards.py
.\.conda\npunlock\python.exe .\npunlock-probe\research\smollm_whole_layer_reference_guard.py
.\.conda\npunlock\python.exe .\npunlock-probe\research\summarize_smollm_whole_layer.py
```

模型 safetensors SHA：`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`。
