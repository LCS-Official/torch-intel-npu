# 当前性能瓶颈与 RMSNorm 编译候选（2026-10-07）

本轮完成当前版本的性能定位，并实际试验了一个保持计算语义的 RMSNorm O3 候选。
校准 W8 与原 W8 的耗时总体接近，未证明稳定加速；O3 候选反而更慢，未接入生产。
当前四层模型的主要设备热点仍是 SHAVE RMSNorm，decode 还受 KV/遮罩 Select 影响。
正式模型、扩展、runtime、RMSNorm C/ELF 均未改变，没有新增依赖。

## 普通执行 API 的交错计时

Lunar Lake / Core Ultra 7 258V、NPU 8086:643e、驱动 32.0.100.5540、编译器 8.3、
Torch 2.14.1+cpu。固定 SmolLM2-135M-Instruct 前四层，使用同一组真实 embedding 输入。
前缀 B1/S32 与 B4/S16；decode 从各自固定前缀快照追加一个 token。
三种模式均先完成编译、上传和预热，再在同一进程轮换顺序、正反交错，31 轮，
每种模式每项 62 个样本；另起进程完整重复一次。

下面是未开启 host/task profiling 的 API 中位延迟。计时包括四层同步执行、
Python/C++ 调度和新输出/KV 分配；结果清理在计时外。排除首次准备、编译及显式传输。

| 进程 | 形状 / 操作 | 原 W8 ms | 校准 W8 ms | FP16 ms |
|---|---|---:|---:|---:|
| 0 | B1/S32 prefill | 3.2653 | 3.2985 | 3.3266 |
| 1 | B1/S32 prefill | 3.7173 | 3.6348 | 3.6884 |
| 0 | B1 decode | 2.2710 | 2.2356 | 2.3813 |
| 1 | B1 decode | 2.4629 | 2.4787 | 2.6106 |
| 0 | B4/S16 prefill | 5.0612 | 5.1173 | 5.1065 |
| 1 | B4/S16 prefill | 4.8514 | 4.7772 | 4.9221 |
| 0 | B4 decode | 2.4486 | 2.5132 | 2.6459 |
| 1 | B4 decode | 2.3669 | 2.4158 | 2.4259 |

校准相对原 W8 的耗时变化为 -2.22%～+2.64%，多数项目的方向在两进程之间改变。
B4 decode 在两个进程分别慢 2.64%、2.07%，保留这个小幅退化观测，不能宣称完全零开销。
校准发生在离线准备阶段，推理代码仍相同；权重数值改变及桌面/频率波动可能影响测量，
本轮没有进一步定位这项小差异的原因。

预热后的全部计时循环新增驱动分配为 **0**，逻辑输出分配均命中缓存。
每个四层操作仍为 **4 次图调用**，零新增显式上传、下载、设备间复制、打包或图编译。
这里是四层模型段的延迟，不是完整 30 层 LLM 的 token/s，也不是 NPU 峰值 TOPS。

## 主机分段：等待和命令提交

另一个隔离进程绑定带 QPC 计数器的诊断 DLL，仍执行普通未开启 task profiling 的图。
沿用相同三模式交错测量。下表仅列校准 W8；它不能与上表跨进程相减得到加速。

| 操作 | API 中位 ms | fence wait 中位 ms | 命令相关阶段中位合计 ms | wait / API 均值 |
|---|---:|---:|---:|---:|
| B1/S32 prefill | 3.1286 | 2.4256 | 0.20 | 77.3% |
| B1 decode | 2.2393 | 1.5804 | 0.20 | 66.7% |
| B4/S16 prefill | 5.1674 | 4.4174 | 0.22 | 84.4% |
| B4 decode | 2.4200 | 1.8223 | 0.17 | 73.0% |

命令相关阶段包含参数绑定、append、close、submit、list/fence reset。
这里每个样本的驱动载荷 alloc/free 都是 0。wait 包含设备计算、搬运、设备同步及主机唤醒，
不能把它全部叫作纯计算。API 时间还包含未被这九项计数器覆盖的 Python/C++ 工作与计数器开销。
中位数与均值采用不同口径，表中的各中位数不是严格可相加的分解。

## 当前设备热点

用已有隔离 profiling DLL 重新编译当前 FP16、校准 W8 图；四层、两种形状、
prefill/decode 共 32 张图，每图 3 份通过重建 query 获得的独立任务数据。
所有 profiling 图的输出、KV、位置与普通图逐位一致。

下表是校准 W8 对应阶段占**已记录设备时间跨度**的比例中位数。
同类并行任务使用区间并集，四层因同步提交而按层相加，不累计重复计算重叠时间。

| 操作 | RMSNorm | Select（KV/遮罩等） | DPU | attention softmax |
|---|---:|---:|---:|---:|
| B1/S32 prefill | 83.04% | 0% | 4.26% | 2.44% |
| B4/S16 prefill | 85.65% | 0% | 4.19% | 1.62% |
| B1 decode | 38.85% | 12.64% | 5.04% | 3.52% |
| B4 decode | 33.94% | 22.97% | 5.50% | 3.06% |

这些结果来自更新后的默认 SIMD 转换内核，确认此前优化后 RMSNorm 仍是热点。
FP16 也呈现相同趋势；全部样本、时间跨度、SHAVE/DPU 区间与波动范围保存在 `summary.json`。
校准 W8 图仍有 `amode=0, wmode=2` 的 DPU 矩阵任务。

各类区间可能互相重叠，比例不能作为完整的互斥饼图，也不能当作普通 API 延迟的比例。
profiling 会改变编译图调度和执行成本；例如本轮校准 W8 B1 prefill 的带 profiling API
约 5.97 ms，明显高于普通图的约 3.3～3.6 ms。因此这里用它定位热点，不报告 profiling 图的加速比。

这批数据没有 DMA task 时间记录。未记录的空档可能含搬运、同步、调度等，
不能据此确定“主要受 IO 限制”，也不能把 task 数为零理解为没有 DMA。
证据支持优先优化 SHAVE RMSNorm 和 decode 的 KV 更新；这些形状下只优化 GEMM 的整段收益受限。
这项判断不外推到大 batch 矩乘、长上下文、完整 LLM 或 GPU/NPU 并行。

## 实际试验：O3 RMSNorm 未改善性能

使用相同的生产 RMSNorm C，唯一的编译选项变动为 `-O2`→`-O3`，仍保留
`-ffp-contract=off`，没有启用 fast-math、近似开方或改变精度。
在同一编译器图载体中只替换 SHAVE 代码窗口，窗口外 blob 字节完全相同。
为了复用原有工具，候选的内部名称仍为 `codec`，在本轮表示 O3 编译结果。

代码从 **4,592 字节增加到 11,088 字节**，未超过 21,216 字节载体窗口。
104 项数值检查通过，包括全部有限 FP16 编码、零/次正规数、极值、舍入压力、真实 embedding
及 NaN/Inf。有限值和符号零按原始位比较；特殊值测试中的 NaN 按 NaN 等价规则比较。

独立 RMSNorm 使用预分配输出，同载体、21 轮 ABBA/BAAB，每种内核 42 个样本：

| 行数（宽 576） | 当前内核 ms | O3 ms | O3 耗时增加 |
|---:|---:|---:|---:|
| 1 | 0.28910 | 0.31495 | 8.94% |
| 4 | 0.61125 | 0.63835 | 4.43% |
| 32 | 0.58225 | 0.64995 | 11.63% |
| 256 | 2.12000 | 2.16690 | 2.21% |
| 512 | 3.22115 | 3.32075 | 3.09% |

随后在同一个四层模型、同一输入和 seed KV 上切换两套图，31 轮，每种实现 62 个样本。
覆盖 FP16、校准 W8，B1/S32、B4/S16 的 prefill/decode，共 8 组；
O3 的中位延迟全部更高，增加 **0.41%～6.57%**，每组仅 3～14/31 轮更快。
四组模型的 104 个输出/KV/位置数组逐位一致，旧 seed 快照保留原值。

因此候选未采用，也没有为了追求速度放宽精度。代码增大和变慢是观测事实；
本轮没有测 instruction-cache 或寄存器 spill，不能把它们确定为变慢的原因。
由于候选已经在全部组中未显示收益，没有继续重复更多进程或替换生产内核。

## 核对、来源和复现

最终有效测试共 **15,242 次 NPU 调用**：性能定位 10,400 次、O3 候选 4,842 次。
**876 项数组逐位核对**通过，包括跨普通进程、主机诊断、设备 profiling、O3 对照和旧 KV 保留。
32 张 profiling 图提供 96 份新任务数据、14,736 条任务记录，原始 binary 逐字段与 JSON 对照。
所有测试进程 shutdown 后活跃、闲置缓存、隔离载荷字节均为 0，context 没有失败。

离线核对脚本验证模型、校准权重、生产源码/二进制、诊断 DLL、当前 RMSNorm ELF、
原始 NPZ、task binary 和候选 blob 的指纹/字节。结果在 `summary.json`、`audit-run.log`；
原始普通计时在 `benchmark-0/1.json`，主机分段在 `host.json`，设备数据在 `device-fp16/`
与 `device-calibrated/`。`*-trace.json` 可用 Perfetto 查看，每张图独立相对时间轴。
O3 的源码、汇编、ELF、构建记录及数值/模型检查位于 `rmsnorm-o3/`。

首次工具核对误将后来修改过的 Python builder 当作诊断 DLL 的构建输入，
在主机测试开始前退出；已保留首轮两个普通 benchmark 与错误日志到 `initial-tooling/`。
修正后仍核对所有参与 DLL 构建的 C 文件与生成 C、DLL 指纹，当前 Python helper 单独记录指纹。
这次工具失败和首轮测量不计入上述最终调用统计。随后完整重跑并通过。
测试没有锁定频率或控制桌面负载，两轮绝对时间有波动。

在 `E:\Pytorch_xfire_NGPU` 使用已有专用环境；硬件任务须串行，重跑会覆盖本轮结果：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_performance.py -ScriptArguments '--case','control'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_o3.py -ScriptArguments '--case','build'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_o3.py -ScriptArguments '--case','numeric'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_o3.py -ScriptArguments '--case','model','--label','process0'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\w8_performance_audit.py
```

复用现有 profiling、计时和数值检查工具，没有新增执行框架。
当前结果支持继续针对 RMSNorm 内部实现及 KV 更新布局做优化，不支持把离线 W8 校准当作速度优化。
