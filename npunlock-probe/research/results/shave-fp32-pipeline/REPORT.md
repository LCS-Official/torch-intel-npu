# 当前 30 层 FP16 decoder：SHAVE FP32 归约诊断与四行候选

2026-10-08，Core Ultra 7 258V / Lunar Lake NPU 8086:643e，驱动 32.0.100.5540，图编译器 8.3。

结论：当前 FP16 模型仍值得优化 RMSNorm 中的 FP32 计算，但本轮“四行交错 SIMD”候选虽然通过全部 112 项数值检查，实际更慢，未接入默认内核。另发现 30 层 decode 每步仍有 28 次驱动分配，适合下一轮单独验证缓存上限。

这里研究的是 FP16 模型里的 FP32 平方和、补偿累加与归一化，不是严格 FP32 GEMM，也没有把 FP32 降为 FP16 或 TF32。模型为冻结 SmolLM2-135M-Instruct，全部 30 层 decoder 在 NPU；embedding、最终归一化、输出头在本轮计时之外。

## 普通基线与主机开销

输入常驻设备，固定同一前缀、同一 KV 种子；decode 每次只处理一个位置，计时不会推进种子。每项预热 3 次，31 轮共 62 个计时样本。包括 Python、提交、同步等待；排除图编译、初始上传、校验下载及返回对象之后的销毁。没有控制桌面负载。

| Batch / 前缀 | 操作 | 普通基线 ms | 主机计数器版本 ms | 其中 fence 等待 ms | 其中驱动分配 ms |
|---|---|---:|---:|---:|---:|
| 1 / 32 | prefill | 21.437 | 21.012 | 16.407 | 0 |
| 1 / 32 | decode | 18.452 | 17.770 | 11.617 | 0.737 |
| 4 / 16 | prefill | 37.415 | 33.034 | 27.757 | 0 |
| 4 / 16 | decode | 24.586 | 19.150 | 12.394 | 0.786 |

普通基线保存在 10 月 7 日，主机剖析在 10 月 8 日恢复后完成；跨进程、跨会话差值不能解释为优化收益。两者所有输出逐位一致。精确数值与全部样本见 `baseline-0.json`、`host.json`；上表分项分别取中位数，不能按表中数值相减得到某一因素的精确贡献。

预热后四项均没有显式主机载荷复制、重新编译或创建会话。prefill 没有新增驱动分配；两种 decode 各 62 次调用均有 1,736 次驱动分配，即每步 28 次。设备剖析的三次独立样本也都记录到每步 28 次 `zeMemAllocShared`。

当前 allocator 同时限制 16 MiB 和 64 个闲置缓冲块。30 层 decode 返回的 KV、位置和隐藏状态超过这个块数，存在缓存淘汰的合理解释。尚未进行调大块数的对照，因此不将其写成已证明的全部原因，也不承诺调整后的收益。分配耗时本身约 0.7–0.8 ms，不能解释整段等待时间。

## 当前真实任务热点

每种形状保存 30 层 prefill 和 30 层 decode，共 120 个图；每图重建 profiling query 后取 3 份不同的任务记录，合计 360 份。RMSNorm 统计先在每层内合并时间区间，再对 30 个同步图求和；分母是各层任务时间跨度之和。

| Batch / 前缀 | 操作 | RMSNorm 占任务时间跨度 |
|---|---|---:|
| 1 / 32 | prefill | 74.16% |
| 1 / 32 | decode | 32.85% |
| 4 / 16 | prefill | 80.04% |
| 4 / 16 | decode | 30.57% |

比例是先逐样本计算再取中位数。各任务类别可以重叠，这些百分比不是 NPU 硬件利用率，也不是整段 API 耗时占比。profiling 图比普通图更慢，不能用它替换上面的普通基线。

29,160 个 SHAVE 任务都返回了非零 `active_cycles` 和 `stall_cycles`；28,800 个 DPU 任务的这两个字段均为零。归一化的原始周期合计已写入 `audit.json`。这些字段可用于后续同内核、同形状的对照，但没有分指令发射率、停顿原因或整个 NPU 的利用率数据，不将二者比例直接称为流水线效率。

[Intel profiling ABI](https://github.com/intel/level-zero-npu-extensions/blob/main/ze_graph_profiling_ext.h) 定义了这些任务字段。[NPU compiler profiling 文档](https://github.com/openvinotoolkit/npu_compiler/blob/develop/guides/how_to_use_profiling.md) 说明 profiling 会增加缓冲管理和调度开销，NPU40xx 驱动栈没有 DMA profiling 支持，且 DPU 时间可能受动态频率影响。本轮任务数据也未提供 DMA 时长，因此不能把未观测的空隙全归为 IO。

## 四行交错候选：数值通过，性能不采用

原内核每次向量解码、平方四个相邻元素，再按列顺序做标量 Neumaier 补偿累加。候选让四个 SIMD 通道分别负责四行，保持每行 576 个元素的处理顺序，向量计算累加和补偿；不足四行继续使用原分支。仍用 `-O2 -ffp-contract=off`，没有启用 fast-math。输出的两次 FP16 舍入以及权重乘法保持原语义。

112 项检查通过：不同长度及尾行、全部有限 FP16 编码、65,536 种编码回转、随机舍入压力、真实 embedding、两组模型 gamma，以及新四行路径中的 Inf、NaN、符号零、最小 FP16 次正规数和大数。有限值及零按位相等，NaN 按 NaN 等价比较，不承诺 payload 相等。

独立调用使用同一 carrier、预分配输出、同进程交错 A/B，21 轮每版本 42 个样本。热循环没有显式主机复制或新建会话，也没有新增驱动分配。

| 行数 | 当前内核 ms | 四行候选 ms | 候选耗时增加 | 候选胜出轮数 / 21 |
|---:|---:|---:|---:|---:|
| 1 | 0.22425 | 0.23260 | 3.7% | 9 |
| 4 | 0.48525 | 0.56265 | 16.0% | 2 |
| 32 | 0.53475 | 0.63105 | 18.0% | 1 |
| 256 | 1.61455 | 2.01250 | 24.6% | 0 |
| 512 | 2.65860 | 3.27145 | 23.1% | 0 |

候选确实生成了向量加减，但循环内也出现较多标量比较和寄存器溢出后的栈读写。对 `.LBB0_37` 到回跳指令的静态前缀，当前内核未见栈访问，候选有 10 条；代码段从 4,528 字节增至 6,208 字节。该静态统计不含后续延迟槽指令，不能当作动态周期数或单一原因的因果证明。对应片段保存为 `row4/*-reduction-loop.txt`。

由于所有代表性独立调用均未加速，本轮停止该候选，没有继续跑整模型候选 A/B，也没有修改生产 C、ELF、后端或模型路径。整模型候选测试入口保留，但本轮未执行。

## 验证与后续

离线审计重读了 57,960 条原始任务记录，验证 344 字节 ABI、时间区间、文件 SHA-256、Power / Select 补丁组合、驱动构建绑定以及 1,272 次数组逐位对照。另有候选的 112 项数值检查、562 次 NPU 调用。所有完整进程结束后 live/cache/quarantine 字节归零，未发生 context failure。源码和二进制指纹在 `audit.json` 与 `row4/build.json`，来源快照在 `sources/`。

本轮 165 份研究图证据逻辑大小约 0.87 GiB，已执行 Windows LZX 透明压缩，逐文件验证压缩前后 SHA-256 一致；记录在 `compression.json`。默认运行时、内核 ELF 和模型文件未作压缩或替换。

下一步建议先做一个小对照：保留 16 MiB 字节上限，只增加 allocator 缓存块数，检查能否消除 30 层 decode 的 28 次分配及实际收益。SHAVE 后续应减少比较、寄存器压力和访问开销；直接增加 SIMD 通道工作量未必有收益。当前证据仍不支持优先转去优化与这个 FP16 模型关系较小的通用 FP32 GEMM。

从项目根目录复现，所有 NPU 命令须串行执行；同名输出会被覆盖：

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\shave_fp32_pipeline.py -ScriptArguments '--case','benchmark','--label','baseline-0'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\shave_fp32_pipeline.py -ScriptArguments '--case','host','--label','host'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\shave_fp32_pipeline.py -ScriptArguments '--case','device','--batch','1','--prefix','32'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\shave_fp32_pipeline.py -ScriptArguments '--case','device','--batch','4','--prefix','16'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_row4.py -ScriptArguments '--case','build'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\rmsnorm_row4.py -ScriptArguments '--case','numeric','--label','probe'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\shave_fp32_audit.py
```
