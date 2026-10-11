# 四层 SmolLM：DPU / SHAVE 设备任务诊断

2026-10-06，Core Ultra 7 258V / Lunar Lake NPU 4000，设备 8086:643e，
驱动 32.0.100.5540，驱动图编译器 8.3，Torch 2.14.1+cpu。

当前四层融合图的首要设备计算瓶颈是自定义 SHAVE RMSNorm。
prefill 中，它占已记录设备时间跨度约 82%～87%，占 SHAVE 活跃时间约 93%～95%；
decode 中分别约 35%～42%、45%～55%，其他 SHAVE 运算与未记录区段也值得继续检查。
本轮增加了隔离的诊断工具、离线核对和时间线导出，正式 runtime、内核和模型实现保持原有指纹。

## 实验范围与统计方法

使用本地 SmolLM2-135M-Instruct 的真实 embedding 和连续前四层权重，
分别运行 FP16、W8A16，batch 1 / prefix 32，以及 batch 4 / prefix 16。
每组先做 prefill，再以原有 seed KV 做单 token decode；KV 容量固定为 64。
这是四层子图实验，没有测量完整语言模型生成的 tokens/s。

相同模型先正常编译，再保留原有编译参数并追加 `PERF_COUNT="YES"`。
使用现有 RMSNorm ELF 和完整 carrier、CMX、MemRef 检查，在 profiling 图中替换相同 Power carrier。
profiling 记录中的 `rmsnorm_carrier?t_Power` 和 `rmsnorm_ffn?t_Power`
实际执行我们的 RMSNorm 内核；名字中的 Power 不是另一次幂运算。

每张图在第一次执行之前创建 profiling pool/query；每个模型操作预热 3 次，采样 3 次。
每次正式采样前，先在上一次同步完成且 command list 已重置后销毁、重建 query/pool。
重建及读取 profiling 数据都在调用计时之外。每次调用同步执行四层的四张图。
每张图的三个 task 数据文件 SHA256 均不同，排除了重复读取旧结果的问题。

Level Zero task ABI 为 344 字节，区分 DPU、SW/SHAVE、DMA。
统计对每张图的区间 `[start, start + duration)` 求并集，再相加四层结果。
因此多个 SHAVE 同时执行不会被重复计入活跃时间。
时间跨度为每张图最早已记录任务开始至最晚已记录任务结束。
不同引擎的区间可能重叠，DPU / SHAVE 的占比不要求合计为 100%。
各列分别取三次样本的中位数；百分比也是逐样本计算后取中位数。

## 实测结果

时间单位 ms。RMSNorm、SHAVE、DPU 均为各自任务的区间并集；RMSNorm 包含在 SHAVE 中。

| 配置 | 操作 | Torch 调用 | fence 等待 | 已记录设备跨度 | SHAVE | RMSNorm | DPU | RMSNorm / 跨度 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B1 / S32 / FP16 | prefill | 6.4928 | 5.3210 | 4.2146 | 3.7272 | 3.4634 | 0.1688 | 82.3% |
| B1 / S32 / W8A16 | prefill | 6.1160 | 5.2402 | 4.0948 | 3.7421 | 3.4668 | 0.1547 | 84.6% |
| B4 / S16 / FP16 | prefill | 10.3859 | 9.3053 | 8.1078 | 7.2801 | 6.8307 | 0.3084 | 84.2% |
| B4 / S16 / W8A16 | prefill | 10.1697 | 8.8065 | 7.8103 | 7.1888 | 6.8166 | 0.2889 | 87.2% |
| B1 / S32 / FP16 | decode | 4.5823 | 3.5685 | 2.4820 | 1.8415 | 0.9415 | 0.1127 | 37.9% |
| B1 / S32 / W8A16 | decode | 4.2259 | 3.2931 | 2.2783 | 1.7299 | 0.9484 | 0.1075 | 41.7% |
| B4 / S16 / FP16 | decode | 4.7190 | 3.7297 | 2.6964 | 2.0785 | 0.9320 | 0.1395 | 34.6% |
| B4 / S16 / W8A16 | decode | 4.7212 | 3.3290 | 2.3572 | 1.9017 | 0.8456 | 0.1219 | 35.9% |

decode 的其他可见 SHAVE 开销：

| 配置 | 非 RMSNorm SHAVE | Select | attention softmax | 已记录跨度内空档 |
|---|---:|---:|---:|---:|
| B1 / S32 / FP16 | 0.9000 | 0.2733 | 0.0733 | 0.5495 |
| B1 / S32 / W8A16 | 0.7818 | 0.2741 | 0.0732 | 0.4345 |
| B4 / S16 / FP16 | 1.1457 | 0.5697 | 0.0772 | 0.5237 |
| B4 / S16 / W8A16 | 1.0556 | 0.5210 | 0.0724 | 0.3913 |

Select 包含 K/V 更新和注意力遮罩选择。这里各项也是区间并集，可能相互重叠。
attention softmax 由任务名中的 `/attention_pattern/softmax/` 识别；
其 profiling layer_type 虽标为 MatMul，不能算作 DPU 矩乘。
Equal、Greater、Convert、Multiply、Swish 等记录保存在原始 JSON 和汇总中。

并行任务累计时间会显著放大数字。例如 B4 FP16 decode 的 SHAVE 任务累计时间
中位数为 10.4580 ms，而 SHAVE 区间并集为 2.0785 ms。
本报告不把前者当成推理延迟，也不据此计算实际加速比。

## 当前可判断的优化方向

1. 优先针对现有 RMSNorm 内核做阶段剥离，测量归约、半精度转换及输出阶段的当前成本。
   内核已经有四元素向量解码/编码，不能把重新加入相同向量化当成新的优化。
   源码仍有顺序补偿归约；它是否是主要内部成本，需要下一轮实验验证。
   保留 FP32 归约、特殊值处理、以及乘 gamma 前的 FP16 舍入语义。
2. decode 的第二个明确目标是 KV 更新相关 Select、位置与遮罩处理。
   B4 中 Select 活跃时间约 0.52～0.57 ms，已经超过全部 DPU 任务的活跃时间。
   后续改变缓存写入方式仍需验证旧 seed KV 和此前输出不会被覆盖。
3. 这批形状下，继续只优化 DPU GEMM 或进一步压低权重精度的端到端收益会受上述阶段限制。
   该判断针对这些四层、小 batch 的模型图，不外推到大 batch GEMM 或其他模型。

## 数据边界

本平台没有 DMA task 时间记录，汇总明确标记 `dma_duration_available=false`。
原始 DMA 任务数为 0 只代表没有记录，不能据此说没有搬运开销。
设备时间线空档、fence 等待与任务跨度之间的差值，都不能直接归为 DMA。
它们可能包含未记录搬运、同步、调度、驱动及主机唤醒等阶段。

Intel 的[官方 profiling 说明](https://github.com/openvinotoolkit/npu_compiler/blob/develop/guides/how_to_use_profiling.md)
确认 NPU40xx 驱动栈没有 DMA profiling 支持，并说明 profiling 会增加 CMX / DMA 工作，
以及 DPU 时间戳可能受到动态频率变化影响。因此本轮数据用于定位任务热点，
不替代未开启 profiling 的生产吞吐测试，也不承诺消除 RMSNorm 后可获得相同比例的加速。
未控制其他桌面负载，每组仅三个设备样本；原始区间保存在 `summary.json`。

ABI 依据 [Intel Level Zero profiling header](https://github.com/intel/level-zero-npu-extensions/blob/main/ze_graph_profiling_ext.h)。
运行时未导入或安装 OpenVINO 包，仍通过已有 npunlock 与 Intel 驱动编译器执行。

## 正确性、资源与原始证据

- 4 个隔离进程，每个 80 次成功 NPU 调用，共 320 次；其中计时的融合层图调用共 96 次。
- 32 张 profiling 图，每图 3 份独立任务数据，共 96 份、14,736 条任务记录。
- 104 项正常图 / profiling 图的输出、KV、位置数组按原始字节一致。
- 后续 decode 后的 48 项原 seed KV / 位置数组仍保持原值。
- 24 次计时模型调用均有四次 append/submit/wait/reset，计时内驱动载荷分配与释放均为零。
- 每个进程 shutdown 后，活跃载荷与闲置缓存均为零，context 未失败。
- 前一阶段 `warm-allocation` 的完整离线核对再次通过。

正式 Session DLL SHA256：
`d959fc5b1fb9a3171c192d1f607b11431af634203805d46cdcd6bba243933a51`。

正式 Torch PYD SHA256：
`25d48cf772022ca7908934483c123e38cb8923ddd86223c00c8d994f1a39ef60`。

RMSNorm ELF SHA256：
`b8bf4b2a205587c29cdd25422c83b1033634f2f53161a8d713ed1ba603bf4799`。

隔离 profiling DLL SHA256：
`6d0758051e685b15ac265110708a5fa31e9c77d8c1fc18a9eeda95e681776995`。

测量脚本快照为 `measurement.py`，完整构建参数和生成 C 的指纹在 `build.json`。
四组最终记录为 `b1-s32-fp16.json`、`b1-s32-w8a16.json`、
`b4-s16-fp16.json`、`b4-s16-w8a16.json`，对应日志、outputs.npz、blob、task binary 同目录保存。
`summary.json` 保存区间统计与原始三个样本，`audit.json` 保存核对结果和文件指纹。

八个 `*-trace.json` 文件可在 [Perfetto](https://ui.perfetto.dev/) 中查看。
每个 graph 使用独立面板、相对起点；图之间的提交间隔没有放到同一全局时间轴。
visual lane 只为显示并行重叠，不代表物理 SHAVE/DPU 编号。

初次遗漏 query 的失败保存在 `failures/missing-query/`，属于诊断工具初始化问题。
最初复用 query 产生的重复任务记录保存在 `initial-query-reuse/`，不参与本轮汇总。
已通过第一次执行前创建 query、每次采样重建 query/pool 解决，两次问题都没有修改生产运行时。

复现命令，在 `E:\Pytorch_xfire_NGPU` 中运行；重跑前先保存原结果目录：

```powershell
.\run_torch_npu.ps1 -Script .\torch_npu_device_profile.py -ScriptArguments '--case','build'
.\run_torch_npu.ps1 -Script .\torch_npu_device_profile.py -ScriptArguments '--case','all'
.\run_torch_npu.ps1 -Script .\torch_npu_device_profile_audit.py
```

离线核对绑定本轮的测量脚本、生产构建及内核指纹；修改这些文件后需要重新建立测量快照与基线。
