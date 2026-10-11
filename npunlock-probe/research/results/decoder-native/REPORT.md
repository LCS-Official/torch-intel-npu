# 30 层 decoder：原生等待、设备任务与两项 RMSNorm 候选

2026-10-09，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动 32.0.100.5540，编译器 8.3。
模型 SmolLM2-135M-Instruct，30 层全部在 NPU，B1、prefill 24 token、随后一次 decode。
本报告只测驻留 decoder；CPU embedding / 最终 norm / lm_head、输入上传与输出下载不在计时内。
decode 使用既有默认四层分组。没有新增 OpenVINO runtime 依赖。

本轮确认主要已记录热点仍在 SHAVE，特别是自定义 RMSNorm；
两项保持原算术顺序的隔离候选均通过数值检查，但没有完整模型加速，均未采用。
生产 Python runtime、native C / DLL / PYD、模型适配器、RMSNorm C / ELF / 汇编保持本轮开始时的版本。

## 原生阶段：没有启用设备任务 profiling 的图

使用隔离诊断 DLL 的 QPC 计数器，每种操作 62 个预热后样本。
数值为各项中位数，不能要求不同项的中位数严格相加等于总时间。

| 阶段 | prefill，ms | decode，ms |
|---|---:|---:|
| 整个 decoder 调用 | 18.307 | 12.484 |
| 原生 fence 等待 | 13.990 | 10.065 |
| 队列提交 | 1.112 | 0.688 |
| 参数绑定 | 0.159 | 0.152 |
| append graph | 0.079 | 0.036 |
| 原生计数器覆盖之外，约 | 2.840 | 1.496 |

decode 每次 30 次 append、300 次参数绑定、8 次提交 / 等待 / reset。
prefill 为 30 次 append、180 次参数绑定、30 次提交 / 等待 / reset。
两条暖路径驱动共享内存分配 / 释放均为零，逻辑 Tensor 分配由既有缓存复用。
计数器之外仍含 Python、C++ 绑定验证、锁及部分原生控制流程，不能全部标作 Python。
等待含设备执行、排队和未单独记录的搬运等，不能直接称为纯 DPU 算时。

原始样本：[host.json](host.json)。

## 设备任务：另行启用 PERF_COUNT 的图

FP16 / W8A16 各 60 张图，每图重新创建 query 后采样 3 次，共 360 份独立记录。
每项先在单图内合并同类任务的重叠区间，再跨 30 层求和，最后取 3 次采样的中位数。
RMSNorm 根据 `rmsnorm_carrier` / `rmsnorm_ffn` 名称核对，属于 SHAVE 子集。

| 模式 / 操作 | DPU，ms | SHAVE，ms | 其中 RMSNorm，ms | 各图任务跨度之和，ms |
|---|---:|---:|---:|---:|
| FP16 prefill | 0.994 | 13.339 | 11.508 | 15.978 |
| FP16 decode | 0.792 | 9.650 | 4.908 | 13.807 |
| W8A16 prefill | 0.980 | 14.933 | 12.400 | 17.525 |
| W8A16 decode | 0.803 | 10.750 | 5.390 | 14.383 |

FP16 decode 的其余 SHAVE 热点还包括 Multiply 约 1.43 ms、Select 约 0.98 ms、
Equal 约 0.63 ms、attention MatMul 约 0.52 ms、Convert 约 0.49 ms。
这些类别可能互相重叠，不能直接求和构造整次调用的关键路径。

profiling 改变图调度：FP16 的 profiled prefill / decode 墙钟为 27.33 / 21.20 ms，
W8A16 为 30.98 / 22.19 ms，不能把本表的任务时间从上一节正常图时间中直接相减。
不同模式也在不同时段采样，不能据此判断 W8A16 的正常生成一定比 FP16 慢。
驱动没有提供 DMA 时长记录；表中没有 DMA 项不代表 IO 为零，也不能排除共享带宽瓶颈。

任务明细：[FP16](profile/b1-s24-fp16.json)、[W8A16](profile/b1-s24-w8a16.json)。
分类聚合：[summary.json](summary.json)。

## 两项隔离候选：均未接入生产

只替换同一编译图的 Power 代码窗口，窗口之外的布局 / 调度 / 常量字节保持一致。
KV Select 继续使用生产内核。每项完整模型比较 31 轮，轮内正反交错执行，每个版本 62 个样本。

| 候选 | prefill 原 / 候选，ms | decode 原 / 候选，ms | decode 更快轮数 |
|---|---:|---:|---:|
| 两行逐位相同时计算一行并复制 | 18.279 / 18.732 | 12.592 / 12.671 | 11 / 31 |
| 单行 Neumaier 使用带 NaN 保护的非负 FP32 位序比较 | 18.268 / 18.868 | 12.468 / 12.646 | 6 / 31 |

重复行方案数值检查 184 项，位序比较方案 158 项，均通过。
有限值 / signed zero 逐位一致，独立特殊值测试按既有 NaN-equivalent 规则比较。
每项完整模型的 182 个输出 / KV / position 数组与原内核逐位一致，90 个原 seed 数组保持不变。

重复行候选没有触发目标优化：B1 虽构造两行 carrier，但当前每个 SHAVE MemRef 的
`elements=576`，每次调用只看到一行；两行由不同调用处理。
若要跨调用复用，需要重新处理调度与同步，不能靠越界读取相邻行。
单行比较方案也没有观察到收益；原汇编已将对应选择生成 CMU.MAX.f32 / CMU.MIN.f32，
源码层替换比较不保证更少或更快的指令。
普通形状的独立算子测试也有退化，因此保留两个候选作为失败实验记录。

原始对照：[重复行](duplicate-rows/full30.json)、[单行位序](scalar-bits/full30.json)。

## 审计、范围与下一项

[audit.json](audit.json) 核验 360 份任务数据指纹、两模式 plain / profiled 的 182 个数组、
各自 90 个 seed 数组、host 与设备剖析 plain 控制的 182 个跨进程数组、
候选 C / ELF / 汇编指纹、原始计时样本与生产文件未改变。
全部进程正常退出，上下文未失败，活动 / 缓存 / 隔离内存及会话均归零。
结果目录约 76 MiB；未保存整套模型编译 blob，输出压缩保存，设备诊断输出沿用已有 NPZ 格式。

本轮没有改动生产计算路径，未重跑完整训练回归或真实 32-token 生成。
W8A16 已有首步 logits 5.69% 相对 L2 误差超出 3% 门槛的问题也未在本轮修复或重新认证。
桌面后台负载 / 频率未锁定；候选采用与否以同进程交错对照为依据。

下一项优先对**真实一行 MemRef 的 RMSNorm**做归约 / FP16 解码 / 输出舍入分段消融，
再针对最耗时阶段优化；同时关注 KV 更新 / 掩码小任务数量。
当前证据不支持继续依赖“一个调用处理两行”的假设，也不支持仅凭 TOPS 峰值预估单 token 加速。

## 复现

以下命令会覆盖同名结果，复测前应保留本目录冻结证据，全部硬件任务串行运行。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decoder_native_profile.py -ScriptArguments '--case','build'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decoder_native_profile.py -ScriptArguments '--case','host'
foreach ($precision in @('fp16','w8a16')) {
    .\run_torch_npu.ps1 -Script .\npunlock-probe\research\decoder_native_profile.py -ScriptArguments '--case','tasks','--mode',$precision
}
foreach ($probe in @('rmsnorm_duplicate_rows.py','rmsnorm_scalar_bits.py')) {
    foreach ($probeCase in @('build','numeric','model')) {
        .\run_torch_npu.ps1 -Script ".\npunlock-probe\research\$probe" -ScriptArguments '--case',$probeCase
    }
}
# 只核验保存的证据，不运行 NPU。
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\decoder_native_audit.py
```
