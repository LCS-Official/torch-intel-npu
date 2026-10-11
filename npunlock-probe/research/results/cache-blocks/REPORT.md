# 闲置缓存 64 → 128 块：30 层 decode 的重复分配修复

2026-10-08，Core Ultra 7 258V / Lunar Lake NPU 8086:643e，驱动 32.0.100.5540，图编译器 8.3，Torch 2.14.1+cpu。

已将正式 allocator 的缓存块数上限从 64 调至 128，闲置缓存字节上限仍为 16 MiB。普通 `intel_npu` Tensor 自动使用该策略，无需修改模型调用。正式代码仅调整 C++ 上限与 Python stats 中对应的容量；研究扩展中的容量切换接口没有装入正式 PYD。

## 已确认的原因

30 层 SmolLM2-135M-Instruct 的 prefill 结束后保留 62 块闲置缓冲区，decode 结束后需要 92 块。旧上限为 64，decode 每步恰好重复分配、释放 28 块；本轮只改变块数，128 块时这些操作均消失。

| 形状 | 操作 | 128 块配置下实际闲置块数 | 闲置字节数 |
|---|---|---:|---:|
| B1 / 前缀 32 | prefill | 62 | 1,548,288 |
| B1 / 前缀 32 | decode | 92 | 1,476,984 |
| B4 / 前缀 16 | prefill | 62 | 6,045,696 |
| B4 / 前缀 16 | decode | 92 | 5,907,576 |

这些容量都低于 16 MiB，因此本轮模型的重复分配由块数限制触发。16 MiB 约束的是已释放缓冲区的保留量；活跃 Tensor 内存另行统计。更大的模型、更多层或不同形状仍可能触发块数或字节数淘汰，不能承诺所有工作负载都零分配。

## 同进程 A/B

先从原 64 块源码构建独立研究 PYD，只有它提供 64/128 切换。每次切换清空闲置缓存，预热 3 次，再测 5 次；9 轮交替 ABBA / BAAB，每配置每项共 90 个计时样本。两种配置共享同一组编译图、输入、权重、KV 种子和输出语义。实际固定 128 块正式 PYD另行确认。

计时范围为常驻输入的 30 层 FP16 decoder，包含 Python、驱动提交、同步等待及返回对象释放。embedding、最终归一化和输出头在计时之外；decode 固定在同一 KV 种子上。桌面负载未控制，表中收益来自同进程交错对照。

| 普通执行 | 64 块调用 ms | 128 块调用 ms | 64 块调用＋释放 ms | 128 块调用＋释放 ms | 完整步耗时降低 | 胜出轮数 / 9 |
|---|---:|---:|---:|---:|---:|---:|
| B1 prefill | 21.519 | 21.296 | 21.555 | 21.337 | 1.01% | 5 |
| B1 decode | 18.602 | 17.057 | 19.641 | 17.086 | 13.01% | 9 |
| B4 prefill | 33.854 | 33.696 | 33.890 | 33.736 | 0.45% | 5 |
| B4 decode | 19.384 | 17.664 | 20.557 | 17.692 | 13.94% | 8 |

另一个独立进程使用主机计数器 DLL，但不启用设备任务 profiling，重复相同 A/B：

| 主机计数器执行 | 64 块调用＋释放 ms | 128 块调用＋释放 ms | 完整步耗时降低 | 胜出轮数 / 9 |
|---|---:|---:|---:|---:|
| B1 prefill | 21.207 | 21.243 | -0.17% | 4 |
| B1 decode | 19.248 | 16.886 | 12.27% | 9 |
| B4 prefill | 33.202 | 33.239 | -0.11% | 5 |
| B4 decode | 20.154 | 17.439 | 13.47% | 9 |

prefill 的变化约为 -0.2%～1.0%，不视为稳定收益。decode 在两个进程中均有明确收益。原始数据见 `plain.json`、`host.json`。

## 分配与释放计数

每项每配置 90 次计时调用，64 块 decode 各记录 2,520 次驱动分配，即每步 28 次；128 块为 0。主机 DLL 同时证明 64 块每步 28 次 `zeMemAllocShared`、28 次 `zeMemFree`，128 块两者均为 0。

| 64 块 decode 分项中位数 | B1 ms | B4 ms |
|---|---:|---:|
| 驱动分配 | 0.786 | 0.739 |
| 驱动释放 | 0.957 | 0.973 |
| 返回后的对象释放 | 1.026 | 1.046 |

128 块的驱动分配和释放计时均为零；返回后的对象释放降至约 0.03 ms。每个分项单独取中位数，不能直接相加作为整体收益的精确分解。等待时间仍约 12 ms，后续还需继续优化设备任务与提交路径。

所有预热后的计时块没有显式主机载荷复制、重新编译或创建会话；每个完整 decoder 步恰好 30 次图调用。各层图哈希在容量切换前后相同，输出、KV、位置和旧种子按字节对照一致。

## 正式构建与安全验证

正式固定 128 块的 PYD 已完成独立确认，`production/confirmed.json` 记录 8,280 次成功 NPU 调用。预热后四种计时均无新增驱动分配，364 项输出和缓存数组与研究扩展的 128 块版本逐位一致。

| 正式构建的函数调用中位数 | B1 ms | B4 ms |
|---|---:|---:|
| prefill | 21.383 | 33.247 |
| decode | 17.002 | 17.360 |

这张表采用已有普通计时器，排除返回之后的对象释放；收益百分比采用上方同进程 A/B 的调用加释放计时。

已通过的检查：

- allocator 的 5 项检查：精确尺寸复用、最后一个 Storage/view 引用、16 MiB/128 块边界、4 线程复用、持有 Storage 时 shutdown。
- 3 项闲置顺序检查：淘汰最早释放块、中间命中不破坏顺序、最近尺寸可复用。
- 模拟不确定完成后的缓存隔离：不触发硬件故障，不调用驱动 free/reset，缓存及活跃字节遵守 quarantine 规则。
- 33 项真实设备检查，4,160 次 NPU 调用；FP16/FP32 特殊值、转换、归约、广播、矩阵、视图、线程及 shutdown。
- 11 项训练检查，2,732 次 NPU 调用；FP16/FP32 梯度、MLP、卷积、TinyResNet 与不支持行为拒绝。

正常进程关闭后 live/cache/quarantine 字节归零。模拟失败单独记录为失败上下文，原始检查与最终统计在 `checks/`。

最终离线审计通过 2,360 次数组逐位对照，检查构建指纹、各计时块的复制/会话/驱动分配计数、主机分配释放计数、缓存界限及正式代码的单行 C++ 容量改动。完整审计在 `audit.json`。

旧构建与来源快照在 `before/`，新构建在 `after/`，研究扩展在 `diagnostic/`。模型、SHAVE ELF、原生 session DLL 和驱动选择均保持原指纹。本轮新数组使用压缩 NPZ，没有重复保存各层完整图。

## 复现

从项目根目录运行。构建研究扩展会使用已保存的原 64 块来源；同名实验结果会被覆盖。NPU 命令须串行执行。

```powershell
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','build'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','probe','--label','plain'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','probe','--host','--label','host'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','confirm','--label','confirmed'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','check','--check','guards'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','check','--check','order'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','check','--check','failure'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','check','--check','device'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','check','--check','training'
.\run_torch_npu.ps1 -Script .\npunlock-probe\research\cache_blocks_probe.py -ScriptArguments '--case','audit'
```

下一步：重复分配已消除，可在当前构建上继续分析 SHAVE RMSNorm 的标量补偿依赖与寄存器压力；decode 的零碎算子和 30 次同步提交也值得单独测量。通用 FP32 GEMM 仍不是这个 FP16 模型的首要热点。
