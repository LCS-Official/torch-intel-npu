# Parallel direct FP32 SHAVE GEMM — 2026-10-04

本轮已在 Lunar Lake NPU 8086:643e / driver 32.0.100.5540 / compiler 8.3 上实现并验证单 tile 双 SHAVE 和四 tile GEMM。64x64 四 tile 已快于旧 NPU 路径，但 32x32 未加速。普通 `a @ b` 保持原实现，实验内只临时替换 `_matrix_fp32`，退出恢复。

## 实现

- 单 tile 双任务：边长 2..64 的完整 FP32 方阵。原编译器将行分为两个 SHAVE 任务，A/C 使用互不重叠的行；B 从各任务的地址回推至已完整上传的 B 首地址。整矩阵 CMX 范围和输入 DMA 完成关系必须经主机验证，拒绝部分上传或错误依赖；Level Zero 仍绑定完整拥有的分配。
- 四 tile：边长 8/16/32/64。在 NPU 上将完整 A/B 分别复制成四份，各 tile 用一个 SHAVE 计算 `row % 4 == tile` 的行，其他行写零，再由已有 NPU sum 内核合并四份 C。此路线为四路 SHAVE，并非八路。数据量 O(4*S*S)，无需 M*K*N 展开，也没有跨 tile CMX 读取。
- 四 tile 端到端需要四次图提交：两次输入广播、一次 GEMM、一次结果归约。计时包含这些 NPU 数据搬运和临时分配。
- `parallel1` 为单输出列循环，`parallel4` 每次处理四列；`parallel_fast` 再将普通 FP32 加法/乘法路径内联，稀有 IEEE/次正规数处理保留原辅助函数。判定条件与 `arithmetic.c` 相同，K 顺序及 Neumaier 补偿累加不变，`-ffp-contract=off` 禁止乘加融合。四 tile 使用相同快速计算循环。
- fast / four-tile 内核 text 为 8464 / 8544 bytes，均低于已验证的 Power 代码窗口 21216 bytes，无额外 kernel data。首次将全部辅助代码强制内联导致超出窗口，构建检查拒绝；最终只内联普通路径。
- patch 复用原有设备/编译器 profile、原代码指纹、FP32 CMX、密集布局及输出不重叠检查，并额外验证完整字节复制、IO relocation、DMA 链和完成 barrier。仅允许替换代码窗口及强制 DMA in-order 的 ordering 位，不改写尺寸、范围、地址或 barrier。单 tile 修改最后 B 上传的 ordering 位；四 tile 强制所有矩阵输入/输出 DMA in-order。
- 无 OpenVINO Python/runtime 依赖，仍需要 Intel 驱动编译器、Level Zero 和离线 MoviTools 构建链。

## 正确性与资源

114 组检查全部通过：单 tile 三种内核各 32 组、四 tile 18 组。包含 2/3/4/5/7/8/16/17/31/32/64 普通尺寸、FP32 精度区分、次正规数、抵消、Inf/NaN、随机 uint32 位模式，以及加法/乘法快速路径边界。所有结果与旧 NPU 路径的 uint32 位模式一致；有限值另对照 FP64 计算后转 FP32 的参考误差。这是已测用例的逐位一致，不能据此证明所有输入均正确舍入。

16x16 Linear + MSE + SGD：FP32/FP16 各 3 步，loss、输入梯度、参数梯度及更新后的参数与旧路径逐位一致。该训练使用四 tile 路线；非支持形状仍在 NPU 上走旧矩阵实现。FP16 依旧由 NPU 先转 FP32 计算、最终一次转回 FP16，并非原生 FP16-input GEMM。

离线 Python 46 项检查通过，新增完整复制/依赖证明及错误宽度、错误地址、缺失 barrier、错误尺寸和载体的拒绝检查。完整硬件验证进程退出码 0，2691 次 NPU 提交，context_failed=false。关闭后 live_buffers/live_buffer_bytes/cached_sessions 均为 0；关闭前仅保留 4 个缓存标量。预热计时中新增显式主机载荷复制和编译均为 0；计数不包含驱动内部 DMA。

## 性能

同一进程、相同 resident A/B、预热 3 次、9 轮按版本交错/反转顺序测量，以下为同步端到端中位数 ms。排除上传、下载和编译，包含 Python 调用、图 fence、NPU 广播/归约及临时分配。CPU 为 Torch 8 线程，仅供量级参照。桌面负载没有控制，用户此前报告视频播放；本轮不是严格空闲基准，不能将跨轮差异都归因于内核。

| 方阵 | 旧 NPU | 串行直接 | 双任务四列 | 双任务 fast | 四 tile fast | CPU | 本轮最快 NPU 路线 |
|---|---:|---:|---:|---:|---:|---:|---|
| 8x8 | 2.013 | 0.851 | 0.591 | 0.468 | 1.934 | 0.027 | parallel_fast |
| 16x16 | 3.597 | 4.055 | 2.177 | 1.641 | 3.661 | 0.045 | parallel_fast |
| 32x32 | 4.583 | 18.368 | 9.166 | 5.000 | 6.181 | 0.056 | old |
| 64x64 | 29.888 | 140.897 | 67.936 | 34.864 | 21.019 | 0.133 | four_tile |

64x64 从串行直接约 141 ms 降到四 tile 约 21 ms，较同轮旧路径约 29.9 ms 快约 1.42 倍，图提交从 21 次降为 4 次。上一轮四 tile 同样约 21.0 ms，旧路径约 27.2 ms（约 1.29 倍）。旧路径耗时有波动，因此采用同轮交错比较。

8x8/16x16 单 tile fast 分别约 0.47/1.64 ms，四 tile 的额外复制及同步成本抵消了计算收益。32x32 最快的直接变体仍未跑赢旧路径，因此没有启用统一的默认替换。上述小矩阵 CPU 仍明显更快，当前优化不代表训练已快于 CPU/GPU。

## 边界与下一步

仍限完整密集 FP32 小方阵、指定本机 profile 和可证明的 compiler carrier；不提供任意形状、批量、原生 FP16 输入或异步矩阵执行。256x256、卷积/ResNet 训练的主要 GEMM 瓶颈尚未解决。四 tile 借助复制和归约避开跨 tile 访问，额外内存/提交成本已包含在测试内。

下一步先将相同内核扩展到有界矩形 M/N/K tile，沿用完整输入绑定与严格 DMA 校验，跑通非方阵 Linear 前后向和 128/256 方阵，再根据端到端实测为支持的形状选择路线。性能分流后再考虑原生 FP16 输入；不能直接推广本轮四 tile 到全部矩阵。

## 复现与文件

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_parallel_gemm.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_four_tile_gemm.py
.\run_torch_npu.ps1 -Script torch_npu_gemm_demo.py -ScriptArguments '--parallel'
.\.conda\npunlock\python.exe -m unittest discover -s npunlock/tests/python -p test_*.py
```

- `square-gemm-validation.json/.log`：最终硬件检查、训练、计时、源代码和 ELF 哈希、资源关闭记录。
- `square-*-*-contract.json`：各尺寸/内核的代码替换、完整输入跨度、DMA ordering 与 barrier 证明。
- `four-tiles-*.blob/.json`：四 tile 原始载体及 ELF 布局检查；仅编译检查，不执行原 Power 运算。
- `before-inline/`：第一轮双任务内核和测试的快照。`../../gemm-parallel-before/` 保存本轮修改前源码；`../gemm/REPORT.md` 保存原串行结果。
- `npunlock/tests/fixtures/npu4000/square_gemm_parallel.c`、`gemm_parallel/`：源代码、四个 ELF、汇编、构建哈希和离线载体样本。
- `npunlock.npu4000.compile_parallel_square_gemm` / `compile_four_tile_square_gemm`：实验低层编译入口。四 tile 入口只返回部分结果图，复制输入与 NPU 合并由 demo 组合完成。
- `run_torch_npu.ps1` 的 `-ScriptArguments` 支持传递实验参数。

DMA ordering 的公开 ABI 依据：[Intel NPU40XX DMA header](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_dma_hw_40xx.h)。实际载体是否满足该契约仍以本机 ELF 严格检查为准。
