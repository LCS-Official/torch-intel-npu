# Direct FP32 SHAVE GEMM prototype — 2026-10-04

已验证完整 A/B -> C 的小方阵 FP32 内核。当前保留为实验路径，普通 PyTorch `a @ b` 仍使用既有实现。

## 实现与边界

- 支持边长 2..64 的两个完整、连续 FP32 方阵；单 SHAVE invocation，不展开 M*K*N 乘积张量。
- 使用驱动 `ReferenceSW` + `NPU_TILES=1` 编译 Power carrier，再替换其指纹已验证的代码窗口。原 Power 运算不会执行。
- 编译后检查设备 8086:643e / compiler 8.3、原代码指纹、完整行主序尺寸/步长、FP32 CMX 地址、缓冲区范围、输入输出不重叠以及恰好一次调用。拒绝普通分块 carrier。
- 复用已有 `mul32` / `add32`，包括次正规数修正；FP32 Neumaier 补偿累加，编译关闭浮点乘加融合。内核 text 为 6320 bytes，无额外 kernel data。
- 不依赖 OpenVINO Python/runtime 包；仍依赖 Intel 驱动编译器、Level Zero 和现有 MoviTools 内核构建链。
- 当前未实现矩形 GEMM、原生 FP16 输入内核或多 SHAVE 并行 GEMM。FP16 训练实验仍先在 NPU 上转为 FP32，输出沿用一次最终转回 FP16。

## 检查结果

23 组数值检查全部通过，包括 11 个普通尺寸 2/3/4/5/7/8/16/17/31/32/64，以及 4/16/64 三种尺寸的 FP32 区分探针、次正规数、抵消、Inf/NaN。输出与旧 NPU 路径按 uint32 位模式逐位一致；有限值另对照 FP64 计算后的 FP32 结果。

16x16 Linear + MSE + SGD 的 FP32/FP16 训练各 3 步通过，比较 loss、输入梯度、参数梯度和更新后的参数，均与旧路径逐位一致。训练仅在实验脚本内临时替换 `_matrix_fp32`，退出恢复。

共 1305 次 NPU 提交，最终进程退出码 0，context_failed=false，存活缓冲区和字节均为 0。离线 Python 检查 44 项通过，包含分块载体、错误尺寸和代码指纹的拒绝校验。

## 性能

同一进程交错 A/B，预热 3 次、测量 9 对，表中为中位数毫秒；同步 fence 和 Python 提交开销包含在内，上传、下载及编译排除。预热测量期间新增主机载荷复制和编译均为 0。CPU 为 8 个 Torch 线程；CPU 时间很短，仅供量级参照。本轮桌面负载未控制，用户此前报告视频播放，不视为严格空闲基准。

| 方阵 | 旧 NPU ms | 直接 GEMM ms | CPU ms | 图提交次数 | 旧/新速度比 |
|---|---:|---:|---:|---|---:|
| 8x8 | 2.132 | 0.609 | 0.035 | 4 -> 1 | 3.50x |
| 16x16 | 1.901 | 2.646 | 0.032 | 4 -> 1 | 0.72x |
| 32x32 | 4.604 | 18.293 | 0.045 | 4 -> 1 | 0.25x |
| 64x64 | 26.567 | 140.741 | 0.064 | 21 -> 1 | 0.19x |

提交数减少并不自动带来加速。直接内核在单次 SHAVE 中串行完成所有点积，旧 carrier 能分到多个 SHAVE 调用。并行度丢失和标量补偿循环很可能是大尺寸变慢的主要因素，但本轮未对纯设备周期进行分项测量，尚未证明各因素占比。64x64 变慢约 5.3 倍，因此没有启用为默认后端优化。

下一阶段先尝试在完整输入绑定前提下保留多个 SHAVE 的并行输出分块，并改善内核每次循环处理的输出数量；跑赢当前路径后再推广矩形矩阵和 FP16 输入。256x256 的性能瓶颈尚未解决。

## 复现与文件

```powershell
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_gemm_carrier.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_square_gemm.py
.\run_torch_npu.ps1 -Script torch_npu_gemm_demo.py
.\.conda\npunlock\python.exe -m unittest discover -s npunlock/tests/python -p test_*.py
```

- `carrier-summary.json`：默认/单 tile/ReferenceSW 的 15 个 Power 编译检查；另 4 个普通 ReferenceSW MatMul 仍降到 FP16，均拒绝采用。
- `square-gemm-validation.json` / `.log`：最终数值、训练、计时、源代码哈希与资源记录。
- `square-*-contract.json`：各尺寸已验证的 ABI 和代码替换范围。
- `first-pass/`：首次数值/计时通过、训练检查遇到尚未实现的 `pow` 算子后的记录；实验 MSE 改用已支持的乘法。
- `../../gemm-before/`：本次修改前的核心源码及哈希快照。
- `npunlock/tests/fixtures/npu4000/square_gemm.c`、`gemm/`：源代码、ELF、汇编、manifest 及离线载体样本。
- `npunlock.npu4000.compile_square_gemm(size, kernel_elf, libraries=...)`：实验低层入口，可使用现有 Program/session 运行接口。

参考公开编译器源代码：[ReferenceSW pipeline](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/NPU40XX/dialect/VPUIP/transforms/pipelines.cpp)。本机驱动是否保持完整 FP32 输入，以本目录的实际 ELF 检查结果为依据。
