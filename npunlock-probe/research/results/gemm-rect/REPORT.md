# Rectangular FP32 SHAVE GEMM and default square dispatch — 2026-10-04

已在 Lunar Lake NPU 8086:643e / driver 32.0.100.5540 / compiler 8.3 上完成矩形分块 GEMM。128/256 完整方阵已接入普通 `a @ b`、`torch.mm`、`out=` 和 Linear 一阶训练的默认设备路径。其他形状继续使用原实现；一般矩形可显式调用 `torch_npu_rect_gemm.matrix_fp32`，返回 FP32 opmath 结果。

## 实现与边界

- M/N/K 均支持 1..256；按 M/N 分块，点积始终遍历完整 K，不增加跨 K 部分和。每块 M/N<=64、K<=256；K=256 时 M<=63，为 A 的四个元数据元素留空间。
- 每个 tile 的完整缓冲区容量为 `align64(max(4+M*K,K*N,M*N))`，最大 16384 个 FP32 元素。A 以 `[12345,M,K,N]` 四个精确 FP32 标量开头，随后是密集行主序 A；B 从第一个元素开始，剩余位置在 NPU 上补零。
- 在 NPU 上各复制为四份，物理载体形状 `(4,2,capacity/2)`。四个 tile 各用一个 SHAVE 计算互不重叠的输出行，整份 C 先置零。已有 NPU 补偿 sum 合并部分 C，再用 DMA 截取有效输出并组装矩阵。没有 M*K*N 展开、远端 CMX 读取或部分主机分配绑定。
- 复用上一轮快速 FP32 加法/乘法与 Neumaier 补偿，保持 K 顺序、次正规数/IEEE 特殊值处理和 `-ffp-contract=off`。单一通用内核 text 为 8144 bytes，小于已验证的 Power 代码窗口。运行时检查元数据为有限整数、尺寸与所有访问范围；主机继续检查完整 FP32 CMX、布局、IO relocation、DMA 链及 barrier。
- 原四 tile patch 只提取成共享容量校验函数，原方阵入口/限制保持有效。仅修改代码窗口和已证明的 DMA ordering 位，不扩大布局、访问范围或缓冲区。
- 图会话按容量复用，当前 M/K/N 由 NPU 上的标量元数据提供。冷启动只上传缓存的形状参数和零常量；预热期间无测得的显式主机载荷复制或新编译。输入数据的切片、转置、补零、复制、归约和拼装均在 NPU 上完成。
- FP16 输入仍先在 NPU 上转 FP32，再做一次最终输出转换。这是 SHAVE FP32 GEMM，没有使用 DPU FP16/INT8 矩阵引擎。无需 OpenVINO Python/runtime，仍依赖驱动编译器、Level Zero 和离线 MoviTools。

## 正确性和训练

38 组数值/API 检查全部通过：30 组矩形/尾块/完整转置、次正规数、抵消、精度探针、Inf/NaN 和随机 uint32 位模式；128/256 方阵各 FP32/FP16 共 4 组；默认 `mm`/`@`/`out=` 共 4 组。所有输出与旧路径逐位一致。有限值另对照 FP64 参考，普通随机矩阵最大绝对误差约 5.72e-8；精度探针的累计误差另逐项记录，不声称所有点积正确舍入。

三种训练配置各 FP32/FP16、各 3 步 SGD 均通过：

1. batch16 的 48->24->7 ReLU MLP：非方阵前向、输入/参数梯度。
2. batch65 的 96->67->13 ReLU MLP：前后向跨 M/N 分块，含尾块和转置。
3. batch128 的 Linear(128,128)：通过普通默认设备路径执行，无实验内核替换。

loss、输入梯度、参数梯度及 SGD 更新与旧 NPU 实现逐位一致，并通过 CPU 参考容差检查。默认 128 Linear 的 CPU 最大绝对差 FP32 约 1.86e-9，FP16 为 0。一般矩形训练实验只在子进程内临时替换矩阵函数，结束恢复。

47 项离线 Python 检查通过；33 项设备回归检查通过，包括存储/生命周期、广播、归约、矩阵、精度、拒绝条件和线程检查。六个隔离矩形测试进程共 31015 次 NPU 提交，额外设备回归 4160 次。全部退出码 0，context_failed=false；关闭后 live_buffers/live_buffer_bytes/cached_sessions 均归零。性能进程峰值约 2.68 MiB 设备缓冲区、380 个缓存图，未触发 512 图上限或驱逐。

## 性能

同进程、相同 resident A/B、预热 2 次，再计数一次，7 轮交错旧/新顺序。以下为 FP32 同步端到端中位数 ms，包含 Python、分块、NPU 打包/复制/归约/输出拼装及 graph fence，排除输入上传、结果下载和编译。CPU 为 Torch 8 线程，仅供量级参照。桌面负载未控制，用户此前报告视频播放；这不是严格空闲基准。

| 矩阵 | 旧 NPU ms | 矩形 SHAVE ms | CPU ms | 图提交 | 旧/新速度比 |
|---|---:|---:|---:|---|---:|
| 16x48 @ 48x24 | 3.989 | 10.650 | 0.042 | 4 -> 8 | 0.37x |
| 128x96 @ 96x48 | 68.357 | 62.843 | 0.067 | 51 -> 21 | 1.09x |
| 128x128 @ 128x128 | 509.921 | 192.682 | 0.187 | 161 -> 52 | 2.65x |
| 256x256 @ 256x256 | 3963.295 | 1537.416 | 0.291 | 1030 -> 185 | 2.58x |

128/256 方阵约快 2.65/2.58 倍，因此默认分流仅启用这两个已测形状。128x96 @ 96x48 的约 9% 差异可能受负载波动影响，没有据此扩大默认范围。小矩形多了打包和同步，反而变慢；其旧路径保持有效。默认分流是按逻辑尺寸选择，与 FP16/FP32 输入一致；FP16 的转换也由 NPU 执行。

当前仍远慢于 CPU，不能把 NPU 路径内部加速等同于平台训练性能超过 CPU/GPU。训练结果证明前后向和更新正确，本轮没有测量完整训练吞吐量。

## 文件和复现

```powershell
Set-Location E:\Pytorch_xfire_NGPU
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_rect_gemm.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_rect_gemm.py
.\run_torch_npu.ps1 -Script torch_npu_rect_gemm_demo.py
.\.conda\npunlock\python.exe -m unittest discover -s npunlock/tests/python -p test_*.py
.\run_torch_npu.ps1 -Device -Quick
```

普通 API：先导入 `torch_intel_npu`，将匹配 dtype 的完整 128/256 方阵 `.to('intel_npu')`，随后照常 `a @ b`。其他支持的矩阵仍走旧 NPU 路径。显式矩形实验入口限 M/N/K<=256，需在 `no_grad` 或已有 ATen/autograd 路径内调用；它本身不是独立 autograd Function。

- `rect-numeric/large/training/training-tiled/performance/dispatch.json` 和对应 `.log`：逐进程数值、训练、采样计时、指纹及关闭记录。
- `summary.json`：最终范围、检查计数、当前源码哈希与默认分流形状。
- `carrier-validation.json`、`capacity-*.blob`：从 64 到 16384 容量的完整载体编译检查；原 Power 只用作载体，不执行。
- `first/second/third-validation-demo.py`：检查脚本的历史版本，匹配各进程记录的源哈希。测试期间修正了 CPU 参数快照的 clone，并增加分块及默认接入检查；最终计算模块仅更新说明文字，内核/矩形算法未改变。
- `rect-module-before-doc.py`：各实验进程所用计算模块，最终版本只改了模块说明文字。
- `../../gemm-rect-before/`：本轮前源码/设备报告快照。旧 `_matrix_fp32` 的 AST 与保留的 `_matrix_fp32_expanded` 完全一致（除函数名），新增选择器单独通过普通 API 和训练检查。
- `npunlock/tests/fixtures/npu4000/rect_gemm.c`、`gemm_rect/`：内核源码、ELF、汇编、manifest 和离线载体。
- `torch_npu_rect_gemm.py`：矩形执行；`torch_intel_npu.py`：已测方阵默认分流。

下一步先减少每块的补零/打包和完整容量归约开销，再检查每 tile 两个 SHAVE 的八路布局。以非方阵训练的端到端数据决定是否扩大默认分流；尚未实现 K>256 的新内核、原生 FP16 输入、任意模型覆盖或 GPU 协同。
