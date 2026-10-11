# Eight-task FP32 SHAVE GEMM — 2026-10-04

在本机 Lunar Lake NPU 8086:643e、驱动 32.0.100.5540、图编译器 8.3 上，矩形 GEMM 已支持四个 tile 各两个完整输入的 SHAVE 任务。单次矩阵调用复用打包数据，以 DMA 选取输出行，保留真实 FP32 和完整 K 的补偿累加顺序。普通 Torch API 的 128/256 完整方阵默认切换到八路；其他形状继续使用原 NPU 路径。

## 同轮性能

同进程、相同常驻 A/B，预热各 2 次，再计数一次，7 轮旋转及反转执行顺序。以下为 FP32 同步端到端中位数 ms，包含 Python、分块、NPU 打包/复制/输出拼装和每图 fence，排除输入上传、输出下载和冷编译。CPU 是 Torch 8 线程参考；桌面负载未控制，不是严格空闲基准。

| 矩阵 | 原展开 NPU | 上版四路 | 本版四路 | 本版八路 | CPU | 上版四路/八路 |
|---|---:|---:|---:|---:|---:|---:|
| 16x48 @ 48x24 | 3.739 | 8.363 | 5.430 | 4.656 | 0.035 | 1.80x |
| 128x96 @ 96x48 | 64.544 | 61.807 | 43.389 | 27.275 | 0.059 | 2.27x |
| 128x128 @ 128x128 | 509.810 | 191.271 | 142.561 | 82.487 | 0.197 | 2.32x |
| 256x256 @ 256x256 | 4010.110 | 1533.816 | 1067.118 | 598.852 | 0.268 | 2.56x |

“上版四路”加载 `../../gemm-eight-before/torch_npu_rect_gemm.py`，复现上轮逐块打包及全容量归约。它与“本版四路”共享本轮重建的四路内核；这样比较的是主机调度/打包/合并改动，不声称使用上轮旧 ELF。“本版八路”使用独立的八任务 ELF。四种输出均与原展开 NPU 路径逐位一致。

图提交数（原展开 / 上版四路 / 本版四路 / 八路）：16x48x24 为 4/8/7/9；128x96x48 为 51/21/16/20；128 方阵为 161/52/36/44；256 方阵为 1030/185/104/144。八路需要额外的 DMA 转置/截取，提交数比本版四路多，但并行计算收益足以抵消。各预热采样期间，测得的显式主机上传/下载/设备复制和新增编译均为零；计数器不覆盖驱动内部 DMA。

128/256 方阵相对原展开路径约快 6.18/6.70 倍，相对上版四路约快 2.32/2.56 倍。小矩阵仍比原展开路径慢，默认分流没有扩大。CPU 在这些尺寸上仍快得多；本轮没有测量完整训练吞吐量。

## 改动和验证边界

- 打包缓存仅存在于一次 `matrix_fp32` 调用内：A 按行块、输出列数和容量复用，B 按列块和容量复用。SGD 下一次调用重新读取输入，不保留跨调用的权重副本。
- 四路和八路均由 DMA 选取每行的唯一生产者，省去完整部分 C 的浮点归约。四路 Gather 源最多 65536 项；八路完整部分 C 最多 131072 项，先用已有整块 DMA 转置并截取为最多 32768 项，再 Gather 最多 4096 项。原 Gather 源上限 65536、已有运行输出上限 8192 均未扩大。
- 八路载体形状为 `(8,8,capacity/8)`，每 tile 两份完整 A/B/C。编译器将同 tile 的两份相邻矩阵合为一条完整 DMA，校验宽度为 `2*capacity*4` bytes，IO relocation 对应八份完整分配。每个 SHAVE 只读自己完整的 A/B，不依赖远端 CMX。
- 布局探测发现四副本载体的某些形状虽生成八任务，却只给每个任务部分输入；这些候选仅编译检查，没有执行。最终采用八完整副本；1152、8256、16384 容量的布局及补丁合同均通过离线检查。
- 继续校验设备/编译器 profile、Power 指纹、FP32、CMX、密集布局、访问范围、IO relocation、队列链及 SHAVE barrier。每条输入/输出矩阵 DMA 共 12 个 ordering 位保留原验证方式；只修改代码窗口和这些位。
- 通用内核以行号 modulo 4/8 分工，每个副本先置零，再计算自己的行。FP32 加法/乘法、次正规数、IEEE 特殊值、Neumaier 补偿及完整 K 顺序保持原实现，仍用 `-ffp-contract=off`。四路/八路代码分别为 8144/8240 bytes。
- M/N/K 的显式实验范围仍为 1..256；每块 M/N<=64，K<=256。K=256 时 M<=63，为 A 的四个形状参数留空间。FP16 输入仍在 NPU 转为 FP32，最终一次转回 FP16；尚未使用原生 FP16 或 DPU 矩阵引擎。
- 未增加 OpenVINO Python/runtime 依赖；仍依赖 Level Zero、Intel 驱动图编译器和离线 MoviTools。没有扩大到任意模型、K>256 的直接内核或 GPU 协同。

## 正确性和训练

八路 38 组检查通过：30 组矩形、尾块、完整转置、精度探针、抵消、次正规数、Inf/NaN 和随机 uint32 位模式；128/256 方阵 FP32/FP16 共 4 组；普通 `torch.mm`、`@`、`out=` 共 4 组。所有输出与旧 NPU 路径逐位一致，别名输出在提交前拒绝。有限值另对照 FP64 参考；普通随机大矩阵最大绝对误差约 5.72e-8，不声称点积总是正确舍入。优化四路另通过同样 30 组数值检查。

以下三种配置各 FP32/FP16、各连续 3 步 SGD 均通过，loss、输入/参数梯度和权重更新与旧 NPU 路径逐位一致，且满足 CPU 参考容差：

1. batch16，48→24→7 ReLU MLP。
2. batch65，96→67→13 ReLU MLP，前后向含跨块、尾块和转置。
3. batch128，Linear(128,128)，普通默认设备路径，无临时替换矩阵函数。

CPU 最大绝对差：上述 FP32 配置分别约 3.73e-9、3.73e-9、1.86e-9；FP16 分别为 0、2.38e-7、0。这些检查覆盖连续更新，确认打包复用不会造成陈旧权重。

48 项离线 Python 检查通过，包括八路不完整输入、错误 DMA 地址/宽度、无效依赖和副本数的拒绝；33 项通用设备回归通过。六个主实验进程共 34569 次 NPU 提交，四路兼容数值进程 993 次，设备回归 4160 次。全部退出码 0，`context_failed=false`，关闭后 `live_buffers/live_buffer_bytes/cached_sessions` 均为 0。性能进程同时比较四条路径，峰值设备缓冲区约 7.88 MiB、缓存图 407 个，未触发 512 图上限；这不构成单独八路的峰值内存基准。

## 复现与记录

```powershell
Set-Location E:\Pytorch_xfire_NGPU
# 已保存 ELF，普通使用无需重新构建
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_rect_gemm.py
# 只编译检查布局，不执行候选 Power
.\run_torch_npu.ps1 -Script npunlock-probe/research/probe_eight_gemm.py
.\run_torch_npu.ps1 -Script torch_npu_rect_gemm_demo.py
.\run_torch_npu.ps1 -Script torch_npu_rect_gemm_demo.py -ScriptArguments '--case','numeric','--four'
.\.conda\npunlock\python.exe -m unittest discover -s npunlock/tests/python -p test_*.py
.\run_torch_npu.ps1 -Device -Quick
```

普通使用仍是导入 `torch_intel_npu`，将 Tensor/model `.to('intel_npu')`，然后按既有支持范围使用 Torch API。128/256 完整方阵自动使用八路。显式 `torch_npu_rect_gemm.matrix_fp32(a,b,replicas=4或8)` 返回 FP32，需在 `no_grad` 或既有 ATen/autograd 路径内调用，它不是独立 autograd Function。

- `rect-numeric/large/training/training-tiled/performance/dispatch.json` 和 `.log`：逐进程结果、采样值、源码指纹与关闭状态；`four/rect-numeric.*` 是四路兼容记录。
- `device-regression.*`：本轮设备回归；上轮同名报告已保存在 `../../gemm-eight-before/`。
- `summary.json`：范围、检查计数、最终源码指纹和性能汇总。
- `layout-4rep-summary.json`：未执行的初期四副本布局；`layout-summary.json`、`capacity-*-replicas-8.blob/.json` 与 `capacity-*-contract.json`：八份完整输入的编译布局和补丁证据。
- `numeric-demo.py`：数值检查的早期脚本，之后只扩展性能比较；`module-before-default.py` 与最终模块仅默认副本数 4→8 不同；`npu4000-before-doc.py` 与最终模块仅修正文档字符串。各记录源哈希都能匹配最终文件或这些快照。
- `../../gemm-eight-before/`：本轮前源码和哈希；原 Torch 选择器未改变，仅被调用矩形函数的默认副本数变为八。
- `npunlock/tests/fixtures/npu4000/rect_gemm.c`、`gemm_rect/`、`gemm_eight/`：通用源码、各自 ELF、汇编、manifest 和离线载体；`torch_npu_rect_gemm.py` 是执行入口。

进一步提速需要区分真实 FP32 算术成本、重复完整输入 DMA 和逐图同步开销。原生 FP16/DPU 路线可另行探索，以保留现有 FP32 精度语义；本轮不据小矩阵结果扩大默认范围。
