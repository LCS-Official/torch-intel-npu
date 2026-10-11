# RMSNorm 同载体消融：计算、搬运与输出分配

2026-10-05，Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动
32.0.100.5540，编译器 8.3。结果：`passed`。

当前大输入的主要优化目标是 SHAVE 的转换、舍入和输出处理，不能把整体慢归因于
主机 IO。小输入的固定提交/同步/搬运开销占比明显。此次只新增隔离诊断工具，
没有修改默认 RMSNorm、设备后端、DLL 或推理调用路径，也没有新增依赖。

## 如何保持对照可比

`build_rmsnorm_diagnostic.py` 复用现有 4000xx 编译/汇编/链接工具、
`arithmetic.c`、`resident_activation.c` 和 RMSNorm 的 memref/转换校验。
诊断 ELF 单独写到本目录，原默认 C/ELF 不覆盖。

`rmsnorm_diagnostic.py` 对每个行数只编译一个已验证的完整行 Power carrier，
使用其原始未替换 blob，为八种代码分别替换同一个 text 窗口。
将整个代码窗口置零后，八份 blob 的 SHA256 必须相同。
因此输入/输出规格、CMX 布局、DMA、barrier、任务和调度字节全部保持一致。
仍使用 1..7 行的单 tile ReferenceSW、>=8 行的四 tile 完整行模式。
不会让编译器因算子不同而改成另一套搬运或分块方案。

| 代码 | 计算与输出 |
|---|---|
| fill | 写 FP16 +0，不读取 x/gamma 内容；载体仍保留两个输入 DMA |
| copy | 逐元素按原始 16 位编码复制 x |
| convert | FP16→FP32→FP16，使用原正常值及边界转换函数 |
| affine | 解码 x/gamma、FP32 相乘、舍入 FP16 |
| epilogue | 单位归一化系数，保留两次 FP16 舍入及 gamma 乘法 |
| mean | 原 FP32 平方、Neumaier 补偿累加、除 576、加 eps；将结果转 FP16 后填满该行 |
| inverse | 上述归约再加原精确 FP32 sqrt 和倒数；转 FP16 后填满该行 |
| full | 未改变的完整生产 RMSNorm ELF |

Epilogue 使用 `1 + gamma[first]*0` 得到运行时单位系数，保持有限 gamma 下
结果为 1，同时避免将归一化乘法直接常量消除；汇编保留 SAU.MUL.f32。
它是输出循环的对照，不等于生产内核按每行方差得到的系数。因此其输出数值分布、
边界路径与完整 norm 可能不同，不能将八个版本的耗时差值严格相加。

Fill 初始化全部输出，不使用未定义输出的“空内核”。Copy/Fill 都仍包含 Python/C++
调用、参数绑定、同步提交、相同的驱动 DMA 和轻量 CMX 读写，**不是纯 DMA 耗时**。
本轮没有硬件 task 时间戳或独立带宽测量。

## 数值和清理

共 256 项检查。随机输入覆盖行数 1、7、8、9、17、32、256、259、512，
分别使用模型第 0 层两个真实 norm gamma；另检查零、FP16 subnormal/signed zero、
交替 ±65504，以及两种性能输入。每个复用输出在数值检查前填为 7，检查是否全部覆盖。

Fill、Copy、Convert、Affine、Epilogue 共 160 项要求 CPU 参考位模式完全一致；
Mean、Inverse、Full 共 96 项使用既有 `rtol=.002, atol=.001`。
Mean 的 ±65504 测试会在最后的 FP16 诊断输出中成为 Inf，符合 CPU 参考；
内部平方、累加及完整 norm 仍为 FP32。这没有改变生产路径的精度契约。

共执行 4,864 次 NPU 图提交；其中计时区域 3,968 次，预热 640 次，数值检查 256 次。
计时区域无显式主机上传、下载、packing 或新建会话。每次新输出模式确实增加一次
NPU buffer allocation，复用模式不增加 allocation。
结束后 live buffer、隔离 buffer、缓存会话均为零，context 未失败。

`diagnostic.json` 保存源码、C++ build、DLL、包版本、原始采样、布局与修改证据，
`build-manifest.json` 保存各 C/ELF/汇编和生成源的 SHA256；`sources/` 为源码快照。
默认完整 ELF 仍为
`1c19b63d8e083e59986304f5d40467746950d5d585399591bb2942db3e3e6a47`。

## 计时方法与实测

同一个进程比较八种代码 × 两种输出方式：每次新建输出（fresh，计入分配和释放）
或复用已分配输出（reuse）。输入和 gamma 均已在 NPU，图和会话已准备。
每种方式预热 5 次，再进行 31 轮轮换顺序计时。CPU 8 线程，记录中位数和排序
近似 P10/P90。主机没有输入/输出复制发生在计时区域。

输入有单位尺度随机值与真实 embedding 两组。真实数据来自已有
`perf-b4-s64-w8a16.npz` 的 host，RMS 约 0.1102；1/32 行取前部，512 行重复两份。
使用真实 input_layernorm gamma。原始 NPZ 在 `real-input.npz`，hash 在 JSON 中。
这组结果针对第 0 层 norm 的输入分布，不代表所有深层或特殊值的运行成本。

以下为真实 embedding、复用输出、每次 norm 图的中位数，单位 ms：

| 计算 | 1 行 | 32 行 | 256 行 | 512 行 |
|---|---:|---:|---:|---:|
| Fill | 0.289 | 0.311 | 0.429 | 0.620 |
| Copy | 0.294 | 0.295 | 0.577 | 0.829 |
| Convert | 0.328 | 0.397 | 1.112 | 1.986 |
| Affine | 0.411 | 0.481 | 1.656 | 3.017 |
| Epilogue | 0.407 | 0.530 | 2.106 | 3.811 |
| Mean | 0.351 | 0.381 | 1.134 | 2.029 |
| Inverse | 0.367 | 0.371 | 1.186 | 2.085 |
| Full | 0.528 | 0.725 | 3.753 | 7.247 |

256 行 Copy 约 0.577 ms，Full 约 3.753 ms；512 行约 0.829 vs 7.247 ms。
同一载体的轻量拷贝比完整 norm 快约 6.5/8.7 倍，说明当前运行不能由搬运基线解释，
自定义代码增加了显著成本。Copy/Full 比值约 15%/11% 只是一种对照，
不是测得的“IO 占比”。改变 kernel 也会改变设备状态、寄存器占用和执行行为。

256/512 行的 Epilogue 为 2.106/3.811 ms，Mean 为 1.134/2.029 ms，
Inverse 为 1.186/2.085 ms。转换与输出循环值得优先优化；单独将平方归约向量化
不一定带来明显整体收益，与上一轮向量原型结果相符。
Inverse 和 Mean 的差距很小，部分小尺寸甚至反向，不能把差值当作精确开方时间；
目前没有证据表明精确 sqrt/reciprocal 是首要瓶颈。

256 行的 Full P10/P90 约 3.672/3.919 ms，Copy 约 0.493/0.705 ms；
512 行 Full 约 7.157/7.499 ms，Copy 约 0.790/0.986 ms。
主要层级差距大于本轮计时波动，但几十微秒的差异不足以支持细粒度归因。
桌面负载和设备频率没有锁定，所有结论以本轮相同进程的对照为依据，
不与先前不同运行的绝对耗时拼接。

## 输出分配的影响

真实 embedding 下完整 norm：

| 行数 | 每次新输出 ms | 复用输出 ms | 相对减少 |
|---|---:|---:|---:|
| 1 | 0.596 | 0.528 | 11.4% |
| 32 | 0.851 | 0.725 | 14.8% |
| 256 | 3.908 | 3.753 | 3.9% |
| 512 | 7.372 | 7.247 | 1.7% |

复用输出帮助小输入，但不足以解决大输入的 norm 计算成本。
这里的差距包括分配/释放、地址绑定与缓存状态变化，并非纯 `zeMemAllocShared`
单函数耗时。Python `_run`、C++ allocator、native runtime 的源码快照可用于继续定位。
当前源码中的提交函数复用 context queue，但每次创建/销毁 command list 和 fence，
并等待 fence 完成；本轮没有测这些 API 各自的时长。

## 下一步

优先研究正常范围的批量 FP16 解码与舍入，以及输出循环的向量化；特殊值继续保留
现有验证过的路径，并要求 norm 与旧实现逐位对照。上一轮 half 向量转换错误应继续
作为工具链检查项，不能直接依赖那条 builtin。
其次考虑固定形状推理的输出复用和提交对象复用，收益更偏向小输入。
设备内部 DDR/CMX 与 driver 时间若要精确拆分，需要额外 profiler/task 时间戳。

本轮没有把这些诊断版本接入模型，也没有增加训练、KV cache、全模型 NPU 或
CPU/GPU/NPU 协同功能。数值和性能检查都是已有 PrivateUse1 后端上的真实 NPU 执行。

在 `E:\Pytorch_xfire_NGPU` 下串行复现：

```powershell
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_rmsnorm_diagnostic.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/rmsnorm_diagnostic.py
```
