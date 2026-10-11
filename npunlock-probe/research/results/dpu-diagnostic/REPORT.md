# Intel NPU DPU 瓶颈诊断与 OpenVINO 对照

2026-10-06；Core Ultra 7 258V / Lunar Lake，NPU 8086:643e，驱动 32.0.100.5540。

当前可以确定的优化重点是复用执行管线、缩减输入量化图，以及按形状选择 tile 数。
OpenVINO 在同一编译图的交替测试中，比当前 npunlock 的标准提交路径减少约
19%～37% 的延迟；诊断 DLL 的固定地址复用路径又比 OpenVINO 减少约 7%～21%。
这说明运行时的改进有直接价值。实验未替换正式 runtime，也未接入常规 Torch API。

本轮的“FP16-FQ”指 FP16 输入经 FakeQuantize 转为 INT8 计算，仍是 W8A8 矩乘；
“prequant-U8”指提前准备相同量化编码，以 U8 作为图输入。
两者都是 FP16 输出。这里的 TOPS 不能标成 FP16 TFLOPS。

## 同一个编译 blob 的运行时对照

先由 OpenVINO 编译相同的已有 IR，再将导出的原始 ELF blob 交给 npunlock
执行；每一行的三种运行时使用完全相同的编译 blob、输入、权重与输出形状。
OpenVINO 显式使用 NPU，`EXECUTION_DEVICES=NPU`。没有 CPU/GPU 算子回退。

输入已写入各自常驻缓冲区，编译、分配、输入写入及数值验证均在计时之外。
OpenVINO 使用 `request.get_input_tensor().data` 填充输入，
`infer(share_outputs=True)` 避免 Python API 默认的输出复制。
各实现 5 次预热，31 轮交替并旋转顺序；每次同步完成才计时结束。
表中数字是本轮中位延迟，单位 ms。

| M/K/N | 输入路线 | 编译器 | native standard | native replay | OpenVINO |
|---|---|---|---:|---:|---:|
| 1/1024/1024 | FP16-FQ | DRIVER | 0.2596 | 0.1631 | 0.1896 |
| 1/1024/1024 | FP16-FQ | PLUGIN | 0.3649 | 0.2030 | 0.2410 |
| 1/1024/1024 | prequant-U8 | DRIVER | 0.3382 | 0.1900 | 0.2116 |
| 1/1024/1024 | prequant-U8 | PLUGIN | 0.2885 | 0.1749 | 0.2160 |
| 128/1024/1024 | FP16-FQ | DRIVER | 0.3172 | 0.2103 | 0.2319 |
| 128/1024/1024 | FP16-FQ | PLUGIN | 0.3346 | 0.2039 | 0.2302 |
| 128/1024/1024 | prequant-U8 | DRIVER | 0.3859 | 0.2281 | 0.2643 |
| 128/1024/1024 | prequant-U8 | PLUGIN | 0.3438 | 0.2157 | 0.2569 |
| 1024/1024/1024 | FP16-FQ | DRIVER | 0.7116 | 0.5193 | 0.5570 |
| 1024/1024/1024 | FP16-FQ | PLUGIN | 0.6163 | 0.4200 | 0.4539 |
| 1024/1024/1024 | prequant-U8 | DRIVER | 0.5706 | 0.4045 | 0.4470 |
| 1024/1024/1024 | prequant-U8 | PLUGIN | 0.6113 | 0.4013 | 0.4445 |
| 2048/512/256 | FP16-FQ | DRIVER | 0.5502 | 0.3349 | 0.4247 |
| 2048/512/256 | FP16-FQ | PLUGIN | 0.4175 | 0.2915 | 0.3380 |
| 2048/512/256 | prequant-U8 | DRIVER | 0.3366 | 0.2296 | 0.2649 |
| 2048/512/256 | prequant-U8 | PLUGIN | 0.3781 | 0.2498 | 0.2947 |

1024 方阵、prequant-U8、DRIVER 对照的有效吞吐为：
standard **3.76 TOPS**，OpenVINO **4.80 TOPS**，固定地址 replay **5.31 TOPS**。
计算公式为 `2*M*K*N / (median_ms*1e9)`，乘法和加法各计一次。
不计算补零、复制、量化或并行副本的额外工作量。

OpenVINO 版本为 2026.4.1；显式 DRIVER 使用驱动编译器 **8.3**，
显式 PLUGIN 使用随包编译器 **8.4**。本机默认偏好 PLUGIN。
同一行内部的运行时比较有效；DRIVER/PLUGIN 的表格块不是同轮交替编译器测试，
不能将块间差异全部归因于编译器版本。
PLUGIN 的部分 FP16-FQ 图有更少的 DMA 描述符，是进一步研究编译调度的线索；
描述符数量不等于 DMA 耗时。

原始数据为 [openvino-paired.json](openvino-paired.json)。
40 个 OpenVINO 导出图均通过现有有界 ELF 解析器检查，非 elementwise DPU
精度字段为 U8×U8，未包含 SHAVE invocation；见 [openvino-graphs.json](openvino-graphs.json)。
所有路线的输入变化、归零、恢复和最终输出检查逐元素一致。

## 主机开销与固定地址 replay

诊断只编译了独立的 `session_diagnostic.dll`：对生产 `session_win32.c` 的副本插桩。
正式 DLL、PYD、Torch 分流与模型源码的指纹未变化。
QPC 插桩得到的 fence wait 包含设备、驱动和操作系统等待，不能当作纯 NPU 运算时间。

第一轮 1024 方阵、不带 device profiling 的主机分段中位数：

| 阶段 | FP16-FQ ms | prequant-U8 ms |
|---|---:|---:|
| set arguments | 0.0027 | 0.0029 |
| create command list | 0.0297 | 0.0285 |
| create fence | 0.0007 | 0.0006 |
| append graph | 0.0039 | 0.0036 |
| close list | 0.0003 | 0.0003 |
| queue submit | 0.0211 | 0.0218 |
| fence wait | 0.4394 | 0.3865 |
| destroy list/fence | 0.0918 | 0.0691 |
| execute total | 0.5910 | 0.5342 |

阶段中位数不要求相加等于总时间中位数。
生产执行路径每次创建、关闭、销毁 list/fence，并重复设置参数。
实验预先录制一个固定地址的 command list，之后 reset fence、submit、wait。
每次仍同步等待，没有依靠并行推理来提高数字。

在原 npunlock 编译图上的独立 replay 测试：

| M/K/N | 路线 | standard ms | replay1 ms | replay16 每次 ms |
|---|---|---:|---:|---:|
| 1/1024/1024 | FP16-FQ | 0.2950 | 0.1695 | 0.15206 |
| 1/1024/1024 | prequant-U8 | 0.3061 | 0.1574 | 0.14973 |
| 128/1024/1024 | FP16-FQ | 0.3721 | 0.2032 | 0.18871 |
| 128/1024/1024 | prequant-U8 | 0.3706 | 0.1919 | 0.18674 |
| 1024/1024/1024 | FP16-FQ | 0.5495 | 0.3314 | 0.30766 |
| 1024/1024/1024 | prequant-U8 | 0.5147 | 0.2937 | 0.27869 |
| 2048/512/256 | FP16-FQ | 0.4479 | 0.2550 | 0.23373 |
| 2048/512/256 | prequant-U8 | 0.3971 | 0.2251 | 0.21394 |

此轮 1024 prequant-U8 replay1 为 **7.31 TOPS**，replay16 为 **7.71 TOPS**。
replay16 是在一个 C 调用里串行执行 16 次，每次都有 fence wait；没有并行运行。
这些数字与上面的 OpenVINO 同轮数据来自不同测量时段，不能混在一起比较。
桌面负载、频率和电源状态未锁定；同轮交替结果比跨轮峰值更有参考价值。

replay 共 5216 次成功执行。输入改变后得到新结果，恢复后逐元素一致；
输入释放后再次执行被拒绝，所有 buffer/session 正常释放。
实验持有绑定的描述符和 session，并沿用提交/等待失败后的上下文隔离。
固定地址和调用方保留 buffer 的约束尚未整合成生产 Torch 的生命周期管理。
见 [replay.json](replay.json)。

## 输入量化与实际 DPU 任务

在相同量化网格上，用 U8 输入替代图内 FP16 FakeQuantize：

| M/K/N | FP16-FQ ms | prequant-U8 ms | prequant-U8 TOPS |
|---|---:|---:|---:|
| 1/1024/1024 | 0.2634 | 0.2640 | 0.00794 |
| 128/1024/1024 | 0.2973 | 0.2807 | 0.9563 |
| 512/1024/1024 | 0.4296 | 0.3862 | 2.7803 |
| 1024/1024/1024 | 0.6218 | 0.5607 | 3.8300 |
| 256/2048/512 | 0.3936 | 0.3809 | 1.4095 |
| 512/2048/512 | 0.4763 | 0.4227 | 2.5402 |
| 2048/512/512 | 0.5396 | 0.4776 | 2.2482 |
| 2048/512/256 | 0.4374 | 0.3566 | 1.5055 |

这些是第一轮诊断 session API 的时间，不是常规 Torch 张量 API 的时间。
输入编码生成在计时外。真实模型仍要安排上游产生量化激活或支付量化成本，
不能把提前准备编码的吞吐直接称为完整模型加速。

另行开启 `PERF_COUNT=YES` 获取 Intel task-level profiling：

| M/K/N | 路线 | DPU 记录数 | 任务时间线跨度 µs |
|---|---|---:|---:|
| 128/1024/1024 | FP16-FQ | 8 | 47.428 |
| 128/1024/1024 | prequant-U8 | 4 | 19.587 |
| 1024/1024/1024 | FP16-FQ | 36 | 216.816 |
| 1024/1024/1024 | prequant-U8 | 32 | 170.321 |

FP16-FQ 比 prequant-U8 多 4 个并行的输入量化 DPU task；MatMul 已经分配给
cluster 0..3。128 的 MatMul 每 cluster 约 19～20 µs；1024 的 MatMul
分为 8 个输出 tile、每 tile 4 个 cluster，每个记录约 21 µs。
所以计算已经有并行执行，尚有图切分和执行管线优化空间。

1024 prequant-U8 的 DPU 任务时间线按有效工作量折算约 **12.61 TOPS**，
它不是含 DMA、提交和同步的整体吞吐，也不是 MAC 阵列峰值。
各 cluster 的 duration 总和包含并行重叠，不能用总和推算端到端延迟。
这次驱动查询没有返回 DMA/SW task 记录；不能据此判断 DMA 没有成本。
active/stall cycles 都未提供可用数值，无法计算阵列利用率。
profiling 会改变图和调度，只用于诊断；性能表使用关闭 profiling 的图。
见 [128 profiling](profile-128-1024-1024.json) 和 [1024 profiling](profile-1024-1024-1024.json)。

## 4 / 6 tile 对照

OpenVINO 查询 `NPU_MAX_TILES=6`，默认 LATENCY 图本次分配 4 个 tile。
使用相同 IR，分别以 `NPU_TILES=4/6` 编译，4 个配置在 31 轮内交替执行：

| M/K/N | 编译器 | 4 tile ms | 6 tile ms | 4 tile TOPS | 6 tile TOPS |
|---|---|---:|---:|---:|---:|
| 128/1024/1024 | DRIVER | 0.2496 | 0.2542 | 1.075 | 1.056 |
| 128/1024/1024 | PLUGIN | 0.2530 | 0.2471 | 1.061 | 1.086 |
| 1024/1024/1024 | DRIVER | 0.4545 | 0.4093 | 4.725 | 5.247 |
| 1024/1024/1024 | PLUGIN | 0.4478 | 0.4064 | 4.796 | 5.284 |

全部使用 prequant-U8，结果逐元素一致。ELF 中实际出现 tile 0..5 的 DPU sections。
1024 方阵约减少 9%～10% 延迟；128 小矩乘没有一致收益。
这个测试使用 OpenVINO 运行时，没有与 native replay 叠加测量。
因此不能宣称“6 tile + replay”已经实现某个更高吞吐，也不应将 6 tile 固定为所有模型默认。
见 [openvino-tiles.json](openvino-tiles.json)。

Intel 官方列出的本机 NPU 峰值是 **47 INT8 TOPS**，见
[258V 规格](https://www.intel.cn/content/www/cn/zh/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html)。
单个同步矩乘、实际切分/DMA/频率和完整峰值条件不同；现有 profiling 不足以把
47 TOPS 与实测差距全部分配到 I/O、计算或主机同步。

## 可以借鉴的实现及下一步

[OpenVINO NPU 插件](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu)
和 [NPU compiler](https://github.com/openvinotoolkit/npu_compiler) 均开源，使用 Apache-2.0。
插件与编译器是不同层：插件管理 buffers、请求、command lists 与同步；
编译器安排 DPU/SHAVE 工作、切分与内部内存传输。
参见 [插件说明](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md)
及 [compiler 说明和许可证](https://github.com/openvinotoolkit/npu_compiler/blob/develop/README.md)。

核对了 [zero_pipeline.cpp](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/src/backend/src/zero_pipeline.cpp)：
Pipeline 构造阶段创建 list/fence 并录制图，push 提交已录制的 list，
pull 等待，reset 重置同步对象；参数变化有更新 command list 的路径。
本地 replay 按已有 npunlock runtime 实现了同类复用实验，没有复制 OpenVINO 源码。
[InferRequest 设计](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/docs/inference-request.md)
也解释了 Tensor 生命周期、上下文归属与缓冲区共享的处理。

下一步优先将 list/fence 对象复用接入正式 session runtime，并完成绑定变化时的
更新/重录、Tensor/Storage 生命周期、输出独立性、异常恢复与有界缓存检查。
先覆盖固定图的真实模型推理，再测常规 API 与模型总耗时。
随后探索量化激活跨层传递和按矩阵形状选择 tile 数。
这些实现可以继续通过 Level Zero 直接执行，不要求增加 OpenVINO 运行时依赖。

本轮也验证了 OpenVINO 导出的两种编译器 ELF 均能由我们的 native runtime 执行。
这为可选的离线编译路线提供了证据；本轮推理模式无状态，尚不保证任意带状态模型可直接互换。

## 复现、证据和范围

独立测试环境为 `.conda/openvino-bench`，其中安装 OpenVINO 2026.4.1 / NumPy 2.5.3。
原 `.conda/npunlock` 环境没有安装 OpenVINO。
本目录主记录共 **8716 次成功执行**：8 形状诊断 592 次，2 profiling 形状 36 次，
replay 5216 次，OpenVINO baseline 624 次，同 blob 交替对照 1936 次，tile 对照 312 次。
输出均与量化网格的参考结果逐元素一致，参考计算还以 INT64 点积抽查。
未测随机非量化网格、任意模型、低位训练或异步并发。
误用 Python API 默认输出复制的初始对照保存在 `openvino-output-copy/`，不计入以上主记录。
早期诊断脚本字段/初始化失败保存在 `failures/`；已修正，并确认资源释放。

在项目根目录的 PowerShell：

```powershell
# 独立诊断 DLL；不覆盖生产 DLL
.\run_torch_npu.ps1 -Script npunlock-probe/research/build_dpu_diagnostic.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/dpu_diagnostic.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/dpu_diagnostic.py -ScriptArguments '--profile','--shape','1024','1024','1024'
.\run_torch_npu.ps1 -Script npunlock-probe/research/dpu_replay_benchmark.py

# OpenVINO baseline 与 tile 对照
.\.conda\openvino-bench\Scripts\python.exe npunlock-probe/research/openvino_dpu_benchmark.py
.\.conda\openvino-bench\Scripts\python.exe npunlock-probe/research/openvino_dpu_benchmark.py --tiles-test

# 同 blob 两套运行时交替对照，仅本 PowerShell 进程设置路径
$env:PYTHONPATH="$PWD;$PWD\npunlock\python"
$env:NPUNLOCK_NATIVE_DIR="$PWD\.conda\npunlock\Lib\site-packages\npunlock\_bin"
.\.conda\openvino-bench\Scripts\python.exe npunlock-probe/research/openvino_runtime_comparison.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/inspect_openvino_graphs.py
.\run_torch_npu.ps1 -Script npunlock-probe/research/audit_dpu_diagnostic.py
```

审计结果见 [audit.json](audit.json)：原始中位数/单位/吞吐、图精度与 hash、
40 个导出 ELF、源文件/历史快照、资源关闭及正式 runtime 指纹均通过检查。
历史源码与诊断 DLL 位于 `initial-runtime/`，脚本快照保留对应原始测量指纹。
重跑产生的新样本及源码指纹可能不同；审计脚本核对的是本轮公开记录及保留的历史快照。
