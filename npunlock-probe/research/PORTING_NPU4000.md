# npunlock 的 Lunar Lake / NPU4000 移植研究

日期：2026-10-03。平台：Core Ultra 7 258V，NPU device `0x643e`，驱动 `32.0.100.5540`，图编译器 `8.3`。

**结论：值得移植，最小自定义 SHAVE 内核已在本机通过硬件验证。** 本次没有完成通用 npunlock 后端；得到的是固定 1×16 FP16 carrier 的研究原型，以及正式移植需要处理的具体差异。

## 已验证的链路

```text
C 加一内核 → MoviTools 4000xx 编译 → 汇编 → 链接 ELF32
             ↓
解析本机 Abs 图的 ELF64 / NPU4000 任务与参数
             ↓
将新机器码放入已有代码窗口 → Level Zero 加载 / 执行 → CPU 对照
```

| 项目 | 结果与证据 |
|---|---|
| 目标工具链 | 本地已有的 MoviTools 接受 `4000xx`，完成编译、汇编、链接；错误目标名会被拒绝 |
| 自定义运算 | `out[i] = (__fp16)((float)in[i] + 1.0f)`，按参数中的维度计算元素数 |
| 新内核 | ELF32 780 字节；`.text` 144 字节，入口 `0x1d000000`，没有非空 `.arg.data` 或未解决重定位 |
| 本机 carrier | 1×16 FP16 Abs；53,920 字节 ELF64；8 个 range、8 个 invocation，tile 索引 0–3 |
| 原内核来源确认 | 图中 5,040 字节代码与 Intel 公开的 `activation_abs.4000xx.elf` 的 `.text` 逐字节一致 |
| 最小替换 | 新代码写入原窗口起始位置；实际改变 143 字节；文件长度、section table、range、invocation 和重定位均不变 |
| 驱动接受 | 修改后的图在本机加载、执行成功；运行时返回 vendor `0x8086`、device `0x643e`、driver `1005540` |
| 正确性 | 整数序列与两组固定种子的随机 FP16 输入，共 48 个输出逐元素精确一致，最大误差均为 0；额外校验原图仍为 Abs |

这验证了**匹配目标的机器码、当前样本的参数 ABI、当前驱动接受补丁图、实际输出正确**。不再需要把这些条件全部当作未知。

原图代码 SHA-256：`5b51680501c037dfeb9fe4653df9d453fd5b42224681b41338b6de591d225a9d`。

原图 SHA-256：`939110db2d9e656fb692793b7e2cd3b57b9502399ab794fd0feb575c1074cb4c`。

补丁图 SHA-256：`3f3ad7ff7597e77588933ba300f23b0925fbef123d78ec13e09ee8e7412a76d3`。

完整硬件记录：[custom4000-hardware.json](E:/Pytorch_xfire_NGPU/npunlock-probe/research/results/custom4000-hardware.json)。替换范围与哈希：[inplace-patch.json](E:/Pytorch_xfire_NGPU/npunlock-probe/research/results/inplace-patch.json)。

## 与现有实现的差异

### 1. CMX workspace 不是文件内容

上次报错的 type `0x8aaaaaad` 有正式名称：`VPU_SHT_CMX_WORKSPACE`。Intel reader 对它、`VPU_SHT_CMX_METADATA=0x8aaaaaac` 和标准 `SHT_NOBITS` 都不读取文件数据。所以 size 大于图文件长度是允许的内存描述，不能按文件 payload 判断越界。

修复需要区分“有文件数据”和“只有内存描述”的节。其他实际文件数据、section table、字符串表、符号和重定位仍必须严格校验边界。扩展图文件时，也不能把 CMX workspace 的逻辑 offset 当作文件 offset 去平移。

来源：[Intel ELF 扩展类型](https://github.com/openvinotoolkit/npu_compiler_elf/blob/d325f45f2cb405b5fa2ff17a30de9469f1641b73/core/include/vpux_elf/types/vpu_extensions.hpp)、[Intel reader 的处理](https://github.com/openvinotoolkit/npu_compiler_elf/blob/d325f45f2cb405b5fa2ff17a30de9469f1641b73/core/include/vpux_elf/reader.hpp)。

### 2. 任务布局和 section 分组不同

| 对象 | npunlock 当前 NPU3720 假设 | 本机 NPU4000 样本 / 公开定义 |
|---|---|---|
| kernel text | `.text.KernelText` | `shave.text` |
| params | `.text.KernelParams` | `shave.params` |
| range | 单一 `.text.ActKernelRanges`；24 字节/条 | `task.shave.range.<tile>.<engine>`；40 字节/条 |
| invocation | 单一 `.text.ActKernelInvocations`；64 字节/条 | `task.shave.invocation.<tile>.<engine>`；96 字节/条 |
| range 字段 | 旧偏移 | entry 在 `+8`；text window base 在 `+16`；code size 在 `+24` |
| invocation 字段 | 旧偏移 | range 在 `+0`；kernel args 在 `+8`；data window 在 `+16`；tile 在 `+84`；range index 在 `+88` |

本次解析出了四组 tile section，每组两个 range 和两个 invocation。40/96 字节同时由公开头文件和实际文件大小/字段内容验证。tile 索引是图里的任务分组，不能仅凭它推断物理计算单元数量。

需要按 section 的 tile/engine、条目索引和引用关系建立映射；不能只改名字继续用旧尺寸。非计算 workload、barrier、下一任务链接、runtime、stack 和数据窗口应保留。

来源：[Intel NPU4000 SHAVE runtime ABI](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nnrt_shavert.h)。

### 3. 重定位与固定地址都要支持

Intel 公开 ELF 类型定义：`R_VPU_64=0`、`R_VPU_32=4`、清除 tile 地址位的 32/64-bit OR 类型为 17/18。npunlock 的旧图路径对指针重定位采用自己的 NPU3720 观察值，不能整体套到新图上。

本机实际图中：

- range 的 `text_window_base` 有 type 0 的重定位，指向 `shave.text`。
- invocation 的 `kernel_args` 有 type 0 的重定位，指向 `shave.params` 的不同偏移。
- 参数的 dims/strides 地址有 type 4 的重定位。
- invocation→range 指针和 MemRefData 的 CMX dataAddr 已有固定值，没有对应重定位。例如 tile 0 的 input/output 为 `0x40218000` / `0x40218040`，tile 1 对应 `0x40418000` / `0x40418040`。

通用目标发现必须同时理解这些形式，验证固定 CMX 地址的 tile 与内存范围。不能因为缺少某条重定位就认定参数损坏，也不能把实际符号关系全部替换为硬编码偏移。

来源：[range 重定位生成](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/NPU40XX/dialect/NPUReg40XX/ops/act_kernel_range.cpp)、[invocation 重定位生成](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/NPU40XX/dialect/NPUReg40XX/ops/act_kernel_invo.cpp)。公开生成代码也可能被后续编译 pass 折叠，实际图必须再核对。

### 4. MemRefData 有可复用部分，但不能泛化为所有内存 ABI

公开 `MemRefData` 仍为 40 字节：dataAddr 为 64-bit；isStatic `+8`；rank `+12`；dimsAddr `+16`；stridesAddr `+20`；dtype `+24`；dimsOrder `+28`；location `+36`。

本机样本的输入/输出为 static、FP16、CMX、rank 4、dims `[8,1,1,1]`。本次内核读取 dataAddr 的低 32 位以及 dimsAddr，并成功执行。这给“静态 CMX 张量”的 generation-specific helper 提供了硬件依据。DDR、动态 shape、非连续 strides、FP32、DMA kernel 和其他参数格式仍未验证。

原图逻辑输入是 16 元素，8 次 invocation 每次描述 8 元素。不能把它们简单相加并要求等于逻辑张量的 16；需要继续追溯 DMA、padding、tile 和张量片段的映射。现有 `_automatic_target_groups` 的 exact-cover 假设需要检查，而不是机械地选择第一个 group。

来源：[Intel 参数结构](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/sw_runtime_kernels/kernels/inc/common_types.h)、[参数序列化与重定位](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/dialect/VPUASM/ops/kernel_params.cpp)。

### 5. 工具链已经够做简单内核

本地工具版本：moviCompile `00.114.11.4207`，moviAsm `1.13.16 64-bit`，moviLLD `3.0.9`。它们对 `4000xx` 的编译、汇编和链接都成功；不存在“当前包只能生成 3720xx”的硬障碍。

现有项目的限制分别位于：

- `src/shavecc/shavecc.c`：只允许 `3720xx`；assembler `--cv` 写死为 `3720xx`；只展开 NPU3720 header。
- `python/npunlock/_native.py`：编译 CPU 参数写死为 `3720xx`。
- `src/common/elf32.c`：入口和 data window 校验的报错文字以 3720xx 为前提。地址本身在本次 4000xx 自定义 ELF 和公开 ELF 上相同，有机会保留验证逻辑。
- `src/patchblob/graph_blob.c`：旧 section、任务尺寸、参数和重定位假设。
- `python/npunlock/__init__.py`：自动目标分组及 compile→patch 的连接需要适配。

现有 C `shavecc_options` 已有 `target_cpu` 字段，首先开放经过验证的 `4000xx`，不需要重造一套编译 API。初期增加专用 `npu4000_kernel.h`，保留原 NPU3720 helper。

本次没有链接 math library。Intel 的构建配置为 37xxxx/40xxxx 分别选择库；tanh/exp 等非线性内核的数学库应单独验证。不能因为简单加法通过就承诺现有 `mlibm.a` 在所有目标上正确。

公开 compiler 的 ShaveCodeGen 可以把 MLIR/LLVM IR 转为 SHAVE，但其最终代码生成仍调用 MoviTools。所用公开 LLVM target 列表未见 SHAVE；直接构建 LLVM 不会自动获得完整 SHAVE C 编译器。本次直接复用已有 DLL 路线最短。

来源：[分架构库配置](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/sw_runtime_kernels/kernels/CMakeLists.txt)、[LLVM IR 到二进制实际实现](https://github.com/openvinotoolkit/npu_compiler/blob/0b38f7d42113ff329ac2bdd33583d123de4ccf2f/src/vpux_compiler/src/utils/llvm_to_binary.cpp)。

## 建议的正式实现顺序

| 阶段 | 最小交付 | 验收条件 | 难度判断 |
|---|---|---|---|
| 1. NPU4000 inspection | 识别 CMX 节、40/96 字节任务、参数、引用、tile/engine；支持 CLI 输出 | 离线 fixture 能解析；截断、越界和不支持格式明确拒绝 | 中低；关键定义已确认 |
| 2. 官方编译入口 | C/Python 开放 `4000xx`，assembler 跟随 target，加入 NPU4000 helper | 编译/汇编/链接检查通过；3720xx 原测试不退化 | 中低；简单内核已验证 |
| 3. 一元内核替换 | 从图中自动发现所有对应 range；先只允许新代码放进现有窗口 | copy/add1/mul2 在多个 shape、正负输入上通过；补丁范围可审计 | 中；本次已有一个成功样本 |
| 4. 更大代码与多算子 | 正确扩展 `shave.text`，更新文件 offset、符号/引用和代码窗口长度；多个 custom node 的映射 | 新代码大于 carrier、多个共享/独立 kernel、不同 tile/engine 都通过 | 中高；需完整引用关系和保留性测试 |
| 5. 扩展数据与执行 | 双输入、scalar broadcast、FP32、stride、math library、可靠的会话复用 | 按明确兼容矩阵通过数值、重复执行和版本测试 | 高；当前证据不能覆盖 |

第一版建议明确支持 **Windows、本机 NPU4000、静态、dense、FP16、一元、CMX、代码可放入已有窗口、copied 执行 API**。先把这个范围做可靠，再扩展。

原型利用的是 carrier 的空余代码窗口：保留原 5,040 字节窗口，只替换前 144 字节。它没有证明任意大小代码的 ELF 追加重排可用。当前只包含一个自定义节点，也没有验证不同逻辑节点共享 kernel text 时能否独立替换。

新路径应通过 device/provenance 与实际 ELF 特征共同判断，遇到未知版本明确报告 unsupported。可以内部将多 section 的任务索引展平，尽量保留现有 C API；先不引入插件框架或通用后端工厂。NPU3720 旧路径保留，由两代 fixture 验证。

首批 shape 应覆盖 1、16、17、31、32、33、2048 元素以及二维张量，测试对齐、尾部处理、padding 和 tile 拆分。二元/广播和 FP32 放在一元支持之后。重复执行需分 copied 与 shared 两条路径；此前 shared 会话的 device-lost 问题仍未解决。

## 可以怎样贡献上游

建议拆成独立可审阅的改动，而不是一次提交整套 PyTorch 后端：

1. 修复 `_InferOptions` 多余初始化参数。现有一行差异已经保留。
2. 增加 NPU4000 离线 inspection、生成 fixture 的命令、边界检查和元数据报告。
3. 增加 `4000xx` 工具目标与一元替换，附上本机数值校验和明确的支持边界。
4. 后续再提交图增长、多节点、数据类型和会话稳定性改动。

提交内容可包括源码、可再生成的小图、测试输入/输出、哈希、版本和公开来源清单。提取的厂商 DLL/库保持在本机；本次未创建远程 issue、PR 或发送消息。

“完全支持本机 NPU”应落实为兼容矩阵：哪些 shape、dtype、内存布局、kernel 类型、驱动/compiler 版本可以运行。SHAVE 支持本身不会自动补齐 DPU 自定义编程、PyTorch dispatcher、autograd、动态 shape 或 GPU/NPU 协调。

## 复现与研究产物

已保存 37 份固定提交的公开源码/内核文件，来源、字节数和 SHA-256 见 [manifest.json](E:/Pytorch_xfire_NGPU/npunlock-probe/research/sources/manifest.json)。下载的 Git LFS 内核已核对 LFS oid 和大小。公开 compiler 提交为 `0b38f7d42113ff329ac2bdd33583d123de4ccf2f`，ELF 子模块为 `d325f45f2cb405b5fa2ff17a30de9469f1641b73`。

无需安装 MSVC 或重建 C 库。运行现有脚本：

```powershell
$python = 'E:\Pytorch_xfire_NGPU\.conda\npunlock\python.exe'

# 离线解析图，包含截断与文件数据越界检查
& $python 'E:\Pytorch_xfire_NGPU\npunlock-probe\research\inspect_npu4000.py'

# 隔离 worker 编译、汇编、链接；不执行 NPU
& $python 'E:\Pytorch_xfire_NGPU\npunlock-probe\research\probe_movi_targets.py'

# 只生成并审计补丁图
& $python 'E:\Pytorch_xfire_NGPU\npunlock-probe\research\prove_custom4000.py' --build-only

# 原图 Abs 校验 + 补丁图加一校验，外部期限 45 秒
& $python 'E:\Pytorch_xfire_NGPU\npunlock-probe\research\prove_custom4000.py'
```

核心原型：[inspect_npu4000.py](E:/Pytorch_xfire_NGPU/npunlock-probe/research/inspect_npu4000.py)、[工具链探测](E:/Pytorch_xfire_NGPU/npunlock-probe/research/probe_movi_targets.py)、[补丁与硬件验证](E:/Pytorch_xfire_NGPU/npunlock-probe/research/prove_custom4000.py)。解码结果：[npu4000-inspection.json](E:/Pytorch_xfire_NGPU/npunlock-probe/research/results/npu4000-inspection.json)。

三个硬件输入都用 copied API，未测性能、功耗或长时间稳定性。本次结果支持着手正式移植，也清楚限定了下一步需要验证的内容。
