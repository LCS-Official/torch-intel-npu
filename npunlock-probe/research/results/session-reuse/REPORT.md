# 正式运行时 command list/fence 复用

日期：2026-10-06。平台：Core Ultra 7 258V / Lunar Lake NPU4000（8086:643e），
Windows 驱动 32.0.100.5540，驱动编译器 8.3，Torch 2.14.1+cpu。

正式 `npunlock_session.dll` 已启用每张图复用一份 command list 和 fence。
普通 `intel_npu` Tensor 算子和现有模型适配器自动使用，无需修改用户调用代码。
本轮只改变提交路径，未修改 FP16/FP32 算术、DPU 矩阵内核或 SmolLM 适配器。
生产 `.conda/npunlock` 环境仍未安装 OpenVINO；继续依赖 Intel 驱动和图编译器。

## 实现与生命周期

每次调用验证并绑定当前输入/输出地址，重新记录图命令，提交并等待执行完成。
成功等待后，在返回给调用者之前清空 command list，并将 fence 重置为未触发状态。
因此，调用者可以释放输出 Tensor；下一次调用可以绑定新的输出地址。
执行命令不会留在闲置列表中，也没有基于裸指针的 replay 缓存或额外 Tensor 引用。

命令列表和 fence 在图关闭时先于图对象销毁。上下文仍使用原来的同步串行队列。
提交失败或等待超时、device-lost 时继续隔离上下文，不重置或释放可能仍在使用的资源。
可确认未提交或已完成的 append/reset 错误则销毁这对对象；原生会话可在下次调用重建。
Python 层既有驱动错误处理策略保留。

设计遵循 [Level Zero command list/fence API](https://github.com/oneapi-src/level-zero/blob/master/include/ze_api.h)
对执行期间不可重置的约束。采用对象复用和逐次记录，未接入上阶段固定地址 replay。
同时去掉暖调用中重复的 pre-init 参数绑定；首次初始化仍保留所需绑定。
公共 `graphinfer.h` ABI 未变。

每张缓存图现在会保留一对驱动对象，直到图关闭。Torch 图缓存上限仍为 512；
`live_buffer_bytes` 只计共享载荷分配，不包含驱动内部 command list/fence 内存。

## 相同编译图的矩阵对照

使用上阶段保存的同一份 ELF blob、相同固定输入/参考输出。新旧 DLL 分别创建上下文，
使用同样的 Python `Session.infer` 包装和驻留缓冲区；编译、分配、上传不计入时间。
每组预热 5 次，交替执行 31 轮，表中是各路径中位数。
延迟降低按 `1 - after/before` 计算。

两种入口都执行 **W8A8/INT8 矩乘**：FP16-FQ 在图中将 FP16 输入量化，
prequant-U8 直接接收预量化输入；FP16 输出类型不改变矩乘的计算精度。
有效 TOPS 按 `2*M*K*N / 同步调用耗时` 计算，包含本次提交、等待及图内 DMA 等开销。

| M / K / N | 入口 | 旧版 ms | 新版 ms | 延迟降低 |
|---|---|---:|---:|---:|
| 1 / 1024 / 1024 | FP16-FQ | 0.3343 | 0.2301 | 31.17% |
| 1 / 1024 / 1024 | prequant-U8 | 0.3465 | 0.2309 | 33.36% |
| 128 / 1024 / 1024 | FP16-FQ | 0.3662 | 0.2558 | 30.15% |
| 128 / 1024 / 1024 | prequant-U8 | 0.3746 | 0.2595 | 30.73% |
| 1024 / 1024 / 1024 | FP16-FQ | 0.6278 | 0.4832 | 23.03% |
| 1024 / 1024 / 1024 | prequant-U8 | 0.6248 | 0.4348 | 30.41% |
| 2048 / 512 / 256 | FP16-FQ | 0.4918 | 0.3549 | 27.84% |
| 2048 / 512 / 256 | prequant-U8 | 0.4330 | 0.3014 | 30.39% |

1024 方阵 prequant-U8 的有效吞吐从约 3.44 升到 **4.94 TOPS**。
这不是芯片峰值，也不能据此声称 FP16 达到 4.94 TFLOPS。
上阶段固定地址 replay 的结果属于另一种路径、测量批次和包装开销，不能拼接比较。

624 次调用成功。原始输入、零输入、恢复输入以及最终输出均核对；新旧结果与保存参考
按 FP16 原始位模式一致。所有共享缓冲区释放、所有会话关闭。
原始时间样本和 blob 指纹见 [native.json](native.json)，测量时脚本见 [native-script.py](native-script.py)。

## 实际模型路径对照

本地 SmolLM2-135M-Instruct 的真实 tokenizer/embedding 输入，连续 **4 个 decoder 层**。
测试 FP16 和 W8A16 模式，分别测 prefill 与从固定 seed KV 开始的一次单 token decode。
输出及 KV 每次重新分配；输入、权重和图已预热。计时区间无显式载荷上传/下载或新图会话。
每个进程、每组操作预热 5 次、采样 31 次。四个隔离进程按 **旧、新、新、旧** 顺序执行，
表中取同一版本两个进程各自中位数的算术平均。

| Batch / prefix | 模式 | 操作 | 旧版 ms | 新版 ms | 延迟降低 |
|---|---|---|---:|---:|---:|
| 1 / 32 | FP16 | prefill | 6.0934 | 5.6799 | 6.79% |
| 1 / 32 | FP16 | decode | 4.5640 | 4.3863 | 3.89% |
| 1 / 32 | W8A16 | prefill | 5.3776 | 5.2169 | 2.99% |
| 1 / 32 | W8A16 | decode | 4.1562 | 4.0937 | 1.50% |
| 4 / 16 | FP16 | prefill | 9.1891 | 7.7536 | 15.62% |
| 4 / 16 | FP16 | decode | 5.5478 | 4.3253 | 22.04% |
| 4 / 16 | W8A16 | prefill | 9.0098 | 8.3375 | 7.46% |
| 4 / 16 | W8A16 | decode | 5.6866 | 4.5988 | 19.13% |

Batch 4 的收益较明显。Batch 1 的两次旧版进程和两次新版进程之间有较大波动，
个别新版进程比某次旧版进程慢，不能将上表小幅收益当作稳定保证。
这是桌面环境、未锁定频率和后台负载下的有限对照；没有测完整 LLM 生成速度。

每个进程 1248 次成功 NPU 图调用，共 4992 次。每组计时/预热增加 288 次图调用，
上传、下载和图会话创建计数均未增加。每个进程保存 104 项输出/KV/位置数组，
其余三个进程与首个进程的全部数组按原始字节比较一致（312 项跨进程比较）。
预先保留的输出和 seed KV 在后续暖调用后也逐位未变。
各进程关闭后载荷缓冲区和字节数归零，上下文健康。

旧版进程通过独立绑定清单使用保存的旧 DLL。当前 C++ `backend.cpp` 源码和公开 ABI
未变；本轮重新构建的同一 PYD 用于所有对照进程。正常 DLL/PYD 构建指纹检查继续启用，
并未绕过校验或在仍有 Tensor 存活的进程内切换 DLL。
原始逐进程样本、汇总及测量源分别见 `model-*.json`、[model-control.json](model-control.json)
和 [model-script.py](model-script.py)。

## 回归与失败处理验证

| 检查 | 结果 |
|---|---|
| 原生 mock | 100 次交替绑定只创建一对对象；完成后重置、关闭恰好释放一次；append/reset 可恢复；提交/超时/device-lost 隔离通过 |
| 低层会话 Quick | 23 项通过：FP16/FP32、共享上下文、错误尺寸/别名/上下文拒绝、串行多线程、重复关闭 |
| Torch 设备 Quick | 33 项、4160 次 NPU 调用通过：Storage/view 生命周期、特殊位模式、cast/reduce/matrix/linear、线程和 shutdown |
| Torch 训练 Quick | 11 项、2732 次 NPU 调用通过：梯度、SGD、FP16/FP32 MLP/卷积/轻量 ResNet |
| 离线证据审核 | 构建/源文件/DLL/PYD 指纹、时间中位数与 TOPS、模型字节结果和回归记录一致 |

回归原始 JSON/log 保存在 `regressions/`，审核见 [audit.json](audit.json)。
设备和训练回归 shutdown 后 `live_buffers=0`、`context_failed=false`。
这些检查覆盖既有支持范围，不构成新增任意模型、完整训练算子或完整 LLM 支持的声明。

最初模型基准中的循环局部变量保留了一个 KV 引用，被载荷生命周期断言发现。
修正基准引用释放后，完整四进程对照重新运行并通过；失败记录保存在
`failures/model-fixture-lifetime`，未纳入上表和成功调用计数。

## 构建与复现

旧 session DLL SHA256：`5d145c0f1dd455994d47b4ca0433afb4812347075e673bf3fa653ca75b5d1e44`。

新 session DLL SHA256：`d959fc5b1fb9a3171c192d1f607b11431af634203805d46cdcd6bba243933a51`。

新 PYD SHA256：`de5b037c50a1dcd253a37e9027b7a780513b0a6483644aef19aa7c856cd73411`。

旧版源码、DLL 和清单保存在 `before/`；新版源码/二进制/清单保存在 `after/`。
本轮生产源码差异见 [production.patch](production.patch)。
原生 GCC `-Wall -Wextra -Werror -Wpedantic` 构建和 C++ 设备扩展构建均成功。
原生 mock 在构建中运行，故障测试不主动触发硬件崩溃。

从 `E:\Pytorch_xfire_NGPU` 运行（硬件测试串行执行）：

```powershell
.\build_npu_session.ps1
.\build_torch_npu_device.ps1
.\run_torch_npu.ps1 -Script .\torch_npu_session_reuse_benchmark.py
.\run_torch_npu.ps1 -Script .\torch_npu_session_reuse_benchmark.py -ScriptArguments '--case','model-control'
.\run_torch_npu.ps1 -Session -Quick
.\run_torch_npu.ps1 -Device -Quick
.\run_torch_npu.ps1 -Training -Quick
.\run_torch_npu.ps1 -Script .\torch_npu_session_reuse_benchmark.py -ScriptArguments '--case','audit'
```

保存的 `native-script.py` 和 `model-script.py` 是本次采样时的源码，不能直接从结果目录
运行，因为它们以脚本所在目录定位工作区。当前根目录基准保留相同测量逻辑，新增离线
audit 入口；audit 核对本次采样快照指纹。重建 DLL 或重新采样后应保存到新的结果批次，
并更新对应采样源快照；不能用本次 audit 为另一个构建自动背书。

后续性能方向：继续测量逐次记录命令、Tensor/共享缓冲区分配及模型层间调度的占比，
再决定是否引入固定工作区或驱动支持的 mutable command list。
